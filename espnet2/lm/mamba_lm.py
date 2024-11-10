from typing import Any, Dict, List, Tuple, Optional, Sequence, Union
import math
import copy
from functools import partial

import logging
import torch
import torch.nn as nn

from espnet2.lm.abs_model import AbsLM

from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.asr.decoder.mamba_decoder import InferenceParams
from espnet2.asr.encoder.mamba_encoder import MambaEncoderLayer, SerialBiMambaEncoderLayer

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

from espnet2.layers.mask_along_axis import MaskAlongAxis, MaskAlongAxisVariableMaxWidth

try:
    from espnet2.asr.state_spaces.ops.triton.layernorm import (
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn,
    )
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def batch_flip(x, index):
    return x.flatten().index_select(dim=0, index=index).view(x.shape)

def compute_flip_fn_and_mask(sp_token_idx: int, input_seq: torch.Tensor, d_model: int):
    """Compute flip function and mask for partially-bidirectional model (i.e. prefixLM).
    
    Args:
        sp_token_idx (int): The index of the special token <generatetext>
        input_seq (torch.Tensor): The input sequence (B, L)
        d_model (int): The dimension of the input sequence
    
    Returns:
        Tuple[None, List[Callable]]: Tuple of mask and flip function
    """
    mask = None
    bs, seq_len = input_seq.shape

    flip_idx = torch.arange(bs*seq_len*d_model, device=input_seq.device)
    flip_idx_vec = flip_idx.view(bs, d_model, seq_len)

    for batch, token in enumerate(torch.unbind(input_seq)):
        sp_token = torch.where(token == sp_token_idx)[0]
        if len(sp_token) == 1:
            flip_idx_vec[batch, :, 1:sp_token] = torch.flip(flip_idx_vec[batch, :, 1:sp_token], dims=[-1])

        elif len(sp_token) == 0:
            flip_idx_vec[batch, :, 1:] = torch.flip(flip_idx_vec[batch, :, 1:], dims=[-1])

        else:
            raise ValueError(
                f"The numbe of `sp_token` (={len(sp_token)}) is incorrect. It shoud be 1 in training and 0 in inference"
            )
    flip_fn = partial(batch_flip, index=flip_idx)
    return mask, flip_fn

def compute_flip_fn_and_mask_parallel(sp_token_idx, input_seq, d_xz):
    mask = (input_seq != 0)[:, None, :]
    bs, seq_len = input_seq.shape
    flip_idxs = [
        torch.arange(bs*d_xz*seq_len, device=input_seq.device),
        torch.arange(bs*d_xz//2*seq_len, device=input_seq.device),
    ]

    flip_idx_before_ssm = flip_idxs[0].view(bs, d_xz, seq_len)
    flip_idx_after_ssm = flip_idxs[1].view(bs, d_xz//2, seq_len)

    for batch, token in enumerate(torch.unbind(input_seq)):
        sp_token = torch.where(token == sp_token_idx)[0]
        if len(sp_token) == 1:
            flip_idx_before_ssm[batch, :, 1:sp_token] = torch.flip(flip_idx_before_ssm[batch, :, 1:sp_token], dims=[-1])
            flip_idx_after_ssm[batch, :, 1:sp_token] = torch.flip(flip_idx_after_ssm[batch, :, 1:sp_token], dims=[-1])

        elif len(sp_token) == 0:
            flip_idx_before_ssm[batch, :, 1:] = torch.flip(flip_idx_before_ssm[batch, :, 1:], dims=[-1])
            flip_idx_after_ssm[batch, :, 1:] = torch.flip(flip_idx_after_ssm[batch, :, 1:], dims=[-1])

        else:
            raise ValueError(
                f"The numbe of `sp_token` (={len(sp_token)}) is incorrect. It shoud be 1 in training and 0 in inference"
            )

    flip_fn = [partial(batch_flip, index=flip_idx) for flip_idx in flip_idxs]
    return mask, flip_fn

class MambaLM(AbsLM):
    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        vocab_size: int,
        ssm_cfg: Dict[str, Any],
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg: Dict[str, Any] = None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        init_rescale: bool = False,
        tie_embedding: bool = False,
        layer_drop_rate: float = 0.0,
        head_bias: bool =False,
        apply_mask: bool = False,
        mask_width_range: Optional[Union[int, Sequence[int]]] = None,
        mask_width_ratio_range: Optional[Union[float, Sequence[float]]] = None,
        num_mask: int = 0,
        prefix_bidir: bool = False,
        prefix_bidir_mode: str = "parallel",
        sp_token_idx: int = -1,
        **factory_kwargs,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.num_blocks = num_blocks
        self.initializer_cfg = initializer_cfg
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        if (
            apply_mask
            and (mask_width_range is not None)
            and (mask_width_ratio_range is not None)
        ):
            raise ValueError(
                'Either one of "mask_width_range" or '
                '"mask_width_ratio_range" can be used'
            )

        if prefix_bidir and sp_token_idx < 0:
            raise ValueError(
                f"`sp_token_idx` (={sp_token_idx}) should be greater than 0 if `prefix_bidir` is true"
            )

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        if prefix_bidir:
            if prefix_bidir_mode == "serial":
                ssm_cfg["bidirectional"] = False
                encoder_layer_cls = SerialBiMambaEncoderLayer
            if prefix_bidir_mode == "parallel":
                ssm_cfg["bidirectional"] = True
                encoder_layer_cls = MambaEncoderLayer
        else:
            encoder_layer_cls = MambaEncoderLayer
    
        self.blocks = repeat(
            num_blocks,
            lambda lnum: encoder_layer_cls(
                d_model,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                layer_idx=lnum,
            ),
            layer_drop_rate,
        )

        self.after_norm = LayerNorm(d_model) if not rms_norm else RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=head_bias, **factory_kwargs)

        # Initialize weights and apply final processing
        if init_rescale:
            logging.info("Rescaling weights in MambaEncoder")
            self.espnet_initialization_fn()

        if tie_embedding:
            self.tie_weights()

        self.sp_token_idx = sp_token_idx
        self.prefix_bidir = prefix_bidir
        self.prefix_bidir_mode = prefix_bidir_mode
        self.d_xz = 2 * ssm_cfg["expand"] * d_model
        self.d_model = d_model

        if apply_mask:
            if mask_width_range is not None:
                self.aug_mask = MaskAlongAxis(
                    dim="time",
                    mask_width_range=mask_width_range,
                    num_mask=num_mask,
                )
            elif mask_width_ratio_range is not None:
                self.aug_mask = MaskAlongAxisVariableMaxWidth(
                    dim="time",
                    mask_width_ratio_range=mask_width_ratio_range,
                    num_mask=num_mask,
                )
        else:
            self.aug_mask = None

    def tie_weights(self):
        self.lm_head.weight = self.embedding.weight

    def espnet_initialization_fn(self):
        self.apply(
            partial(
                self.init_weights,
                n_layer=self.num_blocks,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )

    # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
    def init_weights(
        self,
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    def init_state(self, x: torch.Tensor):
        """Initialize state."""
        inference_params = InferenceParams(
            max_seqlen=x.size(1), max_batch_size=x.size(0)
        )
        inference_params.key_value_memory_dict = self.allocate_inference_cache(
            1, device=x.device
        )
        return inference_params

    def allocate_inference_cache(
        self, batch_size, max_seqlen=None, dtype=None, **kwargs
    ):
        """Allocate inference cache."""
        cache = {
            i: layer.block.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.blocks)
        }
        return cache

    def forward(
        self, input: torch.Tensor, hidden: None, inference_params=None
    ) -> Tuple[torch.Tensor, None]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embedding(input)
        if self.aug_mask is not None and self.training:
            # NOTE (Y. Masuyama): the second x is a dummy length input
            x, _ = self.aug_mask(x, x)

        # NOTE (Y. Masuyama): This is the case for partially-bidirectional Mamba, i.e., prefixLM
        if self.prefix_bidir:
            if self.prefix_bidir_mode == "serial":
                mask, flip_fn = compute_flip_fn_and_mask(
                    sp_token_idx=self.sp_token_idx,
                    input_seq=input,
                    d_model=self.d_model
                )
            elif self.prefix_bidir_mode == "parallel":
                mask, flip_fn = compute_flip_fn_and_mask_parallel(
                    sp_token_idx=self.sp_token_idx,
                    input_seq=input,
                    d_xz=self.d_xz,
                )
        # NOTE (Y. Masuyama): This is the case for unidirectional Mamba, i.e., decoder-only LM
        else:
            mask, flip_fn = None, None

        residual = None
        for block in self.blocks:
            x, residual, _ = block(
                x,
                residual=residual,
                mask=mask,
                inference_params=inference_params,
                flip_fn=flip_fn,
            )

        if residual is not None:
            x = x + residual
        x = self.after_norm(x)
        y = self.lm_head(x)
        return y, None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        raise NotImplementedError

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        n_batch = len(ys)
        x = self.embedding(ys[:, -1:])

        if not isinstance(states[0], InferenceParams):
            inference_params = InferenceParams(
                max_seqlen=1, max_batch_size=n_batch
            )
            inference_params.key_value_memory_dict = copy.deepcopy(states[0])
        else:
            inference_params = copy.deepcopy(states[0])
            for k in inference_params.key_value_memory_dict.keys():
                conv_states = torch.cat(
                    [state.key_value_memory_dict[k][0] for state in states], dim=0
                )
                ssm_states = torch.cat(
                    [state.key_value_memory_dict[k][1] for state in states], dim=0
                )

                if not self.prefix_bidir:
                    inference_params.key_value_memory_dict[k] = (conv_states, ssm_states)

                else:
                    conv_states_pb = torch.cat(
                        [state.key_value_memory_dict[k][2] for state in states], dim=0
                    )
                    ssm_states_pb = torch.cat(
                        [state.key_value_memory_dict[k][3] for state in states], dim=0
                    )
                    inference_params.key_value_memory_dict[k] = (
                        conv_states, ssm_states, conv_states_pb, ssm_states_pb
                    )

        # NOTE (Y. Masuyama): This function is assumed to handle only text tokens
        # Hence, even in the case of prefix LM, we do not need `flip_fn`
        mask, flip_fn = True, None
        residual = None
        for block in self.blocks:
            x, residual, _ = block(
                x,
                residual=residual,
                mask=mask,
                inference_params=inference_params,
                flip_fn=flip_fn,
            )

        if residual is not None:
            x = x + residual
        x = self.after_norm(x)

        logp = self.lm_head(x).log_softmax(dim=-1).squeeze(1)

        inference_params.seqlen_offset += 1

        # decompose states for beam search
        state_list = [
            InferenceParams(
                max_seqlen=1,
                max_batch_size=n_batch,
                seqlen_offset=inference_params.seqlen_offset,
            )
            for _ in range(n_batch)
        ]
        for b in range(n_batch):
            for k in inference_params.key_value_memory_dict.keys():
                conv_states = inference_params.key_value_memory_dict[k][0][b].unsqueeze(
                    0
                )
                ssm_states = inference_params.key_value_memory_dict[k][1][b].unsqueeze(
                    0
                )
                if not self.prefix_bidir:
                    state_list[b].key_value_memory_dict[k] = (conv_states, ssm_states)

                else:
                    conv_states_pb = inference_params.key_value_memory_dict[k][2][b].unsqueeze(
                        0
                    )
                    ssm_states_pb = inference_params.key_value_memory_dict[k][3][b].unsqueeze(
                        0
                    )
                    state_list[b].key_value_memory_dict[k] = (conv_states, ssm_states, conv_states_pb, ssm_states_pb)

        return logp, state_list
