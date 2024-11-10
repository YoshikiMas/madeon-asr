"""Decoder definition."""
from typing import Any, List, Tuple

import math
from functools import partial

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.state_spaces.mamba.mamba_mha import Mamba, Block, DecoderBlock
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.scorer_interface import BatchScorerInterface

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
try:
    from espnet2.asr.state_spaces.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
import logging

from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union
from torch import Tensor
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
import copy

@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

class MambaDecoderLayer(torch.nn.Module):
    """Mamba decoder layer module.
    
    """
    def __init__(
        self,
        d_model: int,
        ssm_cfg=None,
        cross_attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        # initializer_cfg=None,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.ssm_cfg = ssm_cfg
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.layer_idx = layer_idx

        if ssm_cfg is None:
            ssm_cfg = {}
        if cross_attn_cfg is None:
            cross_attn_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        self.attention_cls = partial(MultiHeadedAttention, **cross_attn_cfg,)

        self.norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )

        self.block = DecoderBlock(
            d_model,
            self.mixer_cls,
            self.attention_cls,
            norm_cls=self.norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        self.block.layer_idx = layer_idx

    def forward(self, hidden_states, memory, memory_mask=None, residual=None, mask=None, inference_params=None,):
        """Forward function."""
        hidden_states, residual = self.block(hidden_states,
                                             memory=memory,
                                             memory_mask=memory_mask,
                                             residual=residual,
                                             mask=mask,
                                             inference_params=inference_params,)

        return hidden_states, residual, mask


class MambaDecoder(AbsDecoder, BatchScorerInterface):
    """Mamba decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of hidden vector
        input_layer: input layer type
        dropinp: input dropout
        dropout: dropout parameter applied on every residual and every layer
        prenorm: pre-norm vs. post-norm
        n_layers: number of layers
        transposed: transpose inputs so each layer receives (batch, dim, length)
        tie_dropout: tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        n_repeat: each layer is repeated n times per stage before applying pooling
        layer: layer config, must be specified
        residual: residual config
        norm: normalization config (e.g. layer vs batch)
        pool: config for pooling layer per stage
        track_norms: log norms of each layer output
        drop_path: drop rate for stochastic depth
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        input_layer: str = "embed",
        ssm_cfg=None,
        cross_attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device=None,
        dtype=None,
        num_blocks: int = 6,

        dropout: float = 0.25,
        emb_dropout: float = 0.0,
        layer_drop_rate: float = 0.0,
        init_rescale: bool = False,
        # **kwargs,  # dummy args for compatibility
    ):
        # assert check_argument_types()
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = encoder_output_size
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.odim = vocab_size
        self.num_blocks = num_blocks
        self.initializer_cfg = initializer_cfg
        # self.dropout = dropout

        if input_layer == "embed":
            self.embed = torch.nn.Embedding(vocab_size, self.d_model)
            self.dropout_emb = torch.nn.Dropout(p=emb_dropout) if emb_dropout > 0 else torch.nn.Identity()
        elif input_layer == "embed_pos":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, self.d_model),
                PositionalEncoding(self.d_model, dropout),
            )
            self.dropout_emb = torch.nn.Identity()
        else:
            raise NotImplementedError
     

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.decoders = repeat(
            num_blocks,
            lambda lnum: MambaDecoderLayer(
                d_model=encoder_output_size,
                ssm_cfg=ssm_cfg,
                cross_attn_cfg=cross_attn_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                layer_idx=lnum,
            ),
            layer_drop_rate,
        )

        self.mixer_cls = partial(Mamba, layer_idx=num_blocks, **ssm_cfg, **factory_kwargs)
        self.norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        self.out_mamba = Block(
            encoder_output_size,
            self.mixer_cls,
            norm_cls=self.norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=True,
            )
        self.after_norm = LayerNorm(self.d_model) if not rms_norm else RMSNorm(self.d_model)

        self.output = torch.nn.Linear(self.d_model, vocab_size)

        if init_rescale:
            logging.info("Rescaling weights in MambaEncoder")
            self.espnet_initialization_fn()

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
        inference_params = InferenceParams(max_seqlen=x.size(1), max_batch_size=x.size(0))
        inference_params.key_value_memory_dict = self.allocate_inference_cache(1, device=x.device)
        return inference_params

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None, **kwargs):
        """Allocate inference cache."""
        cache = {
            i: layer.block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.decoders)
        }
        cache[len(self.decoders)] = self.out_mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return cache


    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        inference_params=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        x = self.dropout_emb(self.embed(ys_in_pad))

        residual = None
        for decoder in self.decoders:
            x, residual, _ = decoder(
                x,
                memory=memory,
                memory_mask=memory_mask,
                residual=residual
            )

        x, residual = self.out_mamba(x, residual)
        x = x + residual
        x = self.after_norm(x)

        decoded = self.output(x)
        return decoded, ys_in_lens

    def score(self, ys, state, x):
        raise NotImplementedError

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        # MEMO: WIP

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
        x = self.embed(ys[:, -1:])
        # breakpoint()

        if not isinstance(states[0], InferenceParams):
            inference_params = InferenceParams(max_seqlen=xs.size(1), max_batch_size=n_batch)
            inference_params.key_value_memory_dict = copy.deepcopy(states[0])
        else:
            # merge states
            inference_params = copy.deepcopy(states[0])
            for k in inference_params.key_value_memory_dict.keys():
                # conv_states = torch.cat(
                #     [state.key_value_memory_dict[k][0] * (1 + 0.1*i) for i, state in enumerate(states)], dim=0
                # ) 
                # ssm_states = torch.cat(
                #     [state.key_value_memory_dict[k][1]* (1 + 0.1*i) for i, state in enumerate(states)], dim=0
                # )
                conv_states = torch.cat(
                    [state.key_value_memory_dict[k][0] for state in states], dim=0
                ) 
                ssm_states = torch.cat(
                    [state.key_value_memory_dict[k][1] for state in states], dim=0
                )
                inference_params.key_value_memory_dict[k] = (conv_states, ssm_states)

        residual = None
        for decoder in self.decoders:
            x, residual, _ = decoder(
                x,
                memory=xs,
                memory_mask=None,  # Assume no padding in the encoder output
                residual=residual,
                inference_params=inference_params,
            )

        x, residual = self.out_mamba(x, residual, inference_params=inference_params)
        x = x + residual
        x = self.after_norm(x)

        logp = self.output(x).log_softmax(dim=-1).squeeze(1)

        inference_params.seqlen_offset += 1
        # breakpoint()

        # decompose states for beam search
        state_list = [InferenceParams(max_seqlen=1, max_batch_size=n_batch,seqlen_offset=inference_params.seqlen_offset) for _ in range(n_batch)]
        for b in range(n_batch):
            # states[b].seqlen_offset = inference_params.seqlen_offset
            for k in inference_params.key_value_memory_dict.keys():
                conv_states = inference_params.key_value_memory_dict[k][0][b].unsqueeze(0)
                ssm_states = inference_params.key_value_memory_dict[k][1][b].unsqueeze(0)
                state_list[b].key_value_memory_dict[k] = (conv_states, ssm_states)

        return logp, state_list
