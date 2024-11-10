#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positional Encoding Module."""

import math

import torch


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnableFourierPosEnc(torch.nn.Module):
    """Learnable Fourier Features for Positional Encoding.

    See https://arxiv.org/pdf/2106.02795.pdf

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        gamma (float): init parameter for the positional kernel variance
            see https://arxiv.org/pdf/2106.02795.pdf.
        apply_scaling (bool): Whether to scale the input before adding the pos encoding.
        hidden_dim (int): if not None, we modulate the pos encodings with
            an MLP whose hidden layer has hidden_dim neurons.
    """

    def __init__(
        self,
        d_model,
        dropout_rate=0.0,
        max_len=5000,
        gamma=1.0,
        apply_scaling=False,
        hidden_dim=None,
    ):
        """Initialize class."""
        super(LearnableFourierPosEnc, self).__init__()

        self.d_model = d_model

        if apply_scaling:
            self.xscale = math.sqrt(self.d_model)
        else:
            self.xscale = 1.0

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.max_len = max_len

        self.gamma = gamma
        if self.gamma is None:
            self.gamma = self.d_model // 2

        assert (
            d_model % 2 == 0
        ), "d_model should be divisible by two in order to use this layer."
        self.w_r = torch.nn.Parameter(torch.empty(1, d_model // 2))
        self._reset()  # init the weights

        self.hidden_dim = hidden_dim
        if self.hidden_dim is not None:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_model, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, d_model),
            )

    def _reset(self):
        self.w_r.data = torch.normal(
            0, (1 / math.sqrt(self.gamma)), (1, self.d_model // 2)
        )

    def extend_pe(self, x):
        """Reset the positional encodings."""
        position_v = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1).to(x)

        cosine = torch.cos(torch.matmul(position_v, self.w_r))
        sine = torch.sin(torch.matmul(position_v, self.w_r))
        pos_enc = torch.cat((cosine, sine), -1)
        pos_enc /= math.sqrt(self.d_model)

        if self.hidden_dim is None:
            return pos_enc.unsqueeze(0)
        else:
            return self.mlp(pos_enc.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        pe = self.extend_pe(x)
        x = x * self.xscale + pe
        return self.dropout(x)


class LegacyRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
            reverse=True,
        )

    def forward(self, x):
        """Compute positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class StreamPositionalEncoding(torch.nn.Module):
    """Streaming Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(StreamPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.tmp = torch.tensor(0.0).expand(1, max_len)
        self.extend_pe(self.tmp.size(1), self.tmp.device, self.tmp.dtype)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, length, device, dtype):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= length:
                if self.pe.dtype != dtype or self.pe.device != device:
                    self.pe = self.pe.to(dtype=dtype, device=device)
                return
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, start_idx: int = 0):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x.size(1) + start_idx, x.device, x.dtype)
        x = x * self.xscale + self.pe[:, start_idx : start_idx + x.size(1)]
        return self.dropout(x)


class T5RelativePositionBias(torch.nn.Module):
    # Source: https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16
    def __init__(
        self,
        d_model,
        dropout_rate,
        bidirectional=True,
        num_buckets=32,
        max_distance=128,
        n_heads=12,
        **kwargs,
    ):
        super(T5RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = torch.nn.Embedding(
            self.num_buckets, self.n_heads
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(
                torch.long
            ) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            qlen, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            klen, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, x: torch.Tensor):
        return self.dropout(x), self.dropout(self.compute_bias(x.size(1), x.size(1)))
