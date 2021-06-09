r"""Functional interface"""
from __future__ import division

import torch
import warnings

from torch import nn


def diff_multi_head_attention_forward(query,
                                      key,
                                      value,
                                      pe,
                                      embed_dim_to_check,
                                      num_heads,
                                      in_proj_weight,
                                      in_proj_bias,
                                      bias_k,
                                      bias_v,
                                      add_zero_attn,
                                      dropout_p,
                                      out_proj_weight,
                                      out_proj_bias,
                                      training=True,
                                      key_padding_mask=None,
                                      need_weights=True,
                                      attn_mask=None,
                                      use_separate_proj_weight=False,
                                      q_proj_weight=None,
                                      k_proj_weight=None,
                                      v_proj_weight=None,
                                      static_k=None,
                                      static_v=None
                                      ):

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by \
            num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = nn.functional.linear(query, in_proj_weight,
                                           in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and
                # in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = nn.functional.linear(query, q_proj_weight_non_opt,
                                     in_proj_bias[0:embed_dim])
            k = nn.functional.linear(key, k_proj_weight_non_opt,
                                     in_proj_bias[embed_dim:(embed_dim * 2)])
            v = nn.functional.linear(value, v_proj_weight_non_opt,
                                     in_proj_bias[(embed_dim * 2):])
        else:
            q = nn.functional.linear(query, q_proj_weight_non_opt,
                                     in_proj_bias)
            k = nn.functional.linear(key, k_proj_weight_non_opt,
                                     in_proj_bias)
            v = nn.functional.linear(value, v_proj_weight_non_opt,
                                     in_proj_bias)
    k = q
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                      torch.zeros((attn_mask.size(0), 1),
                                                  dtype=attn_mask.dtype,
                                                  device=attn_mask.device)],
                                      dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((
                            key_padding_mask.size(0), 1),
                            dtype=key_padding_mask.dtype,
                            device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:],
                       dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:],
                       dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros(
                    (attn_mask.size(0), 1),
                    dtype=attn_mask.dtype,
                    device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros(
                        (key_padding_mask.size(0), 1),
                        dtype=key_padding_mask.dtype,
                        device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len,
                                                src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads,
                                                       tgt_len, src_len)

    pe = torch.repeat_interleave(pe, repeats=num_heads, dim=0)
    # numerical stability
    max_val = attn_output_weights.max(dim=-1, keepdim=True)[0]
    attn_output_weights = torch.exp(attn_output_weights - max_val)
    attn_output_weights = attn_output_weights * pe
    attn_output_weights = attn_output_weights / attn_output_weights.sum(
        dim=-1, keepdim=True).clamp(min=1e-6)
    attn_output_weights = nn.functional.dropout(attn_output_weights,
                                                p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(
            tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight,
                                       out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len,
                                                       src_len)
        # return attn_output, attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class DiffMultiheadAttention(nn.modules.activation.MultiheadAttention):
    def forward(self, query, key, value, pe, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        if hasattr(
                self, '_qkv_same_embed_dim'
                ) and self._qkv_same_embed_dim is False:
            return diff_multi_head_attention_forward(
                    query, key, value, pe, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.bias_k,
                    self.bias_v, self.add_zero_attn, self.dropout,
                    self.out_proj.weight, self.out_proj.bias,
                    training=self.training, key_padding_mask=key_padding_mask,
                    need_weights=need_weights, attn_mask=attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight,
                    k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttentio, module has benn implemented. \
                        Please re-train your model with the new module',
                              UserWarning)
            return diff_multi_head_attention_forward(
                    query, key, value, pe, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.bias_k,
                    self.bias_v, self.add_zero_attn, self.dropout,
                    self.out_proj.weight, self.out_proj.bias,
                    training=self.training, key_padding_mask=key_padding_mask,
                    need_weights=need_weights, attn_mask=attn_mask)


class DiffTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_norm=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = DiffMultiheadAttention(d_model, nhead,
                                                dropout=dropout, bias=False)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        self.scaling = None

    def forward(self, src, pe, degree=None, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, pe, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        if degree is not None:
            src2 = degree.transpose(0, 1).contiguous().unsqueeze(-1) * src2
        else:
            if self.scaling is None:
                self.scaling = 1. / pe.diagonal(dim1=1, dim2=2).max().item()
            src2 = (self.scaling * pe.diagonal(dim1=1, dim2=2)).transpose(0, 1).contiguous().unsqueeze(-1) * src2
        src = src + self.dropout1(src2)
        if self.batch_norm:
            bsz = src.shape[1]
            src = src.view(-1, src.shape[-1])
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.batch_norm:
            src = src.view(-1, bsz, src.shape[-1])
        return src

