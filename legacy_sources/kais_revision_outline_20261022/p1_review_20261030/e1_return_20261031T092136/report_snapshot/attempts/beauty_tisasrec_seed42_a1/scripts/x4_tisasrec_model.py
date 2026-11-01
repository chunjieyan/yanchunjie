#!/usr/bin/env python3
"""Standalone pinned RecBole-TRM TiSASRec adaptation for the X4 protocol.

This module defines a model only. It does not construct temporal matrices,
load data/checkpoints, train, infer, set RNG seeds, or write files on import.
The caller provides a visible-history-only integer time matrix. Scores include
padding index zero; CE/evaluation masking belongs to the audited caller.
"""
# MIT License
#
# Copyright (c) 2020 RUCAIBox
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import math

import torch
from torch import nn
import torch.nn.functional as fn

UPSTREAM_REPOSITORY = 'https://github.com/RUCAIBox/RecBole-TRM'
UPSTREAM_COMMIT = '18b47f8ff98980bed2f2113b398e55b484e748b9'
SOURCE_SHA256 = {'LICENSE': '2c33dad5fede86236b54db22e0be64e72095d151f324bcadd354ab88f40c52fe', 'recbole/model/transformer_recommender/tisasrec.py': 'e749713fdf2f7a75f62a68353e5013558b3359662594da0f71cc81b2c2a6a4e6', 'recbole/model/transformer_layers.py': 'f59b3b94431dbfeac257e8786e32040a6f51f6df929b3f4bceb72faeab2c4179', 'recbole/model/layers.py': '992f0aeb103020c398d95620687ca48776dc41119623d4800650fe585438b8b9', 'recbole/model/abstract_recommender.py': 'f63c05eac8606495a58f5866ce93a057dc810da7d0d50dd16cfac2b3a4ff8a12', 'recbole/properties/model/TiSASRec.yaml': '001df1caeaeb32de46c575b81a23dce5c5a770e28e54491696d74f99acbb24f6'}
ADAPTATION_SPEC = {
    'version': 'x4_tisasrec_v1', 'name': 'TiSASRecModel',
    'source': 'MIT RecBole-TRM TiSASRec; pinned commit, not original-paper numerical reproduction',
    'padding': 'valid-first right padding; item index 0; gather length-1',
    'time_protocol': 'Caller: only already-truncated visible history; minimum positive visible chronological gap, or 1; floor(abs(ti-tj)/scale), clip 256; zero for same bucket and padding pairs; no target time.',
    'preserved': 'TimeAware attention/layer/encoder and FeedForward AST; initialization; upstream forward computation; inherited causal mask and gather.',
    'score_all': 'Unmasked full-catalog logits including padding index 0; caller masks index 0 and evaluation seen items except target.',
}

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TimeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(TimeAwareMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        absolute_pos_K_ = self.transpose_for_scores(absolute_pos_K).permute(0, 2, 3, 1)
        absolute_pos_V_ = self.transpose_for_scores(absolute_pos_V).permute(0, 2, 1, 3)
        time_matrix_emb_K_ = self.transpose_for_scores(time_matrix_emb_K).permute(0, 3, 1, 2, 4) # [B, n_heads, L, L, D/n_heads]
        time_matrix_emb_V_ = self.transpose_for_scores(time_matrix_emb_V).permute(0, 3, 1, 2, 4) # [B, n_heads, L, L, D/n_heads]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores_pos = torch.matmul(query_layer, absolute_pos_K_)
        attention_scores_time = torch.matmul(time_matrix_emb_K_, query_layer.unsqueeze(-1)).squeeze(-1) # [B, n_heads, L, L, 1] -> [B, n_heads, L, L]

        attention_scores = attention_scores + attention_scores_pos + attention_scores_time

        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_pos = torch.matmul(attention_probs, absolute_pos_V_)
        context_layer_time = torch.matmul(attention_probs.unsqueeze(-2), time_matrix_emb_V_).squeeze(-2)
        context_layer = context_layer + context_layer_pos + context_layer_time

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TimeAwareTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TimeAwareTransformerLayer, self).__init__()
        self.multi_head_attention = TimeAwareMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V):
        attention_output = self.multi_head_attention(hidden_states, attention_mask, absolute_pos_K, absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
        feedforward_output = self.feed_forward(attention_output)

        return feedforward_output


class TimeAwareTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TimeAwareTransformerEncoder, self).__init__()
        layer = TimeAwareTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, 
                    attention_mask, 
                    absolute_pos_K,
                    absolute_pos_V,
                    time_matrix_emb_K,
                    time_matrix_emb_V,
                    output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, absolute_pos_K,
                                        absolute_pos_V, time_matrix_emb_K, time_matrix_emb_V)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TiSASRecModel(nn.Module):
    """Fixed-protocol independent API; num_items excludes padding index zero.

    Default architecture: H64, two heads/layers, FFN256, GELU, dropout .2,
    time span 256, initialization std .02 and LayerNorm epsilon 1e-12.
    All caller tensors use torch.long on one device. Histories are nonempty,
    valid-first and right padded. time_matrix is [B,L,L], visible-history
    intervals only, with padding pairs zero. No target timestamp is accepted.
    """

    def __init__(self, num_items, hidden_size=64, max_len=50, num_heads=2,
                 num_layers=2, dropout=0.2, time_span=256):
        super(TiSASRecModel, self).__init__()
        values = {'num_items': num_items, 'hidden_size': hidden_size, 'max_len': max_len,
                  'num_heads': num_heads, 'num_layers': num_layers, 'time_span': time_span}
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in values.values()):
            raise ValueError('TiSASRec integer dimensions must be positive integers')
        if hidden_size % num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be finite and in [0,1)')
        self.num_items = num_items
        self.n_items = num_items + 1
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.hidden_size = hidden_size
        self.inner_size = hidden_size * 4
        self.max_seq_length = max_len
        self.max_len = max_len
        self.hidden_dropout_prob = float(dropout)
        self.attn_dropout_prob = float(dropout)
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12
        self.time_span = time_span
        self.initializer_range = 0.02
        self.loss_type = 'CE'

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.absolute_pos_K_embedding = nn.Embedding(self.max_seq_length, self.hidden_size, padding_idx=0)
        self.absolute_pos_V_embedding = nn.Embedding(self.max_seq_length, self.hidden_size, padding_idx=0)
        self.time_matrix_emb_K_embedding = nn.Embedding(self.time_span+1, self.hidden_size, padding_idx=0)
        self.time_matrix_emb_V_embedding = nn.Embedding(self.time_span+1, self.hidden_size, padding_idx=0)
        self.ti_trm_encoder = TimeAwareTransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads, hidden_size=self.hidden_size,
            inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob, hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _validate_inputs(self, item_seq, lengths, time_matrix):
        if item_seq.ndim != 2 or not 1 <= item_seq.shape[1] <= self.max_seq_length or item_seq.shape[0] < 1:
            raise ValueError('item_seq must have shape [B,L], B>=1 and 1<=L<=max_len')
        if lengths.shape != (item_seq.shape[0],) or time_matrix.shape != (item_seq.shape[0], item_seq.shape[1], item_seq.shape[1]):
            raise ValueError('lengths or time_matrix shape does not match item_seq')
        if item_seq.dtype != torch.long or lengths.dtype != torch.long or time_matrix.dtype != torch.long:
            raise ValueError('item_seq, lengths and time_matrix must use torch.long')
        if item_seq.device != lengths.device or item_seq.device != time_matrix.device:
            raise ValueError('item_seq, lengths and time_matrix must share one device')
        if torch.any((lengths < 1) | (lengths > item_seq.shape[1])).item():
            raise ValueError('Every visible history must be nonempty and fit the sequence')
        if torch.any((item_seq < 0) | (item_seq > self.num_items)).item():
            raise ValueError('Item index outside the frozen catalog')
        valid = torch.arange(item_seq.shape[1], device=item_seq.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(item_seq.ne(0), valid):
            raise ValueError('Histories must be valid-first with right padding only')
        if torch.any((time_matrix < 0) | (time_matrix > self.time_span)).item():
            raise ValueError('Time intervals must be clipped integers in [0,time_span]')
        valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
        if torch.any(time_matrix.masked_select(~valid_pairs) != 0).item():
            raise ValueError('Time-matrix padding pairs must be zero')

    def forward(self, item_seq, lengths, time_matrix):
        self._validate_inputs(item_seq, lengths, time_matrix)
        return self._forward_upstream(item_seq, lengths, time_matrix)

    def score_all(self, item_seq, lengths, time_matrix):
        seq_output = self.forward(item_seq, lengths, time_matrix)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_upstream(self, item_seq, item_seq_len, time_matrix):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        absolute_pos_K = self.absolute_pos_K_embedding(position_ids) # [B, L, D]
        absolute_pos_V = self.absolute_pos_V_embedding(position_ids)

        time_matrix_emb_K = self.time_matrix_emb_K_embedding(time_matrix) # [B, L, L, D]
        time_matrix_emb_V = self.time_matrix_emb_V_embedding(time_matrix)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        absolute_pos_K = self.dropout(absolute_pos_K)
        absolute_pos_V = self.dropout(absolute_pos_V)
        time_matrix_emb_K = self.dropout(time_matrix_emb_K)
        time_matrix_emb_V = self.dropout(time_matrix_emb_V)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.ti_trm_encoder(
                                    input_emb, 
                                    extended_attention_mask,
                                    absolute_pos_K,
                                    absolute_pos_V,
                                    time_matrix_emb_K,
                                    time_matrix_emb_V,
                                    output_all_encoded_layers=True
                                    )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        return output

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask
