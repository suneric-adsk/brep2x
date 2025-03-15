import copy
import torch
from torch import nn
from torch.nn import functional as F
from fairseq import utils

from .attention import MultiheadAttention

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, n_layer, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, n_layer)
        self.norm = norm

    def forward(
        self, 
        tgt, 
        memory, 
        memory2=None, 
        tgt_mask=None,
        memory_mask=None, 
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        output = tgt
        for mod in self.layers:
            output = mod(
                output, 
                memory, 
                memory2=memory2, 
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayerGlobalImproved(nn.Module):
    def __init__(self, d_model, d_global, n_head, d_ff=2048, dropout=0.1, act="relu", d_global2=None):
        super(TransformerDecoderLayerGlobalImproved, self).__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear_global = nn.Linear(d_global, d_model)
        if d_global2 is not None:
            self.linear_global2 = nn.Linear(d_global2, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation_fn = utils.get_activation_fn(act)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerGlobalImproved, self).__setstate__(state)

    def forward(self, tgt, memory, memory2=None, tgt_mask=None, tgt_key_padding_mask=None, *args, **kwargs):
        tgt1 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.linear_global(memory)
        tgt = tgt + self.dropout2(tgt2)  # implicit broadcast

        if memory2 is not None:
            tgt2_2 = self.linear_global2(memory2)
            tgt = tgt + self.dropout2_2(tgt2_2)

        tgt1 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation_fn(self.linear1(tgt1))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class PositionalEncodingLUT(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)
