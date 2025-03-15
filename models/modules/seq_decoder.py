import torch
from torch import nn
import torch.nn.functional as F

from .layers.transformer import *
from .utils.macro import *

def _make_batch_first(*args):
    # S, N, ... -> N, S, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)

class ConstEmbedding(nn.Module):
    """
    Learned constant embedding
    """
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.PE = PositionalEncodingLUT(d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src

class FCNet(nn.Module):
    """
    Fully connected network
    """
    def __init__(self, d_model, n_cmd, n_param, d_param=256):
        super().__init__()
        self.n_param = n_param
        self.d_param = d_param
        self.cmd_fcn = nn.Linear(d_model, n_cmd) # output command logits   
        self.param_fcn = nn.Linear(d_model, n_param * d_param) # output parameter logits

    def forward(self, x):
        S, N, _ = x.shape
        cmd_logits = self.cmd_fcn(x)  
        param_logits = self.param_fcn(x)  
        param_logits = param_logits.reshape(S, N, self.n_param, self.d_param) 
        return cmd_logits, param_logits

class SeqDecoder(nn.Module):
    """
    Sequence decoder
    """
    def __init__(
        self, 
        n_head, 
        n_layer, 
        d_model, 
        d_z, 
        d_ff, 
        dropout
    ):
        super(SeqDecoder, self).__init__()
        self.embedding = ConstEmbedding(d_model, MAX_PRIM + MAX_FEAT)

        self.decoder = TransformerDecoder(
            decoder_layer = TransformerDecoderLayerGlobalImproved(
                d_model = d_model, 
                d_global = d_z, 
                n_head = n_head, 
                d_ff = d_ff, 
                dropout = dropout,
            ),
            n_layer = n_layer, 
            norm =nn.LayerNorm(d_model)
        )
        
        self.fcn_primitive = FCNet(d_model, N_PRIM_COMMANDS+2, N_PRIM_PARAM, PARAM_DIM)
        self.fcn_feature = FCNet(d_model, N_FEAT_COMMANDS+2, N_FEAT_PARAM, PARAM_DIM)

    def forward(self, z, encode_mode=False):         # z [batch_size, 256]
        z = torch.unsqueeze(z, dim=0)
        if encode_mode: return _make_batch_first(z)

        src = self.embedding(z)   # src [seq_len, batch_size, 256]
        y = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None) # out [seq_len, batch_size, 256]
        y_p = y[:MAX_PRIM]
        y_f = y[MAX_PRIM:]

        prim_cmd, prim_param = self.fcn_primitive(y_p)
        output_primitive = (prim_cmd, prim_param)
        output_primitive = _make_batch_first(*output_primitive)

        feat_cmd, feat_param = self.fcn_feature(y_f)
        output_feature = (feat_cmd, feat_param)
        output_feature = _make_batch_first(*output_feature)

        res = {
            "prim_cmd": output_primitive[0],
            "prim_param": output_primitive[1],
            "feat_cmd": output_feature[0],
            "feat_param": output_feature[1]
        }
        return res