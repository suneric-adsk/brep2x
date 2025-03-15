import torch
import torch.nn as nn
from typing import Optional, Tuple

from fairseq.modules import FairseqDropout, LayerNorm
from .layers.multihead_attention import MultiheadAttention
from .layers.graph_encoders import GraphEncoderLayer, GraphNodeFeature, GraphAttnBias

def init_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class BrepEncoder(nn.Module):
    def __init__(
        self,
        n_indeg: int, 
        n_outdeg: int,
        n_spatial: int,
        n_edge_dist: int,
        edge_type: str,
        mhm_dist: int,
        n_layer: int = 4,
        d_embedding: int = 128,
        d_ff_embedding: int = 128,
        n_head: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        act_dropout: float = 0.1,
        act_fn: str = "gelu",
    ):
        super().__init__()

        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        
        self.graph_node_feature = GraphNodeFeature(
            n_indeg=n_indeg,
            n_outdeg=n_outdeg,
            d_hidden=d_embedding,
            n_layer=n_layer,
        )

        self.graph_attn_bias = GraphAttnBias(
            n_head=n_head,
            d_hidden=d_embedding,
            n_spatial=n_spatial,
            n_edge_dist=n_edge_dist,
            edge_type=edge_type,
            mhm_dist=mhm_dist,
            n_layer=n_layer,
        )
        
        self.tanh = nn.Tanh()

        self.emb_layer_norm = LayerNorm(d_embedding, export=False)

        self.layers = nn.ModuleList([
            GraphEncoderLayer(
                d_embedding=d_embedding,
                d_ff_embedding=d_ff_embedding,
                n_head=n_head,
                dropout=dropout,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
                act_fn=act_fn,
            ) for _ in range(n_layer)
        ])

        # Apply initialization of model params after building the model
        self.apply(init_params)

    def forward(
        self,
        batch_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        padding_mask = batch_data["padding_mask"]
        n_graph, n_node = padding_mask.size()[:2]
        
        if token_embeddings is not None:
            x = token_embeddings
        else:
            x, node_feat = self.graph_node_feature(batch_data["node_data"],
                                        batch_data["face_area"],
                                        batch_data["face_type"],
                                        batch_data["in_degree"], 
                                        batch_data["out_degree"], 
                                        batch_data["padding_mask"])  
            # x[n_graph, max_node, embedding_dim]

        if perturb is not None:
            x[:, 1:, :] += perturb

        attn_bias = self.graph_attn_bias(batch_data["attn_bias"], 
                                         batch_data["spatial_pos"],
                                         batch_data["d2_dist"],
                                         batch_data["a3_dist"],
                                         batch_data["edge_data"],
                                         batch_data["edge_path"], 
                                         batch_data["edge_padding_mask"],
                                         batch_data["graph"],
                                         node_feat
                                         )  #attn_bias[n_graph, n_head, max_node_num, max_node_num]

        # compute padding mask. This is needed for multi-head attention
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1)

        x = self.emb_layer_norm(x)  #torch.nn.LayerNorm(embedding_dim)  x [batch_size, max_node, embedding_dim]
        x = self.dropout_module(x)
        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) # x [max_node, batch_size, embedding_dim]

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, attn_padding_mask=padding_mask, attn_mask=attn_mask, attn_bias=attn_bias)
            if not last_state_only:
                inner_states.append(x)

        graph_rep = x[0, :, :]      
        graph_rep = self.tanh(graph_rep)

        if last_state_only:
            inner_states = [x]

        return inner_states, graph_rep
