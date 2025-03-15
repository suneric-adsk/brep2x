import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout

from .multihead_attention import MultiheadAttention
from .feature_encoders import SurfaceEncoder, CurveEncoder, _EdgeConv

def init_params(module, n_layer):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layer))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, n_indeg, n_outdeg, n_layer, d_hidden):
        super(GraphNodeFeature, self).__init__()
        self.d_hidden = d_hidden
        # node_feature encode
        self.face_geom_encoder = SurfaceEncoder(in_channels=7, output_dims=d_hidden)
        self.face_area_encoder = nn.Embedding(2049, d_hidden, padding_idx=0)
        self.face_type_encoder = nn.Embedding(7, d_hidden, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(n_indeg, d_hidden, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(n_outdeg, d_hidden, padding_idx=0)
        self.graph_token = nn.Embedding(1, d_hidden)

        self.apply(lambda module: init_params(module, n_layer=n_layer))

    def forward(self, x, face_area, face_type, in_degree, out_degree, padding_mask):
        # x [total_node_num, U_grid, V_grid, pnt_feature]
        # padding_mask [batch_size, max_node_num]
        n_graph, n_node = padding_mask.size()[:2]
        node_pos = torch.where(padding_mask == False)
        x = x.permute(0, 3, 1, 2)
        x = self.face_geom_encoder(x)  # [total_nodes, n_hidden]

        face_area = self.face_area_encoder(face_area)  # [total_nodes, n_hidden]
        face_type = self.face_type_encoder(face_type)  # [total_nodes, n_hidden]

        in_degree = self.in_degree_encoder(in_degree)
        out_degree = self.out_degree_encoder(out_degree)

        face_feature = torch.zeros([n_graph, n_node, self.d_hidden], device=x.device, dtype=x.dtype)
        face_feature[node_pos] = x[:] + face_area[:] + face_type[:] + in_degree[:] + out_degree[:]  
        # [total_nodes, n_hidden]->[n_graph, max_node_num, n_hidden] 空节点用0.0填充
        # 增加一个全局虚拟节点 [n_graph, 1, n_hidden]
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        # [n_graph, max_node_num+1, n_hidden]
        graph_node_feature = torch.cat([graph_token_feature, face_feature], dim=1)  
        return graph_node_feature, x
    

class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(self, n_head, n_layer, n_spatial, n_edge_dist, edge_type, mhm_dist, d_hidden):
        super(GraphAttnBias, self).__init__()
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.mhm_dist = mhm_dist

        # spatial_feature encode
        self.spatial_pos_encoder = nn.Embedding(n_spatial, n_head, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, n_head)

        self.d2_pos_encoder = nn.Linear(32, n_head)
        self.a3_pos_encoder = nn.Linear(32, n_head)

        # edge_feature encode
        self.curv_encoder = CurveEncoder(in_channels=7, output_dims=n_head)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(n_edge_dist * n_head * n_head, 1)
            self.node_cat = _EdgeConv(edge_feats = n_head, out_feats = n_head, node_feats = d_hidden)
        self.apply(lambda module: init_params(module, n_layer=n_layer))

    def forward(self, attn_bias, spatial_pos, d2_dist, a3_dist, edge_data, edge_path, edge_padding_mask, graph, node_data):  # node_data [total_nodes, embedding_dim]
        n_graph, n_node = edge_path.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.n_head, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1] 描述每一头注意力下各节点之间的关系矩阵

        # spatial_pos 空间编码------------------------------------------------------------------------------------------------------------
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)  # spatial_pos_bias[n_graph, n_node, n_node, n_head]
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)  # spatial_pos_bias[n_graph, n_head, n_node, n_node]
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here 设置全局虚拟节点到其他节点的距离
        t = self.graph_token_virtual_distance.weight.view(1, self.n_head, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        # spatial_pos 空间编码------------------------------------------------------------------------------------------------------------

        # 欧氏空间编码
        # 在空间编码中增加面-面之间的D2距离----------------------------------------------------------------------------------------------------
        d2_pos_bias = self.d2_pos_encoder(d2_dist)  # [n_graph, n_node, n_node, 32] -> [n_graph, n_node, n_node, n_head]
        d2_pos_bias = d2_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + d2_pos_bias

        # 在空间编码中增加面-面之间的角度编码----------------------------------------------------------------------------------------------------
        a3_pos_bias = self.a3_pos_encoder(a3_dist)  # [n_graph, n_node, n_node, 32] -> [n_graph, n_node, n_node, n_head]
        a3_pos_bias = a3_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + a3_pos_bias

        # edge_feature 边编码------------------------------------------------------------------------------------------------------------
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()  # 记录任意两节点之间的距离[batch_size, max_node_num, max_node_num] 自己到自己的距离记为1
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1  空位（可以看作是虚拟节点） 统一为1，自己到自己的距离记为1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)  # 调整后两个直接相连的节点之间距离也是1

            if self.mhm_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.mhm_dist)

                # 缩减edge_input
                max_dist = self.mhm_dist
                edge_pos = torch.where(edge_padding_mask == False)  # edge_padding_mask [batch_size, max_edges_num]  # edge_pos (batch_size, edges_index)

                # 调整维度，进行curv_encode
                edge_data = edge_data.permute(0, 2, 1)
                edge_data = self.curv_encoder(edge_data)  # [total_edges, n_head]

                # add node_feature to edge_feature
                edge_data = self.node_cat(graph, node_data, edge_data)  # [total_edges, n_head]

                # edge_input扩充 [total_edges, n_head]->[n_graph, max_node_num, max_node_num, max_dist, n_head]
                n_edge = edge_padding_mask.size(1)
                edge_feature = torch.zeros([n_graph, (n_edge + 1), edge_data.size(-1)], device=edge_data.device, dtype=edge_data.dtype)
                edge_feature[edge_pos] = edge_data[:][:]  # edge_feature[n_graph, max_edge_num+1, n_head]

                edge_path = edge_path.reshape(n_graph, n_node * n_node * max_dist)
                dim_0 = torch.arange(n_graph, device=edge_path.device).reshape(n_graph, 1).long()
                edge_input = edge_feature[dim_0, edge_path]
                edge_input = edge_input.reshape(n_graph, n_node, n_node, max_dist, self.n_head)

            edge_input = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.n_head)
            # permute转为[max_dist, n_graph, max_node_num, max_node_num, n_head]
            # reshape转为[max_dist, ---, n_head]

            # 乘以edge_dis_encoder系数，边特征权重按距离递减，超出max_dist后减为0
            edge_input = torch.bmm(
                edge_input,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.n_head, self.n_head
                )[:max_dist, :, :],
            )
            edge_input = edge_input.reshape(
                max_dist, n_graph, n_node, n_node, self.n_head
            ).permute(1, 2, 3, 0, 4)  # edge_input[n_graph, max_node_num, max_node_num, max_dist, n_head]
            # 各个edge上的特征求和取均值
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1)))  # edge_input[n_graph, max_node_num, max_node_num, n_head]
            edge_input = edge_input.permute(0, 3, 1, 2)
            # 最终 edge_output[n_graph, n_head, max_node_num, max_node_num]
        # edge_feature 边编码------------------------------------------------------------------------------------------------------------

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias
    

class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        d_embedding: int = 768,
        d_ff_embedding: int = 3072,
        n_head: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        act_dropout: float = 0.1,
        act_fn: str = "relu",
    ):
        super().__init__()

        # Initialize parameters
        self.d_embedding = d_embedding
        self.n_head = n_head
        self.attn_dropout = attn_dropout

        self.dropout_layer = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.act_dropout_layer = FairseqDropout(act_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(act_fn)
        self.mh_attn = MultiheadAttention(d_embedding, n_head, dropout=dropout)

        # layer norm associated with the self attention layer
        self.attn_layer_norm = LayerNorm(d_embedding)
        self.fc1 = nn.Linear(d_embedding, d_ff_embedding)
        self.fc2 = nn.Linear(d_ff_embedding, d_embedding)
        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(d_embedding)

    def forward(self, x, attn_bias = None, attn_mask = None, attn_padding_mask = None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        x = self.attn_layer_norm(x)
        x, attn = self.mh_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=attn_bias,
            key_padding_mask=attn_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_layer(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.act_dropout_layer(x)
        x = self.fc2(x)
        x = self.dropout_layer(x)
        x = residual + x

        return x, attn
