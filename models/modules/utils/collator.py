# -*- coding: utf-8 -*-
import dgl
import torch
from .macro import *

def pad_mask_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_ones([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze(x, padlen):  #x[num_nodes]
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_face_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):  # x[num_nodes, num_nodes]
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_d2_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 32], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_ang_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 32], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)
     
def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):  #x[num_nodes, num_nodes, max_dist]
    xlen1, xlen2, xlen3 = x.size() 
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = -1 * x.new_ones([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, multi_hop_max_dist=8, spatial_pos_max=32, split="train"):  #items({PYGGraph_1, PYGGraph_1_p}, {PYGGraph_2, PYGGraph_2_p}, ..., batchsize)
    
    if split == "train":
        # primitives & feature
        items_0 = [
            (
                item["sample"].graph,
                item["sample"].node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item["sample"].face_area,  
                item["sample"].face_type, 
                item["sample"].edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item["sample"].in_degree,
                item["sample"].attn_bias,
                item["sample"].spatial_pos,
                item["sample"].d2_dist,
                item["sample"].a3_dist,
                item["sample"].edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]           
                item["sample"].label_prim_cmd,
                item["sample"].label_prim_param,
                item["sample"].label_feat_cmd,
                item["sample"].label_feat_param
            )
            for item in items
        ]
        # primitive
        items_1 = [
            (
                item["sample_p"].graph,
                item["sample_p"].node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item["sample_p"].face_area,
                item["sample_p"].face_type,
                item["sample_p"].edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item["sample_p"].in_degree,
                item["sample_p"].attn_bias,
                item["sample_p"].spatial_pos,
                item["sample_p"].d2_dist,
                item["sample_p"].a3_dist,
                item["sample_p"].edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]
                item["sample_p"].label_prim_cmd,
                item["sample_p"].label_prim_param,
                item["sample_p"].label_feat_cmd,
                item["sample_p"].label_feat_param
            )
            for item in items
        ]
        items = items_0 + items_1
        (
            graphs,
            node_datas,
            face_areas,
            face_types,
            edge_datas,
            in_degrees,
            attn_biases,
            spatial_poses, 
            d2_dists,
            a3_dists,
            edge_paths,
            label_prim_cmds,
            label_prim_params,
            label_feat_cmds,
            label_feat_params
        ) = zip(*items)
    else:
        items = [
            (
                item.graph,
                item.node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item.face_area,
                item.face_type,
                item.edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item.in_degree,
                item.attn_bias,
                item.spatial_pos,
                item.d2_dist,
                item.a3_dist,
                item.edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]   
                item.label_prim_cmd,
                item.label_prim_param,
                item.label_feat_cmd,
                item.label_feat_param,
            )
            for item in items
        ]
        (
            graphs,
            node_datas,
            face_areas,
            face_types,
            edge_datas,
            in_degrees,
            attn_biases,
            spatial_poses, 
            d2_dists,
            a3_dists,
            edge_paths,
            label_prim_cmds,
            label_prim_params,
            label_feat_cmds,
            label_feat_params
        ) = zip(*items)  #解压缩

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        
    max_node_num = max(i.size(0) for i in node_datas)
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(i.size(-1) for i in edge_paths)
    max_dist = max(max_dist, multi_hop_max_dist)

    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])

    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])

    node_data = torch.cat([i for i in node_datas])  #node_datas(batch_size, [num_nodes, U_grid, V_grid, pnt_feature])

    face_area = torch.cat([i for i in face_areas])

    face_type = torch.cat([i for i in face_types])

    edge_data = torch.cat([i for i in edge_datas])  #edge_datas(batch_size, [num_edges, U_grid, pnt_feature])元组

    edge_path = torch.cat(     #edges_paths(batch_size, [num_nodes, num_nodes, max_dist])
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )

    spatial_pos = torch.cat(   #spatial_pos(batch_size, [num_nodes, num_nodes])
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )    
    d2_dist = torch.cat(
        [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_dists]
    )
    a3_dist = torch.cat(
        [pad_ang_pos_unsqueeze(i, max_node_num) for i in a3_dists]
    )

    in_degree = torch.cat([i for i in in_degrees])

    batched_graph = dgl.batch([i for i in graphs])

    # sequence labels
    batched_label_prim_cmds = torch.cat([torch.unsqueeze(i, dim=0) for i in label_prim_cmds])
    batched_label_prim_params = torch.cat([torch.unsqueeze(i, dim=0) for i in label_prim_params])
    batched_label_feat_cmds = torch.cat([torch.unsqueeze(i, dim=0) for i in label_feat_cmds])
    batched_label_feat_params = torch.cat([torch.unsqueeze(i, dim=0) for i in label_feat_params])

    batch_data = dict(
        padding_mask = padding_mask,       #[batch_size, max_node_num]
        edge_padding_mask = edge_padding_mask,  #[batch_size, max_edge_num]
        graph=batched_graph,
        node_data = node_data,             #[total_node_num, U_grid, V_grid, pnt_feature]  
        face_area = face_area,             #[batch_size, max_node_num] / [total_node_num]
        face_type = face_type,             #[batch_size, max_node_num] / [total_node_num]
        edge_data = edge_data,             #[total_edge_num, U_grid, pnt_feature]
        in_degree = in_degree,             #[batch_size, max_node_num]
        out_degree = in_degree,            #[batch_size, max_node_num]
        attn_bias = attn_bias,             #[batch_size, max_node_num+1, max_node_num+1]
        spatial_pos = spatial_pos,         #[batch_size, max_node_num, max_node_num]
        d2_dist = d2_dist,         #[batch_size, max_node_num, max_node_num, 64]
        a3_dist = a3_dist,   #[batch_size, max_node_num, max_node_num, 64]
        edge_path = edge_path,             #[batch_size, max_node_num, max_node_num, max_dist] 空位用-1填充
        label_prim_cmd = batched_label_prim_cmds,
        label_prim_param = batched_label_prim_params,
        label_feat_cmd = batched_label_feat_cmds,
        label_feat_param = batched_label_feat_params
    )
    return batch_data
