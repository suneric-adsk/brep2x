import os 
import sys
sys.path.append('..')
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from occwl.io import load_step

from datakit.graph_builder import GraphBuilder
from torch_geometric.data import Data as Graph
from models.modules.utils.collator import collator

"""
STEP dataset for test only
"""
class STEPDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = self.load_files(root_dir)

    def load_files(self, root_dir):
        """Load files in root_dir"""
        files = os.listdir(root_dir)
        return [f for f in files if f.endswith(".stp") or f.endswith(".step")]
        
    def __len__(self):
        return len(self.files)
    
    def load_one_graph(self, filepath):
        graph_builder = GraphBuilder()
        graph_builder.build_graph(load_step(filepath)[0], 10, 10, 10)
        graph = graph_builder.dgl_graph
        label = graph_builder.label
        dense_adj = graph.adj().to_dense().type(torch.int)
        N = graph.num_nodes()

        pyg = Graph()
        pyg.graph = graph 
        pyg.node_data = graph.ndata["x"].type(torch.FloatTensor)   #node_data[num_nodes, U_grid, V_grid, pnt_feature]
        pyg.face_area = graph.ndata["y"].type(torch.int)     #face_area[num_nodes]
        pyg.face_type = graph.ndata["z"].type(torch.int)     #face_type[num_nodes]
        pyg.edge_data = graph.edata["x"].type(torch.FloatTensor)
        pyg.in_degree = dense_adj.long().sum(dim=1).view(-1)       
        pyg.attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
        pyg.edge_path = label["edges_path"]           # edge_input[num_nodes, num_nodes, max_dist, 1, U_grid, pnt_feature]
        pyg.spatial_pos = label["spatial_pos"]        # spatial_pos[num_nodes, num_nodes]
        pyg.d2_dist = label["d2_distance"]        # d2_distance[num_nodes, num_nodes, 64]
        pyg.a3_dist = label["angle_distance"]  # angle_distance[num_nodes, num_nodes, 64]

        if ("commands_primitive" in label):
            pyg.label_prim_cmd = label["commands_primitive"]
            pyg.label_prim_param = label["args_primitive"]
            pyg.label_feat_cmd = label["commands_feature"]
            pyg.label_feat_param = label["args_feature"]
        else:
            pyg.label_prim_cmd = torch.zeros([10])
            pyg.label_prim_param = torch.zeros([10, 11])
            pyg.label_feat_cmd = torch.zeros([12])
            pyg.label_feat_param = torch.zeros([12, 12])
        # pyg.data_id = int(os.path.basename(filepath)[:-4])
        return pyg
    
    def __getitem__(self, index):
        fn = os.path.join(self.root_dir,self.files[index])
        sample = self.load_one_graph(fn)
        return sample
    
    def _collate(self, batch):
        return collator(
            items = batch,
            split = "test"
        )
    
    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, drop_last=True): 
        return DataLoader(
            dataset = self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=drop_last
        )