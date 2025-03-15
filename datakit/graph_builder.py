import numpy as np
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

import dgl
import torch
from dgl.data.utils import save_graphs, load_graphs

from datakit.utils import bounding_box, center_and_scale_uvgrid, d2_a3_distance

FACE_TYPE_MAP = {"plane": 2,"cylinder": 3,"sphere": 4,"cone": 5,"torus": 6}

def area_level(area, max=1e6, max_level=2048):
    if area > max:
        return max_level
    step = max / (max_level-1)
    return round(area/step)+1

def get_face_type(face):
    surface_type = face.surface_type()
    if surface_type in FACE_TYPE_MAP.keys():
        return FACE_TYPE_MAP[surface_type]
    return 1

def edge_u_visibility(edge, num_u=10):
    visibility = []
    u_values = np.zeros(num_u, dtype=np.float32)
    bound = edge.u_bounds()
    for i in range(num_u):
        u = bound.interpolate(float(i) / (num_u - 1))
        u_values[i] = u
        val = 1 if bound.contains_value(u) else 0
        visibility.append(val)
    visibility = np.asarray(visibility).reshape((num_u, -1))
    return visibility

class GraphBuilder:
    def __init__(self):
        self.dgl_graph = None
        self.label = {}

    def compute_shape_distance(self, graph, bbox, size=32):
        # [num_nodes, num_nodes, size]
        num_nodes = len(graph.nodes)
        dist = np.zeros((num_nodes, num_nodes, size), dtype=np.float32)
        angle = np.zeros((num_nodes, num_nodes, size), dtype=np.float32)
        diag_len = np.linalg.norm(bbox[1]-bbox[0])
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    dist[i,j,0] = 1.0
                face1 = graph.nodes[i]["face"]
                face2 = graph.nodes[j]["face"]
                d2, a3 = d2_a3_distance(face1, face2, diag_len, size)
                dist[i,j] = np.round(d2, decimals=6)
                angle[i,j] = np.round(a3, decimals=6)
        return np.array(dist), np.array(angle)    

    def build_graph(self, solid, crv_sample, srf_sample_u, srf_sample_v, center_and_scale=False):
        # Build face adjacency graph with B-rep entities as node and edge features
        graph_face_feat = []
        graph_face_area = []
        graph_face_type = []
        graph = face_adjacency(solid)
        
        for idx in graph.nodes:
            face = graph.nodes[idx]["face"]
            points = uvgrid(face, method="point", num_u=srf_sample_u, num_v=srf_sample_v)    
            normals = uvgrid(face, method="normal", num_u=srf_sample_u, num_v=srf_sample_v)
            visibility = uvgrid(face, method="visibility_status", num_u=srf_sample_u, num_v=srf_sample_v)
            masks = np.logical_or(visibility == 0, visibility == 1)
            face_feat = np.concatenate([points, normals, masks], axis=-1)
            graph_face_feat.append(face_feat)
            face_area = area_level(face.area())
            graph_face_area.append(int(face_area))
            face_type = get_face_type(face)
            graph_face_type.append(int(face_type))    
        graph_face_feat = np.asarray(graph_face_feat)
        graph_face_area = np.asarray(graph_face_area)
        graph_face_type = np.asarray(graph_face_type)   

        graph_edge_feat = []
        for idx in graph.edges:
            edge = graph.edges[idx]["edge"]
            if not edge.has_curve():
                continue
            points = ugrid(edge, method="point", num_u=crv_sample)
            tangents = ugrid(edge, method="tangent", num_u=crv_sample)
            visibility = edge_u_visibility(edge, num_u=crv_sample)
            masks = np.logical_or(visibility == 0, visibility == 1)
            edge_feat = np.concatenate([points, tangents, masks], axis=-1)
            graph_edge_feat.append(edge_feat)
        graph_edge_feat = np.asarray(graph_edge_feat)

        bbox = bounding_box(graph_face_feat)
        if (center_and_scale):
            graph_face_feat, center, scale = center_and_scale_uvgrid(graph_face_feat, bbox)
            graph_edge_feat[...,:3] -= center
            graph_edge_feat[...,:3] *= scale
        
        d2_dist, a3_dist = self.compute_shape_distance(graph, bbox)

        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        self.dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
        print("Build graph {} node, {} edges".format(self.dgl_graph.num_nodes(), self.dgl_graph.num_edges()))
        self.dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
        self.dgl_graph.ndata["y"] = torch.from_numpy(graph_face_area)
        self.dgl_graph.ndata["z"] = torch.from_numpy(graph_face_type)
        self.dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)

        # Initialize label
        # shortest path [num_nodes, num_nodes, 8] and distance [num_nodes, num_nodes] between nodes, 
        shortest_dist, shortest_path = dgl.shortest_dist(self.dgl_graph, None, True)
        self.label["spatial_pos"] = shortest_dist
        self.label["edges_path"] = shortest_path
        self.label["d2_distance"] = torch.from_numpy(d2_dist)
        self.label["angle_distance"] = torch.from_numpy(a3_dist)

    def read_graph(self, graphfile):
        graph_and_label = load_graphs(graphfile)
        self.dgl_graph = graph_and_label[0][0]
        print("Read graph {} node, {} edges".format(self.dgl_graph.num_nodes(), self.dgl_graph.num_edges()))
        self.label["edges_path"] = graph_and_label[1]["edges_path"]
        self.label["spatial_pos"] = graph_and_label[1]["spatial_pos"]
        self.label["d2_distance"] = graph_and_label[1]["d2_distance"]
        self.label["angle_distance"] = graph_and_label[1]["angle_distance"]
        if "commands_primitive" in graph_and_label[1]:
            self.label["commands_primitive"] = graph_and_label[1]["commands_primitive"]
            self.label["args_primitive"] = graph_and_label[1]["args_primitive"]
            self.label["commands_feature"] = graph_and_label[1]["commands_feature"]
            self.label["args_feature"] = graph_and_label[1]["args_feature"]

    def save_graph(self, filename, format="bin"):
        if format == "bin":
            save_graphs(filename, [self.dgl_graph], self.label)
        elif format == "txt":
            with open(filename, 'w') as file:
                # Write the nodes and their features (if any)
                file.write("Nodes:\n")
                for node in range(self.dgl_graph.num_nodes()):
                    node_features = self.dgl_graph.ndata if len(self.dgl_graph.ndata) > 0 else {}
                    file.write(f"Node {node}")
                    for key, value in node_features.items():
                        file.write(f", {key}: {value[node].tolist()}")
                    file.write("\n")

                # Write the edges and their features (if any)
                file.write("\nEdges:\n")
                for edge in range(self.dgl_graph.num_edges()):
                    edge_features = self.dgl_graph.edata if len(self.dgl_graph.edata) > 0 else {}
                    file.write(f"Edge {edge}")
                    for key, value in edge_features.items():
                        file.write(f", {key}: {value[edge].tolist()}")
                    file.write("\n")
                
                # Write the labels
                file.write("\nLabels:\n")
                for key, value in self.label.items():
                    file.write(f"{key}: {value.tolist()}\n")