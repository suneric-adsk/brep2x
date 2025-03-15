import sys, os
import argparse
import numpy as np
from occwl.solid import Solid
from occwl.viewer import Viewer
from datakit.shape_builder import CadModelCreator
from datakit.utils import solid_count, solid_edges, display_uv_net
from occwl.io import load_step

FEATURE_COLORS_MAP = {
    "rect_slot": (1.0, 1.0, 0.0),
    "tri_slot": (0.0, 0.0, 1.0),
    "cir_slot": (0.0, 1.0, 0.0),
    "rect_psg": (1.0, 0.7, 0.8),
    "tri_psg": (0.0, 1.0, 0.0),
    "hexa_psg": (1.0, 0.0, 0.0),
    "hole": (0.0, 0.0, 1.0),
    "rect_step": (0.5, 0.0, 0.5),
    "tside_step": (1.0, 0.0, 0.0),
    "slant_step": (1.0, 0.6, 0.0),
    "rect_b_step": (0.5, 0.0, 0.5),
    "tri_step": (1.0, 0.7, 0.8),
    "cir_step": (0.0, 1.0, 0.0),
    "rect_b_slot": (1.0, 0.6, 0.0),
    "cir_b_slot": (1.0, 0.0, 0.0),
    "u_b_slot": (1.0, 0.6, 0.0),
    "rect_pkt": (1.0, 0.0, 0.0),
    "key_pkt": (0.0, 0.0, 1.0),
    "tri_pkt": (1.0, 1.0, 0.0),
    "hexa_pkt": (0.0, 1.0, 0.0),
    "o_ring": (1.0, 0.6, 0.0),
    "b_hole": (1.0, 1.0, 0.0),
    "chamfer": (0.5, 0.0, 0.5),
    "fillet": (1.0, 0.7, 0.8),
}

def display_feature(v, shape, face_map):
    for face in face_map.keys():
        label = face_map[face]
        if label == "primitive":
            v.display(face, transparency=0.0, color=(0.4,0.4,0.4))
        else:
            v.display(face, transparency=0.0, color=FEATURE_COLORS_MAP[label])
    
    for edge in solid_edges(shape):
        v.display(edge, color=(0.2,0.2,0.2), update=True)

parser = argparse.ArgumentParser("CAD model creation")
parser.add_argument("--json", type=str, default=None, help="Path to json file")
parser.add_argument("--shape", type=int, default=-1, help="The index of shape to display")
parser.add_argument("--step", type=str, default=None, help="Path to step file")
parser.add_argument("--uvgrid", action='store_true', help="Whether to display uv grid")   

if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.json if args.json else args.step
    v = Viewer(backend="wx")
    v._display.get_parent().GetParent().SetTitle(filename)
    v.display_points(np.array([[0, 0, 0]]), marker="point", color="RED")

    if args.step is not None and os.path.exists(args.step):
        solid = load_step(args.step)[0]
        v.display(solid, transparency=0.0, color=(0.4, 0.4, 0.4))
        if args.uvgrid:
            display_uv_net(v, solid)
    elif args.json is not None and os.path.exists(args.json):
        cadCreator = CadModelCreator(args.json)
        shape, label = cadCreator.get_model(args.shape) 
        display_feature(v, shape, label)
        if args.uvgrid and solid_count(shape) == 1: 
            solid = Solid(shape)
            v.display(solid, transparency=0.0, color=(0.2,0.2,0.2), update=True)
            display_uv_net(v, solid)
    # show the viewer
    v.fit()
    v.show()