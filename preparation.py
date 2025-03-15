import os
import argparse
from occwl.io import load_step
from datakit.graph_builder import GraphBuilder

parser = argparse.ArgumentParser("CAD model creation")
parser.add_argument("--step", type=str, default=None, help="Path to step file") 
parser.add_argument("--graph", type=str, default=None, help="Path to graph file (*.BIN)")
parser.add_argument("--bin", type=str, default=None, help="Folder path to output file (*.BIN)")
parser.add_argument("--txt", type=str, default=None, help="Folder path to output file (*.txt)")
parser.add_argument("--uvgrid", action='store_true', help="Whether to display uv grid")

if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.step if args.step is not None else args.graph   

    builder = GraphBuilder()
    if args.step is not None and os.path.exists(args.step):
        # load the step file, assuming the file contains a single solid
        solid = load_step(args.step)[0]
        builder.build_graph(solid, 10, 10, 10)
    elif args.graph is not None and os.path.exists(args.graph):
        # load an existing graph
        builder.read_graph(args.graph)
    else:
        print("No input file provided")
        exit(1)

    # save graph
    output_folder = args.bin if args.bin is not None else args.txt
    if args.txt is not None and os.path.exists(args.txt):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        ouputfile = os.path.join(output_folder, base_name+".txt")
        builder.save_graph(ouputfile, format="txt")
    elif args.bin is not None and os.path.exists(args.bin):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        ouputfile = os.path.join(output_folder, base_name+".bin")
        builder.save_graph(ouputfile, format="bin")
