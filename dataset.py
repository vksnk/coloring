import json
import pprint
import torch

from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path
from typing import Dict, List, Any

import csp_solver
import visualize


def load_json_from_folder(folder_path: str) -> Dict[str, Any]:
    path = Path(folder_path)

    if not path.is_dir():
        print(f"Error: Path '{path}' is not a valid directory.")
        return {}

    parsed_data = {}

    for json_file in path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:

            data = json.load(f)

            parsed_data[json_file] = data

    return parsed_data


class RigSetDataset(InMemoryDataset):
    def __init__(self):
        pass


if __name__ == "__main__":
    # counter = 0
    basic_graphs = load_json_from_folder("../dataset/loader_test")
    for file_name, json in tqdm(basic_graphs.items(), desc="Processing files"):
        print(file_name)
        # One file can have multiple graphs.
        for func, graph in json.items():
            print(func)
            # Make sure it has required fields.
            assert "edges" in graph
            assert "nodes" in graph
            nodes = {}
            node_index = 0
            for node in graph["nodes"]:
                nodes[node] = node_index
                node_index += 1

            edges1 = []
            edges2 = []
            for edge in graph["edges"]:
                # Make sure it has required fields.
                assert "node 1" in edge
                assert "node 2" in edge
                node1 = edge["node 1"]
                node2 = edge["node 2"]

                # Make sure this is something we've seen in the node list.
                assert node1 in nodes
                assert node2 in nodes

                edges1.append(nodes[node1])
                edges2.append(nodes[node2])

            # Initialize with zeros for now.
            x = torch.tensor([[0]] * len(nodes), dtype=torch.float)
            edge_index = torch.tensor([edges1, edges2], dtype=torch.long)

            if len(nodes) > 28:
                coloring, best_k = csp_solver.solve_graph_coloring_with_heuristic(
                    [edges1, edges2]
                )
            else:
                coloring, best_k = csp_solver.solve_graph_coloring_with_csp(
                    [edges1, edges2]
                )

            print(coloring, best_k)

            data = Data(x=x, edge_index=edge_index, y=coloring, yk=best_k)
            data.validate(raise_on_error=True)

            # if counter == 0:
            # visualize.visualize_graph([edges1, edges2], coloring, best_k)
            # counter += 1
