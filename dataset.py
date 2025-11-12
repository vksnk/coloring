import json
import pprint
import torch

from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path
from typing import Dict, List, Any

import reference_solver
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
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["rig_set.pt"]

    def process_folder(self, folder_name):
        datas = []
        basic_graphs = load_json_from_folder(folder_name)
        for file_name, json in tqdm(basic_graphs.items(), desc="Processing files"):
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

                # If graph is too large we won't be able to find a solution in reasonable
                # amount of time.
                if len(nodes) > 28:
                    coloring, best_k = (
                        reference_solver.solve_graph_coloring_with_heuristic(
                            [edges1, edges2], len(nodes)
                        )
                    )
                else:
                    coloring, best_k = reference_solver.solve_graph_coloring_with_csp(
                        [edges1, edges2], len(nodes)
                    )

                data = Data(x=x, edge_index=edge_index, y=coloring, yk=best_k)
                data.validate(raise_on_error=True)

                datas.append(data)
        return datas

    def process(self):
        data_list = self.process_folder("../dataset/basic_graphs")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    dataset = RigSetDataset("data/")
    print(dataset.len())
    # Visualize random graph.
    edges = dataset[20].edge_index.tolist()
    coloring = dataset[20].y
    visualize.visualize_graph(edges, coloring)
