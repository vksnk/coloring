import json
import pprint
import torch
import random

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

    @property
    def train_mask(self):
        return self._data.train_mask

    @property
    def val_mask(self):
        return self._data.val_mask

    @property
    def test_mask(self):
        return self._data.test_mask

    def process_folder(self, folder_name):
        datas = []
        basic_graphs = load_json_from_folder(folder_name)
        for file_name, json in tqdm(basic_graphs.items(), desc="Processing files"):
            # One file can have multiple graphs.
            for func, graph in json.items():
                # print(func)
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

                edge_index = torch.tensor([edges1, edges2], dtype=torch.long)

                # If graph is too large we won't be able to find a solution in reasonable
                # amount of time.
                if len(nodes) > 25:
                    coloring, best_k = (
                        reference_solver.solve_graph_coloring_with_heuristic(
                            [edges1, edges2], len(nodes)
                        )
                    )
                else:
                    coloring, best_k = reference_solver.solve_graph_coloring_with_csp(
                        [edges1, edges2], len(nodes)
                    )

                data = Data(num_nodes=len(nodes), edge_index=edge_index, yk=best_k)
                data.validate(raise_on_error=True)

                datas.append(data)
        return datas

    def process(self):
        basic_graphs = self.process_folder("../coloring-dataset/basic_graphs")
        codenet_graphs = self.process_folder("../coloring-dataset/codenet_graphs")

        data_list = basic_graphs + codenet_graphs

        speccpu_graph_folders = [
            "500.perlbench_r",
            "502.gcc_r",
            "505.mcf_r",
            "507.cactuBSSN_r",
            "511.povray_r",
            "519.lbm_r",
            "521.wrf_r",
            "526.blender_r",
            "527.cam4_r",
            "531.deepsjeng_r",
            "541.leela_r",
            "544.nab_r",
            "557.xz_r",
            "600.perlbench_s",
            "602.gcc_s",
            "605.mcf_s",
            "628.pop2_s",
            "999.specrand_i",
        ]

        for subfolder in speccpu_graph_folders:
            data_list += self.process_folder(
                f"../coloring-dataset/spec_graphs/{subfolder}"
            )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Split into training, validation and test.
        random.shuffle(data_list)

        n = len(data_list)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True

        data, slices = self.collate(data_list)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print("Total number of graphs", len(data_list))
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = RigSetDataset("data/")
    print(dataset.len())
    # Visualize random graph.
    edges = dataset[20].edge_index.tolist()
    coloring = dataset[20].y
    visualize.visualize_graph(edges, coloring)
