import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_graph(edge_lists, coloring, num_colors):
    # print(edge_lists)
    # Create networkx graph.
    G = nx.Graph()
    G.add_edges_from(zip(edge_lists[0], edge_lists[1]))

    # Get reasonable color pallete.
    if num_colors <= 20:
        # Use a high-quality categorical map if possible
        # 'tab20' has 20 distinct colors.
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i) for i in range(num_colors)]
    else:
        # If we need tons of colors, sample from a continuous rainbow spectrum
        # specifically using standard evenly spaced intervals (0 to 1)
        cmap = plt.get_cmap("rainbow")
        palette = [cmap(i) for i in np.linspace(0, 1, num_colors)]

    color_map = []
    for node in G.nodes():
        color_id = coloring[node]
        color_map.append(palette[color_id])

    # 4. Draw the graph
    plt.figure(figsize=(6, 4))
    nx.draw(
        G,
        with_labels=True,
        node_color=color_map,
        node_size=500,
        font_color="white",
        font_weight="bold",
        edge_color="gray",
    )
    plt.show()
