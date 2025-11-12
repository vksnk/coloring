from ortools.sat.python import cp_model
import networkx as nx


def solve_graph_coloring_with_heuristic(edge_lists, total_nodes, strategy="DSatur"):
    """
    Colors a large graph using heuristics.
    """

    # Construct nx graph.
    G = nx.Graph()
    G.add_edges_from(zip(edge_lists[0], edge_lists[1]))
    # Select strategy.
    nx_strategy = (
        "saturation_largest_first" if strategy == "DSatur" else "largest_first"
    )

    # Compute graph coloring.
    coloring_dict = nx.greedy_color(G, strategy=nx_strategy)
    coloring = [coloring_dict.get(node_index, 0) for node_index in range(total_nodes)]

    # Calculate the number of colors used.
    num_colors = max(coloring) + 1

    return coloring, num_colors


def solve_graph_coloring_with_csp(edge_lists, total_nodes):
    """
    Finds the optimal graph coloring using CSP solver.
    """

    start_nodes = edge_lists[0]
    end_nodes = edge_lists[1]

    # Get all unique nodes from both lists
    nodes_set = set(start_nodes) | set(end_nodes)

    nodes = list(nodes_set)
    num_nodes = len(nodes)

    # We test k=1, k=2, ... until a solution is found.
    # The first k that works is the chromatic number.
    for k in range(1, num_nodes + 2):
        model = cp_model.CpModel()

        # Create one integer variable for each node.
        # The domain of each variable is [0, k-1]
        node_colors = {}
        for node in nodes:
            node_colors[node] = model.NewIntVar(0, k - 1, f"color_{node}")

        # For every edge (u, v), add a constraint that node_colors[u] != node_colors[v]
        for u, v in zip(start_nodes, end_nodes):
            model.Add(node_colors[u] != node_colors[v])

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # If the status is OPTIMAL or FEASIBLE, we found a solution.
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # We found the smallest k that works, so this is the chromatic number.
            coloring = []
            for node_index in range(total_nodes):
                if node_index in node_colors:
                    coloring.append(solver.Value(node_colors[node_index]))
                else:
                    coloring.append(0)
            return coloring, k

    # Should be unreachable, because we always can find at least some solution.
    assert False

    return None, -1
