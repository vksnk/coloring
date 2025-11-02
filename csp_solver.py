from ortools.sat.python import cp_model


def solve_graph_coloring_with_csp(edge_lists):
    """
    Finds the optimal graph coloring from two lists (start nodes, end nodes).

    :param edge_lists: A tuple or list containing two lists:
                       ([start_nodes], [end_nodes])
                       e.g., (['A', 'B'], ['B', 'C'])
    :return: A tuple (coloring_dict, chromatic_number)
    """

    start_nodes = edge_lists[0]
    end_nodes = edge_lists[1]

    # 1. Get all unique nodes from both lists
    # We can use set union for this
    nodes_set = set(start_nodes) | set(end_nodes)

    nodes = list(nodes_set)
    num_nodes = len(nodes)

    # We test k=1, k=2, ... until a solution is found.
    # The first k that works is the chromatic number.
    for k in range(1, num_nodes + 2):
        model = cp_model.CpModel()

        # 2. VARIABLES:
        # Create one integer variable for each node.
        # The domain of each variable is [0, k-1] (the k colors)
        node_colors = {}
        for node in nodes:
            node_colors[node] = model.NewIntVar(0, k - 1, f"color_{node}")

        # 3. CONSTRAINTS: (This is the main change)
        # Use zip() to pair corresponding start and end nodes.
        # For every edge (u, v), add a constraint
        # that node_colors[u] != node_colors[v]
        for u, v in zip(start_nodes, end_nodes):
            model.Add(node_colors[u] != node_colors[v])

        # 4. SOLVE:
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # 5. CHECK RESULT:
        # If the status is OPTIMAL or FEASIBLE, we found a solution!
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # We found the first k that works, so this is the chromatic number.
            coloring = {node: solver.Value(node_colors[node]) for node in nodes}
            return coloring, k

    # Should be unreachable
    return None, -1
