import trapezoid_decomposition as td


# Example #1

vertices_coords = [(0, 5), (5, 5), (2, 3), (3, 1), (6, 2)]
edges_list = [(0, 1), (0, 2), (1, 4), (2, 3), (0, 3), (3, 4)]
point = (3, 4)


# Example #2
#
# vertices_coords = [(10, 15), (5, 15), (12, 8), (6, 7), (1, 2), (5, 0)]
# edges_list = [(0, 1), (1, 2), (2, 3), (1, 4), (3, 4), (2, 5), (3, 5), (0, 2)]
# point = (1, 5)


if __name__ == '__main__':
    graph, tree = td.preprocessing(vertices_coords, edges_list)
    loc = td.point_localization(graph, tree, point, visualize=True)
    td.visualize_tree(tree)
    print(*loc, sep="\n")