import functools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List


class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.w = None
        self.data = None


def visualize_tree(tree):
    graph = nx.DiGraph()

    def traverse(node, parent=None):
        if node is None:
            return

        graph.add_node(node)  # Add node without label

        if parent is not None:
            graph.add_edge(parent, node)

        traverse(node.left, node)
        traverse(node.right, node)

    traverse(tree)

    pos = hierarchy_pos(graph, tree)
    labels = {node: node.data if hasattr(node, 'data') else "" for node in graph.nodes()}
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=800, node_color='lightblue',
            font_weight='bold', font_color='black', font_size=10, edge_color='gray', width=1.0, alpha=0.7)
    plt.title("Trapezoid Decomposition Tree")
    plt.axis('off')


def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = {root: (xcenter, vert_loc)}
    neighbors = list(G.neighbors(root))  # Convert to a list
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos[neighbor] = (nextx, vert_loc - vert_gap)
            pos.update(hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx))
    return pos


def _show_plot(graph, point=None):
    pos = graph.nodes(True)
    pos = {x[0]: (x[1][0], x[1][1]) for x in pos}
    fig, ax = plt.subplots(num="Visualization")
    nx.draw(graph, node_size=400, pos=pos)  # draw nodes and edges
    nx.draw_networkx_labels(graph, pos=pos)  # draw node labels
    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    if point is not None:
        plt.scatter(point[0], point[1], color="red")
    plt.ion()
    plt.show()


# calculates the x-coordinate of a point on a line segment
# based on its y-coordinate using linear interpolation
def _calculate_x(p1, p2, y):
    return -((p2[0]-p1[0]) * (p2[1]-y) / (p2[1]-p1[1]) - p2[0])


# determines the relative location of a point with respect to a line
# segment by calculating the signed distance between the point and the line
def _point_to_line_location(point, p1, p2):
    return -np.sign((p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0]))


# compares two edges based on their y-coordinates, handles overlaps,
# and determines their relative ordering based on their x-coordinates
# at a specific y-coordinate
def _edges_comp(e1, e2):
    e1_p1, e1_p2 = e1[1][0], e1[1][1]
    e2_p1, e2_p2 = e2[1][0], e2[1][1]
    if not max(e1_p1[1], e2_p1[1]) < min(e1_p2[1], e2_p2[1]):
        return 1
    if e1_p1[1] > e2_p1[1]:
        return (-1)*_edges_comp(e2, e1)
    y = (e2_p1[1] + min(e1_p2[1], e2_p2[1])) / 2
    x1 = _calculate_x(e1_p1, e1_p2, y)
    x2 = _calculate_x(e2_p1, e2_p2, y)
    return -1 if x1 < x2 else 1


def _balance_w(node):
    def get_w(n): return 0 if (n is None) else n.w
    if type(node.data) is tuple:
        node.w = get_w(node.left) + get_w(node.right)
    else:
        node.w = get_w(node.left) + get_w(node.right) + 1


def _create_balanced_tree(u):
    if not u:
        return None
    mid = len(u) // 2
    u[mid].left = _create_balanced_tree(u[:mid])
    u[mid].right = _create_balanced_tree(u[mid + 1:])
    _balance_w(u[mid])
    return u[mid]


def _balance(u):
    w_u = sum(i.w for i in u)
    if w_u == 0:
        return _create_balanced_tree(u)
    curr_sum = 0
    r = 0
    for curr_u in u:
        curr_sum += curr_u.w
        if curr_sum >= w_u / 2:
            break
        r += 1
    # balance e(r-1) t(r) e(r) balance
    # ..r-2]   r-1    r    r+1  [r+2..
    #    0      1     2     3     4
    seq: List[Any] = [None]*5
    if r > 1:
        seq[0] = _balance(u[:r-1])
    if r > 0:
        seq[1] = u[r-1]
    seq[2] = u[r]
    if r+1 < len(u):
        seq[3] = u[r+1]
    if r+2 < len(u):
        seq[4] = _balance(u[r+2:])

    if seq[1] is None and seq[3] is None:
        root = seq[2]
    elif seq[1] is None:
        root = seq[3]
        root.left = seq[2]
        root.right = seq[4]
    elif seq[3] is None:
        root = seq[1]
        root.left = seq[0]
        root.right = seq[2]
    else:
        tmp = seq[3]
        tmp.left = seq[2]
        tmp.right = seq[4]
        _balance_w(tmp)
        # ---
        root = seq[1]
        root.left = seq[0]
        root.right = tmp
    _balance_w(root)
    return root


def _trapezoid(g, e, v, i):
    if not v:
        return None
    y_med = g.nodes(True)[v[(len(v) - 1) // 2]][1]
    i_i = [[min(i), y_med], [y_med, max(i)]]

    t = 0
    v_t_indices = {}
    v_i = [[], []]
    for curr_e in e:
        coord_0 = g.nodes[curr_e[0]]
        coord_1 = g.nodes[curr_e[1]]
        for i in range(2):
            if i_i[i][0] < coord_0[1] < i_i[i][1]:
                v_i[i].append(curr_e[0])
            if i_i[i][0] < coord_1[1] < i_i[i][1]:
                v_i[i].append(curr_e[1])

            if coord_0[1] <= i_i[i][0] and coord_1[1] >= i_i[i][1]:
                for v in v_i[i]:
                    v_t_indices[v] = t
                v_i[i] = []
                t += 1
    for i in range(2):
        for v in v_i[i]:
            v_t_indices[v] = t
        v_i[i] = []
        t += 1

    v_t: List[List[int]] = [[] for _ in range(t)]
    for i, v_index in v_t_indices.items():
        v_t[v_index].append(i)

    t = 0
    e_i, u_i = [[], []], [[], []]
    for curr_e in e:
        coord_0 = g.nodes[curr_e[0]]
        coord_1 = g.nodes[curr_e[1]]
        for i in range(2):
            if i_i[i][0] < coord_0[1] < i_i[i][1] or i_i[i][0] < coord_1[1] < i_i[i][1]:
                e_i[i].append(curr_e)

            if coord_0[1] <= i_i[i][0] and coord_1[1] >= i_i[i][1]:
                tmp = _trapezoid(g, e_i[i], v_t[t], i_i[i])
                if tmp is not None:
                    u_i[i].append(tmp)
                node = TreeNode()
                node.w = 0
                node.data = curr_e
                u_i[i].append(node)
                e_i[i] = []
                t += 1
    for i in range(2):
        tmp = _trapezoid(g, e_i[i], v_t[t], i_i[i])
        if tmp is not None:
            u_i[i].append(tmp)
        t += 1
    node = TreeNode()
    node.left = _balance(u_i[0])
    node.right = _balance(u_i[1])
    node.data = next(x[0] for x in g.nodes(True) if x[1][1] == y_med)
    _balance_w(node)
    return node


def preprocessing(vertices_coords, edges_list, visualize=False):
    vertices_indices = list(range(len(vertices_coords)))
    vertices_indices.sort(key=lambda x: vertices_coords[x][1])

    edges_list = [tuple(sorted(list(e), key=lambda p: vertices_coords[p][1])) for e in edges_list]
    edges_list_t = [(e, (vertices_coords[e[0]], vertices_coords[e[1]])) for e in edges_list]
    edges_list_t = sorted(edges_list_t, key=functools.cmp_to_key(_edges_comp))
    edges_list = [e_t[0] for e_t in edges_list_t]

    graph = nx.empty_graph(len(vertices_coords))
    graph.add_edges_from(edges_list)
    for v in graph:
        graph.nodes[v].update(dict(enumerate(vertices_coords[v])))

    if visualize:
        _show_plot(graph)

    i_0 = [min(vertices_coords, key=lambda x: x[1])[1], max(vertices_coords, key=lambda x: x[1])[1]]
    tree = _trapezoid(graph, edges_list, vertices_indices, i_0)
    return graph, tree


def point_localization(graph, tree, point, visualize=False):
    if visualize:
        _show_plot(graph, point=point)
    path = []
    node = tree
    while node is not None:
        data = node.data
        if type(data) is tuple:
            p1 = graph.nodes[data[0]]
            p2 = graph.nodes[data[1]]
            pos = _point_to_line_location(point, p1, p2)
            if pos == -1:
                d = "left"
                node = node.left
            else:
                d = "right"
                node = node.right
        else:
            y = graph.nodes[data][1]
            if point[1] < y:
                d = "bottom"
                node = node.left
            else:
                d = "top"
                node = node.right
        path.append({"data": data, "direction": d})
    return path