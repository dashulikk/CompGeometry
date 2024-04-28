from enum import IntEnum
from matplotlib import pyplot as plt
from matplotlib import patches as patches


class Dim(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1


class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.point_index = None
        self.point = None
        self.line_dim = None

def _sorted_list_split(left_list, right_list, to_split_list):
    if not to_split_list:
        return [], []
    sep_flags = [2] * (max(to_split_list) + 1)
    for p in left_list:
        sep_flags[p] = 0
    for p in right_list:
        if sep_flags[p] != 2:
            raise Exception("_sorted_list_split")
        sep_flags[p] = 1
    split = [[], [], []]
    for p in to_split_list:
        split[sep_flags[p]].append(p)
    return split[0], split[1]


def _m(node):
    return node.point[node.line_dim]


def _is_inside_rect(point, x_range, y_range):
    return (x_range[0] <= point[0] <= x_range[1]) and (y_range[0] <= point[1] <= y_range[1])


def _preprocessing(points, dim_points, non_dim_points, dim):
    if not dim_points:
        return None
    m_index = (len(dim_points) - 1) // 2
    m = dim_points[m_index]

    left_dim_points, right_dim_points = dim_points[:m_index], dim_points[m_index + 1:]
    left_non_dim_points, right_non_dim_points = _sorted_list_split(left_dim_points, right_dim_points, non_dim_points)
    next_dim = Dim.HORIZONTAL if dim == Dim.VERTICAL else Dim.VERTICAL

    node = TreeNode()
    node.point_index = m
    node.point = points[m]
    node.line_dim = dim
    node.left = _preprocessing(points, left_non_dim_points, left_dim_points, next_dim)
    node.right = _preprocessing(points, right_non_dim_points, right_dim_points, next_dim)
    return node


def preprocessing(points):
    x = y = list(range(len(points)))
    x = sorted(x, key=lambda i: points[i][0])
    y = sorted(y, key=lambda i: points[i][1])
    return _preprocessing(points, x, y, Dim.VERTICAL)


def _range_search(node, x_range, y_range, res):
    left, right = x_range if node.line_dim == Dim.VERTICAL else y_range
    m = _m(node)
    if left <= m <= right:
        if _is_inside_rect(node.point, x_range, y_range):
            res.append([node.point_index, node.point])
    if node.left and left < m:
        _range_search(node.left, x_range, y_range, res)
    if node.right and m < right:
        _range_search(node.right, x_range, y_range, res)


def range_search(tree, x_range, y_range):
    res = []
    _range_search(tree, x_range, y_range, res)
    return res


def read_points(filename):
    points = []
    with open(filename) as f:
        input_lines = f.readlines()
        for line in input_lines:
            x, y = line.split()
            points.append((float(x), float(y)))
        return points


def read_region(filename):
    with open(filename) as f:
        line1 = f.readline()
        x1, x2 = line1.split()
        x = [float(x1), float(x2)]
        line2 = f.readline()
        y1, y2 = line2.split()
        y = [float(y1), float(y2)]
        return x, y


def init():
    points = read_points("points.txt")
    x_region, y_region = read_region("regions.txt")

    fig, ax = plt.subplots(2)
    ax[0].set_xlim([0, 9])
    ax[0].set_ylim([0, 9])
    ax[1].set_xlim([0, 9])
    ax[1].set_ylim([0, 9])

    rect = patches.Rectangle((x_region[0], y_region[0]), x_region[1] - x_region[0], y_region[1] - y_region[0], linewidth=1, edgecolor='b',
                             facecolor='none')
    ax[0].add_patch(rect)

    for point in points:
        circle = patches.Circle((point[0], point[1]), radius=0.051, color='b')
        ax[0].add_patch(circle)

    rect2 = patches.Rectangle((x_region[0], y_region[0]), x_region[1] - x_region[0], y_region[1] - y_region[0], linewidth=1, edgecolor='b',
                              facecolor='none')
    ax[1].add_patch(rect2)
    tree = preprocessing(points)
    result = range_search(tree, x_region, y_region)
    for point in result:
        circle = patches.Circle((point[1][0], point[1][1]), radius=0.051, color='b')
        ax[1].add_patch(circle)

    plt.show()


init()
