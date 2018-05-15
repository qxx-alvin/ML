from kdtree import *
from math import sqrt
from collections import namedtuple

# information about the current nearest point returned by
result = namedtuple('ResultTuple', 'nearest_point   nearest_dist    nodes_visited   label')

def find_nearest_legacy(tree, point):
    k = len(point)
    # travel the tree rooted at kd_node, cur_dist is the current nearest_dist
    def travel(kd_node, target, cur_dist):

        if kd_node is None:
            return result([0] * k, float('inf'), 0, 0)

        nodes_visited = 1                   # does this mean I have checked the distance between you and me? No, the vector on the split may not be checked.
                                            # this is number of points visited under this kd_node, including the point on this node!
        s = kd_node.split
        pivot = kd_node.ele                 # the vector to check
 #       pos = kd_node.pos

        if target[s] <= pivot[s]:           # choose the semi-plane to check first
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, cur_dist)       # dive into that semi-plane

        nearest = temp1.nearest_point                   # return when leaf is met, or when all points on this plane and the opposite semi-plane have been checked
        cur_dist = temp1.nearest_dist                       # current nearest
        nodes_visited += temp1.nodes_visited            # cumulate
        label = temp1.label
      #  if dist < cur_dist:                             # no need to check, does this mean dist can't be greater than cur_dist?

        temp_dist = abs(pivot[s] - target[s])           # check the vectors on the split and in the opposite semi-plane
        if cur_dist < temp_dist:
            return result(nearest, cur_dist, nodes_visited, label)         # no need to dive there
                                                                    # there is no conflict, if nearest is not set earlier, this statement won't be executed
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))          # distance to the vector on the split
        if temp_dist < cur_dist:
            nearest = pivot
            cur_dist = temp_dist
            label = kd_node.label

        temp2 = travel(further_node, target, cur_dist)              # dive there
        nodes_visited += temp2.nodes_visited                        # cumulate
        if temp2.nearest_dist < cur_dist:                               # continue updating
            nearest = temp2.nearest_point
            cur_dist = temp2.nearest_dist
            label = temp2.label

        return result(nearest, cur_dist, nodes_visited, label)

    # travel the whole tree
    return travel(tree.root, point, float('inf'))

cur_nearest_list = []
# m nearest search
def find_nearest(tree, point, m=1):
    k = len(point)

    global cur_nearest_list
    # reset current nearest list
    cur_nearest_list = [result([0] * k, float('inf'), 0, 0) for _ in range(m)]

    def travel(kd_node, target):

        if kd_node is None:
            return

        s = kd_node.split
        pivot = kd_node.ele                 # the vector to check

        if target[s] <= pivot[s]:           # choose the semi-plane to check first
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        travel(nearer_node, target)       # dive into that semi-plane

        # check the vectors on the split and in the opposite semi-plane
        temp_dist = abs(pivot[s] - target[s])
        if cur_nearest_list[-1].nearest_dist <= temp_dist:
            return

        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))          # distance to the vector on the split
        if temp_dist < cur_nearest_list[-1].nearest_dist:
            cur_nearest_list.pop()
            cur_nearest_list.append(result(pivot, temp_dist, 0, kd_node.label))
            cur_nearest_list.sort(key=lambda x: x.nearest_dist)

        travel(further_node, target)              # check the opposite semi-plane

        return

    # travel the whole tree
    travel(tree.root, point)
    return cur_nearest_list


if __name__ == '__main__':
#    data = readVector(r'C:\Users\Administrator\Desktop\vec.txt')
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    label = [i for i in range(len(data))]
    kd = KdTree(data, label)
    print(find_nearest(kd, [3, 4.5], 3))
