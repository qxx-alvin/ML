import re

class KdNode():
    def __init__(self, ele, label, split, left, right):
        self.ele = ele                  # k-dimension vector on the node
        self.label = label                  # the vector's index in the dataset
        self.split = split              # index of dimension upon which the dataset is split
        self.left = left                # root node of left subtree
        self.right = right              # root node of right subtree

class KdTree():
    def __init__(self, data, label):
        k = len(data[0])            # dimension of the data
        dataWithLabel = list(zip(data, label))
        # built-in recursive function
        def CreateNode(split, data_set):                    # create a node and split the dataset
            if not data_set:                                # stop condition for the recursion
                return None
            data_set.sort(key=lambda x: x[0][split])           # sort the dataset based on the value of the that dimension
            split_pos = len(data_set) // 2
            median = data_set[split_pos][0]            # vector in the medium
            lb = data_set[split_pos][1]
            split_next = (split + 1) % k                     # cyclic for each axis
            return KdNode(median, lb, split, CreateNode(split_next, data_set[:split_pos]),
                                          CreateNode(split_next, data_set[split_pos + 1:]))         # recursively create the tree

        self.root = CreateNode(0, dataWithLabel)

def readVector(filename):
    data = []
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while line:
            vec = re.split(r'\s+', line)
            data.append(list(map(float, vec)));
            line = f.readline().strip()
    return data

def preorder(root):
    print(root.ele, root.label)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)

if __name__ == '__main__':
#    data = readVector(r'C:\Users\Administrator\Desktop\vec.txt')
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    label = [i for i in range(len(data))]
    kd = KdTree(data, label)
    preorder(kd.root)
