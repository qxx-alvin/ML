from numpy import *
import matplotlib
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import operator
from kdtree_search import *
from time import clock


# Read txt file
def file2matrix(filename):
    with open(filename) as f:
        arrays = f.readlines()
        numberOfLines = len(arrays)
        mat = zeros((numberOfLines, 3))
        label = []
        i = 0
        for line in arrays:
            line = line.strip()
            l = line.split('\t')
            mat[i, :] = l[0:3]
            label.append(int(l[-1]))
            i += 1
        return mat, label


# Normalize
def normalize(data):
    m = data.shape[0]
    minVals = data.min(0)
    maxVals = data.max(0)
 #   print(minVals, maxVals)
    ranges = maxVals - minVals
    normdata = data - tile(minVals, (m, 1))
    normdata = normdata / tile(ranges, (m, 1))
    return normdata, ranges, minVals


# Classify
def kNN_bruteForce(dataset, label, x, k):
    datasetSize = dataset.shape[0]
    diff = tile(x, (datasetSize, 1)) - dataset
    sqDiff = diff ** 2
    sqDis = sqDiff.sum(axis=1)
    sortDisInd = sqDis.argsort()
    classCnt = {}
    for i in range(k):
        lb = label[sortDisInd[i]]
        classCnt[lb] = classCnt.get(lb, 0) + 1
    sortClassCnt = sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCnt[0][0]


# Classify with kdtree
def kNN_kdtree(tree, x, k):
    ret = find_nearest(tree, x, k)
    classCnt = {}
    for i in range(k):
        lb = ret[i].label
        classCnt[lb] = classCnt.get(lb, 0) + 1
    sortClassCnt = sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCnt[0][0]


# Test
def datingClassTest(k=1):
    ratio = 0.1
    mat, label = file2matrix(r'datingTestSet2.txt')
    normMat, ranges, mins = normalize(mat)
    m = normMat.shape[0]
    numTest = int(m * ratio)
    errCnt = 0
    for i in range(numTest):
        predictedClass = kNN_bruteForce(normMat[numTest:m, :], label[numTest:m], normMat[i, :], k)
        if predictedClass != label[i]:
            errCnt += 1
    print('Error rate: ', errCnt / numTest * 100, '%')


# Test with kdtree
def datingClassKdTree(k=1):
    ratio = 0.1
    mat, label = file2matrix(r'datingTestSet2.txt')
    normMat, ranges, mins = normalize(mat)
    m = normMat.shape[0]
    numTest = int(m * ratio)

    tree = KdTree(normMat[numTest:m, :], label[numTest:m])
    errCnt = 0
    for i in range(numTest):
        predictedClass = kNN_kdtree(tree, normMat[i, :], k)
        if predictedClass != label[i]:
            errCnt += 1
    print('Error rate: ', errCnt / numTest * 100, '%')


if __name__ == '__main__':
#    mat, label = file2matrix(r'E:\Python\Algorithms\datingTestSet2.txt')
#    fig = plot.figure()
 #   ax = fig.add_subplot(111)
#    ax = plot.subplot(111, projection='3d')
#    c_dict = {1:'r', 2:'b', 3:'g'}
#    color = [c_dict[lb] for lb in label]
#    ax.scatter(mat[:, 0], mat[:, 1], mat[:, 2], c=15*array(label))
#    plot.show()
    t0 = clock()
    datingClassTest(6)
    t1 = clock()
    print('Time: ', t1 - t0, 's')

    t0 = clock()
    datingClassKdTree(6)
    t1 = clock()
    print('Time: ', t1 - t0, 's')

