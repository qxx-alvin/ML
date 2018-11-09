# 来自《机器学习实战》 第9章 树回归
import numpy as np

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)
    return dataMat

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:         # dataSet is a matrix
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)  # old square error
    bestS = np.inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]): # every possible value
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN: continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if bestS == np.inf:             # can't split because of size of leaf
        return None, leafType(dataSet)
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    return bestIndex, bestValue

dir = 'F:\\学习资料\\machinelearninginaction-master\\machinelearninginaction-master\\Ch09\\'
myDat = loadDataSet('F:\\学习资料\\machinelearninginaction-master\\machinelearninginaction-master\\Ch09\\ex2.txt')
myMat = np.mat(myDat)
myTree = createTree(myMat, ops=(0, 1))

# 剪枝
def isTree(obj):
    return type(obj).__name__ == 'dict'

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if testData.shape[0] == 0: return getMean(tree)         # no test data there, merge the whole sub-tree
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)  # check again
    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0 # after merge
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean             # value after merge
        else: return tree       # no merge
    else: return tree

print(myTree)
myDatTest = loadDataSet("F:\\学习资料\\machinelearninginaction-master\\machinelearninginaction-master\\Ch09\\ex2test.txt")
myMatTest = np.mat(myDatTest)

prune(myTree, myMatTest)
print(myTree)

def regTreeEval(model, inDat):
    return float(model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return regTreeEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        return treeForeCast(tree['left'], inData, modelEval)
    else:
        return treeForeCast(tree['right'], inData, modelEval)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)           # number of rows of a matrix
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

trainMat = np.mat(loadDataSet(dir + 'bikeSpeedVsIq_train.txt'))
testMat = np.mat(loadDataSet(dir + 'bikeSpeedVsIq_test.txt'))
myTree = createTree(trainMat, ops=(1,20))
yHat = createForeCast(myTree, testMat[:, 0])
print(np.corrcoef(yHat, testMat[:, 1], rowvar=0))

SSE = np.power(yHat - testMat[:, 1], 2).sum()
SSR = np.power(yHat - np.mean(testMat[:, 1]), 2).sum()
SST = np.power(testMat[:, 1] - np.mean(testMat[:, 1]), 2).sum()
R2 = 1 - SSE / SST
print(R2, R2**(1/2))
