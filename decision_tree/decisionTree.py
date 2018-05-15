import operator
from math import log
from plotDecisionTree import *
import pickle
import random
import xlrd

def majority(classlist):
    classnum = {}
    for i in classlist:
        classnum[i] = classnum.get(i, 0) + 1
    sortedClassnum = sorted(classnum.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassnum[0][0]


def calcEnt(dataset):
    numEntry = len(dataset)
    labelCnt = {}
    for i in dataset:
        label = i[-1]
        labelCnt[label] = labelCnt.get(label, 0) + 1
    ent = 0
    for key in labelCnt:
        prob = labelCnt[key] / numEntry
        ent -= prob * log(prob, 2)
    return ent


def subSet(dataset, featInd, feat):
    subset = []
    for vec in dataset:
        if vec[featInd] == feat:
            reducedVec = vec[:featInd]
            reducedVec.extend(vec[featInd + 1:])
            subset.append(reducedVec)
    return subset

def calcCondiEnt(dataset, featInd):
#    featList = [example[featInd] for example in dataset]
#    featSet = set(featList)
    m = len(dataset)
    featCnt = {}
    for example in dataset:
        featCnt[example[featInd]] = featCnt.get(example[featInd], 0) + 1
    cEnt = 0
    for key in featCnt:
        subset = subSet(dataset, featInd, key)
        cEnt += featCnt[key] / m * calcEnt(subset)
    return cEnt


def chooseBestFeat(dataset):
    numFeat = len(dataset[0]) - 1
#    ent = calcEnt(dataset)
    minCondiEnt = float('inf')
    bestFeatInd = -1
    for i in range(numFeat):
        cEnt = calcCondiEnt(dataset, i)
        if cEnt < minCondiEnt:
            minCondiEnt = cEnt
            bestFeatInd = i
    return bestFeatInd



def createTree(dataset, labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):             # end condition 1
        return classlist[0]
    if len(dataset[0]) == 1:                                        # end condition 2
        return majority(classlist)
    bestFeatInd = chooseBestFeat(dataset)
    featList = [example[bestFeatInd] for example in dataset]
    featSet = set(featList)
    bestFeat = labels[bestFeatInd]
    label2 = labels[:]
    del(label2[bestFeatInd])
    tree = {bestFeat:{}}
    for feat in featSet:
        lb = label2[:]
        tree[bestFeat][feat] = createTree(subSet(dataset, bestFeatInd, feat), lb)
    return tree


def classify(tree, featList, testVec):
    rootFeat = list(tree.keys())[0]
    branchDict = tree[rootFeat]
    featInd = featList.index(rootFeat)
    res = ''
    for branch in branchDict.keys():
        if testVec[featInd] == branch:
            if isinstance(branchDict[branch], dict):
                res = classify(branchDict[branch], featList, testVec)
            else:   res = branchDict[branch]
            break
    return res


def storeTree(tree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)


def grabTree(filename):
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    return tree


def test_monk():
    with open('monks_test.txt') as fr:
        content = fr.readlines()
    dataset = []
    for line in content:
        example = line.strip().split(' ')
        example.pop()
        example.append(example[0])
        example.pop(0)
        dataset.append(example)
#    print(dataset)
    labels = list(range(len(dataset[0]) - 1))

    trainNumRatio = 0.4
    trainNum = int(len(dataset) * trainNumRatio)
    testNum = len(dataset) - trainNum
    random.shuffle(dataset)

    tree = createTree(dataset[:trainNum], labels)
#    storeTree(tree, 'tree.txt')
#    tree = grabTree('tree.txt')
#    print(tree)
    createPlot(tree)

    # with open('monks_test.txt') as fr:
    #     content = fr.readlines()
    # dataset = []
    # for line in content:
    #     example = line.strip().split(' ')
    #     example.pop()
    #     example.append(example[0])
    #     example.pop(0)
    #     dataset.append(example)
    err = 0
    for example in dataset[trainNum:]:
        prediction = classify(tree, labels, example)
        if prediction != example[-1]:
            err += 1
    print('Data trained: ', trainNum)
    print('Data tested: ', testNum)
    print('Test error rate: ', err / testNum * 100, '%')


def test_mushroom():
    with open('mushroom.txt') as fr:
        content = fr.readlines()
    dataset = []
    for line in content:
        example = line.strip().split(',')
        example.append(example[0])
        example.pop(0)
        dataset.append(example)
#    print(dataset)
    labels = list(range(len(dataset[0]) - 1))

    trainNumRatio = 0.5
    trainNum = int(len(dataset) * trainNumRatio)
    testNum = len(dataset) - trainNum
    random.shuffle(dataset)

    tree = createTree(dataset[:trainNum], labels)
    createPlot(tree)

    err = 0
    for example in dataset[trainNum:]:
        prediction = classify(tree, labels, example)
        if prediction != example[-1]:
            err += 1
    print('Data trained: ', trainNum)
    print('Data tested: ', testNum)
    print('Test error rate: ', err / testNum * 100, '%')


if __name__ == '__main__':

# load leaf data. Holly shit! these data are continuous
#     data = xlrd.open_workbook('leaf/leaf.xlsx')
#     table = data.sheet_by_name('leaf')
#     dataset = []
#     for i in range(table.nrows):
#         example = table.row_values(i)
#         del(example[1])
#         example.append(int(example[0]))
#         del(example[0])
#         dataset.append(example)
# load labels

#    numExample = len(dataset)
#    trainNumRatio = 1
#    trainNum = int(numExample * trainNumRatio)
#    random.shuffle(dataset)

    test_mushroom()



