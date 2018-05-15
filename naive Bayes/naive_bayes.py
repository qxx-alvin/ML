import re
import random
from numpy import *

def getWordVec(filename):
    print(filename)
    with open(filename, encoding='utf-8') as f:
        text = f.read()
    wordList = re.split('\W+', text)
#    return list(map(str.lower, wordList))
    return [word.lower() for word in wordList if len(word) > 2]


def getWordList(vecList):
    wordList = set([])
    for vec in vecList:
        wordList = wordList | set(vec)
    return list(wordList)


def wordExistence(vecList, wordList):
    res = [0] * len(wordList)
    for i in range(len(wordList)):
        if wordList[i] in vecList:
            res[i] = 1
    return res


def naiveBayes(dataset, xRanges, yRange):
    # dataset: list of ([x1, x2, ..., xn], y)
    # xRanges: list of list
    # yRanges: list
    n = len(xRanges)                # num of features
#    y = [example[1] for example in dataset]
#    feat = []
#    for i in range(n):
#        feat.append([eg[0][i] for eg in dataset])
    # build dict for xRanges and yRange, so one traverse is enough
    xDictList = []
    for x in xRanges:
        d = {}
        for i in range(len(x)):
            d[x[i]] = i
        xDictList.append(d)
    yDict = {}
    for i in range(len(yRange)):
        yDict[yRange[i]] = i

    # count frequecies
    pY = [0] * len(yRange)
    pXY = [[[0] * len(x) for x in xRanges] for _ in range(len(yRange))]
    for eg in dataset:
        for xi in range(n):
            pXY[yDict[eg[1]]][xi][xDictList[xi][eg[0][xi]]] += 1
        pY[yDict[eg[1]]] += 1

#    print('pY: ', pY)
#    print('pXY: ', pXY)
    # calc probability
    lam = 1
    pY2 = [0] * len(yRange)
    pXY2 = [[[0] * len(x) for x in xRanges] for _ in range(len(yRange))]
    diff = [[[0] * len(x) for x in xRanges] for _ in range(len(yRange))]
    for i in range(len(yRange)):
        for j in range(n):
            for k in range(len(pXY[i][j])):
                pXY2[i][j][k] = pXY[i][j][k] / pY[i]
                pXY[i][j][k] = (pXY[i][j][k] + lam) / (pY[i] + lam * len(xRanges[j]))
                diff[i][j][k] = abs(pXY2[i][j][k] - pXY[i][j][k])
        pY2[i] = pY[i] / len(dataset)
        pY[i] = (pY[i] + lam * sum([len(xi) for xi in xRanges])) / (len(dataset) + lam * len(yRange) * sum([len(xi) for xi in xRanges]))

    print('pY: ', pY)
    print('pXY: ', pXY)

    print('diff: ', diff)
    return pY, pXY, xDictList, yDict


def classify(vec, pY, pXY, xDictList, yRange):
    max = 0
    ind = -1
    for i in range(len(pY)):
        prod = 1
        for xi in range(len(vec)):
            prod *= pXY[i][xi][xDictList[xi][vec[xi]]]
        prod *= pY[i]
        if prod > max:
            max = prod
            ind = i
    return yRange[ind]


def classifyLog(vec, pY, pXY, xDictList, yRange):
    max = float('-inf')
    ind = -1
    for i in range(len(pY)):
        summ = 0
        for xi in range(len(vec)):
            summ += log(pXY[i][xi][xDictList[xi][vec[xi]]])
        summ += log(pY[i])
        if summ > max:
            max = summ
            ind = i
    return yRange[ind]


def spamTest():
    # prepare the dataset
    vecList = []; classList = []; wordList = set([])
    for i in range(25):
        vecList.append(getWordVec('ham\%d.txt'%(i+1)))
        classList.append(0)
        vecList.append(getWordVec('spam\%d.txt'%(i+1)))
        classList.append(1)
    wordList = getWordList(vecList)
    dataset = [(wordExistence(vecList[i], wordList), classList[i]) for i in range(len(vecList))]
    random.shuffle(dataset)

    # train the classifier
    trainNum = 40
    xRanges = [[1, 0] for _ in range(len(wordList))]
    yRange = [0, 1]
    pX, pXY, xDictList, yDict = naiveBayes(dataset[:trainNum], xRanges, yRange)

    # test
    err = 0
    for eg in dataset[trainNum:]:
        pred = classify(eg[0], pX, pXY, xDictList, yRange)
        if pred != eg[1]:
            err += 1
    print('Error num: ', err)


def mushroomTest():
    with open('mushroom.txt') as f:
        data = f.readlines()
    dataset = []
    for line in data:
        vec = line.strip().split(',')
        dataset.append((vec[1:], vec[0]))

    random.shuffle(dataset)

    # check the dataset
    # same = 0
    # for i in range(len(dataset)):
    #     for j in range(i + 1, len(dataset)):
    #         if dataset[i] == dataset[j]:
    #             same += 1
    # print('same: ', same)

    # get xRanges and yRange
    xRanges = []
    for xi in range(len(dataset[0][0])):
        xList = [eg[0][xi] for eg in dataset]
        xRanges.append(list(set(xList)))
    yList = [eg[1] for eg in dataset]
    yRange = list(set(yList))

    # train the classifier
    trainRatio = 0.8
    trainNum = int(len(dataset) * trainRatio)
    testNum = len(dataset) - trainNum
    pX, pXY, xDictList, yDict = naiveBayes(dataset[:trainNum], xRanges, yRange)

    # test
    err = 0
    for eg in dataset[trainNum:]:
        pred = classify(eg[0], pX, pXY, xDictList, yRange)
        if pred != eg[1]:
            err += 1
    print('Train num: ', trainNum)
    print('Tested num: ', testNum)
    print('Error rate: ', err / testNum * 100, '%')


if __name__ == '__main__':
    # dataset = [([1, 6, 4], 6), ([2, 2, 2], 8), ([4, 6, 4], 6), ([1, 2, 4], 1), ([1, 2, 4], 4), ([1, 6, 1], 1)]
    # xr = [[1, 2, 4], [2, 6], [1, 4, 3, 2]]
    # y = [1, 6, 8, 4]
    mushroomTest()


