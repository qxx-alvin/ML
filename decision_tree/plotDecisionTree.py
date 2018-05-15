import matplotlib.pyplot as plt


decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', va='center', ha='center',
                            bbox=nodeType, arrowprops=arrow_args)

def plotMidText(centerPt, parentPt, txt):
    x = (parentPt[0] - centerPt[0]) / 2 + centerPt[0]
    y = (parentPt[1] - centerPt[1]) / 2 + centerPt[1]
    createPlot.ax1.text(x, y, txt, va='center', ha='center', rotation=30)


def plotTree(tree, parentPt, nodeText):
    numLeaf = getTreeLeafNum(tree)
    depth = getTreeDepth(tree)
    key = list(tree.keys())[0]
    centerPt = (plotTree.xOff + (1 + numLeaf) / 2 / plotTree.totalW, plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeText)
    plotNode(key, centerPt, parentPt, decisionNode)
    val = tree[key]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in val.keys():
        if isinstance(val[key], dict):
            plotTree(val[key], centerPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            plotNode(val[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD


def createPlot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False, xticks=[], yticks=[])
    plotTree.totalW = getTreeLeafNum(tree)
    plotTree.totalD = getTreeDepth(tree)
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1
#    plotNode('Decision', (0.5, 1), (1, 2), decisionNode)
#    plotNode('Left', (0.1, 0.2), (0.4, 0.6), leafNode)
    plotTree(tree, (0.5, 1), '')
    plt.show()


def getTreeLeafNum(tree):
    key = list(tree.keys())[0]
    value = tree[key]
    leafNum = 0
    for key in value.keys():
        if isinstance(value[key], dict):
            leafNum += getTreeLeafNum(value[key])
        else: leafNum += 1
    return leafNum

def getTreeDepth(tree):
    key = list(tree.keys())[0]
    value = tree[key]
    maxDepth = 0
    for key in value.keys():
        if isinstance(value[key], dict):
            depth = 1 + getTreeDepth(value[key])
        else:   depth = 1
        if depth > maxDepth:    maxDepth = depth
    return maxDepth

if __name__ == '__main__':
#    createPlot()
    pass