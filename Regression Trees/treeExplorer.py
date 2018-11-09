from numpy import *
from tkinter import *
import RegTree
import matplotlib
matplotlib.use('TkAgg')             # TkAgg is between Tkinter and matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        # if tolN < 2: tolN = 2
        # myTree = RegTree.createTree()
        pass
    else:
        myTree = RegTree.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = RegTree.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].T.tolist()[0], reDraw.rawDat[:, 1].T.tolist()[0], s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()

def drawNewTree():
    tolN, tolS = int(tolNEntry.get()), float(tolSEntry.get())
    reDraw(tolS, tolN)

root = Tk()
#Label(root, text='Plot Place Holder').grid(row = 0, columnspan=3)
reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text='tolN').grid(row = 1, column=0)
tolNEntry = Entry(root)
tolNEntry.grid(row=1, column=1)
tolNEntry.insert(0, '10')
Label(root, text='tolS').grid(row = 2, column=0)
tolSEntry = Entry(root)
tolSEntry.grid(row=2, column=1)
tolSEntry.insert(0, '1.0')
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)

chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

dir = 'F:\\学习资料\\machinelearninginaction-master\\machinelearninginaction-master\\Ch09\\'
reDraw.rawDat = mat(RegTree.loadDataSet(dir + 'sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop()