"""
cb, 16.07.2018

 - gui for classifier, for easier labeling
"""
import sys
sys.path.append("../")
from abraxasThree.classifierClass import AbraxasClassifier


def startRecording():
    name = e1.get()
    a.setFileSink(fileSinkName=name, fileSinkPath="../")
    a.startReceiveData()


def stopRecording():
    a.startReceiveData(opt="KILL")


def startClassifier():
    a.loadDumpNormParam()
    a.loadDumpClassifier()
    name = e1.get()
    a.setFileSink(fileSinkName=name, fileSinkPath="../")
    a.startReceiveData()
    a.startLiveClassification()


def stopClassifier():
    a.startLiveClassification(opt="KILL")


def quitApp():
    stopClassifier()
    stopRecording()
    master.quit()


if __name__ == '__main__':

    a = AbraxasClassifier(numIrSensors=10, numFrSensors=2, windowWidth=100, windowShift=10, numFreqs=15, numCoeffs=15,
                          enaStatFeats=True, featNormMethod='stand', kernel='rbf', trainFraction=1, waveletLvl1=True,
                          randomSortTT=False, classSortTT=True)

    from tkinter import *

    master = Tk()
    Label(master, text="File Name:").grid(row=0)

    e1 = Entry(master)

    e1.grid(row=0, column=1)

    Button(master, text='Quit', command=quitApp).grid(row=6, column=0, sticky=W, pady=4)
    Button(master, text='Stop recording', command=stopRecording).grid(row=3, column=0, sticky=W, pady=4)
    Button(master, text='Start recording', command=startRecording).grid(row=3, column=1, sticky=W, pady=4)
    Button(master, text='Stop classification', command=stopClassifier).grid(row=4, column=0, sticky=W, pady=4)
    Button(master, text='Start classification', command=startClassifier).grid(row=4, column=1, sticky=W, pady=4)

    mainloop()

