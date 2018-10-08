
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import serial
from multiprocessing import Process, Queue
import time, threading


def display(q, windowWidth):

    win2 = pg.GraphicsWindow(title="brxs")
    win2.setWindowTitle('brxs')
    p2 = win2.addPlot(title="brxs")
    curve = p2.plot(pen='y')

    x_np = []
    y_np = []

    def updateInProc(curve, q, x, y, windowWidth):
        item = q.get()
        x.append(item[0])
        y.append(item[1])
        x = x[np.size(x)-windowWidth:]
        y = y[np.size(y)-windowWidth:]
        curve.setData(x,y)

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: updateInProc(curve, q, x_np, y_np, windowWidth))
    timer.start(0.00000000000001)

    QtGui.QApplication.instance().exec_()

def io(running,q):
    ser = serial.Serial("/dev/ttyACM0", 57600)
    dummy = ser.readline()
    t = 0
    while running.is_set():
        line = ser.readline()
        line = line.decode("utf-8")
        line = line.split(",")[:19]
        try:
            y = int(float(line[analogPort]))
        except:
            y = 0
        t += 0.0165
        q.put([t,y])
        time.sleep(0.0000000000000000000001)
    print("Done")

def qtPlotSerialData(windowWidth=1000):
    global analogPort
    analogPort = 0  # port data to be plotted
    q = Queue()
    run = threading.Event()
    run.set()

    t = threading.Thread(target=io, args=(run,q))
    t.start()

    p = Process(target=display, args=(q, windowWidth))
    p.start()
    while(1):
        port = input("port? ")
        try:
            port = int(port)
        except ValueError:
            port = 0
        if ((port >= 0) & (port < 19)):
                analogPort = port

if __name__ == '__main__':
    qtPlotSerialData(1000)