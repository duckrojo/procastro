__author__ = 'fran'

from Observer import Observer, Observable
import matplotlib.pyplot as plt
import numpy as np

class Window(Observable):
    def __init__(self, x, y):
        self.isOpen = 0
        self.openNotifier = Window.OpenNotifier(self)
        self.closeNotifier= Window.CloseNotifier(self)
        # Here we save the data that we'll eventually want to plot in the "main" (observed) window
        self.datax = x
        self.datay = y
        self.sharex = False # We start not sharing axis
    def open(self): # Opens connection
        self.isOpen = 1
        self.openNotifier.notifyObservers()
        self.closeNotifier.open()
        self.sharex = True
    def close(self): # Closes connection
        self.isOpen = 0
        self.closeNotifier.notifyObservers()
        self.openNotifier.close()
        self.sharex = False
    def closing(self): return self.closeNotifier

    class OpenNotifier(Observable):
        def __init__(self, outer):
            Observable.__init__(self)
            self.outer = outer
            self.alreadyOpen = 0
            f, axarr = plt.subplots(2, sharex=True)
            self.subplot = axarr[0]
        def notifyObservers(self):
            if self.outer.isOpen and not self.alreadyOpen:
                self.setChanged()
                Observable.notifyObservers(self)
                self.alreadyOpen = 1
        def close(self):
            self.alreadyOpen = 0

    class CloseNotifier(Observable):
        def __init__(self, outer):
            Observable.__init__(self)
            self.outer = outer
            self.alreadyClosed = 0
        def notifyObservers(self):
            if not self.outer.isOpen and not self.alreadyClosed:
                self.setChanged()
                Observable.notifyObservers(self)
                self.alreadyClosed = 1
        def open(self):
            alreadyClosed = 0

    def plot(self):
        f, axarr = plt.subplots(2, sharex=self.sharex)
        axarr[0].plot(x, y)
        axarr[1].scatter(x, y)
        plt.show()


class Eye(Observer):
    def __init__(self, name):
        self.name = name
        self.openObserver = Eye.OpenObserver(self)
        self.closeObserver = Eye.CloseObserver(self)
        #subplot.scatter(x, y)
    # An inner class for observing openings:
    class OpenObserver(Observer):
        def __init__(self, outer):
            self.outer = outer
        def update(self, observable, arg):
            print("Eye " + self.outer.name + " is observing\n")
    # Another inner class for closings:
    class CloseObserver(Observer):
        def __init__(self, outer):
            self.outer = outer
        def update(self, observable, arg):
            print("Eye " + self.outer.name + " is closed\n")

# Simple data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

window1 = Window(x, y) # I create the main "Observable" with the data I want to plot
eye1 = Eye("Eye 1") # Create first eye
# Connect observable to eye
window1.openNotifier.addObserver(eye1.openObserver)
window1.closeNotifier.addObserver(eye1.closeObserver)
# "Open" observable
window1.open()
window1.plot()
plt.close("all")
window1.close()
window1.plot()