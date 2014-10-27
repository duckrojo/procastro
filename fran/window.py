from Observer import Observer, Observable

class Window(Observable):
    def __init__(self, name):
        self.isOpen = 0
        self.openNotifier = Window.OpenNotifier(self)
        self.closeNotifier= Window.CloseNotifier(self)
    def open(self): # Opens its petals
        self.isOpen = 1
        self.openNotifier.notifyObservers()
        self.closeNotifier.open()
    def close(self): # Closes its petals
        self.isOpen = 0
        self.closeNotifier.notifyObservers()
        self.openNotifier.close()
    def closing(self): return self.closeNotifier

    class OpenNotifier(Observable):
        def __init__(self, outer):
            Observable.__init__(self)
            self.outer = outer
            self.alreadyOpen = 0
        def notifyObservers(self):
            if self.outer.isOpen and \
            not self.alreadyOpen:
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
            if not self.outer.isOpen and \
            not self.alreadyClosed:
                self.setChanged()
                Observable.notifyObservers(self)
                self.alreadyClosed = 1
        def open(self):
            alreadyClosed = 0

class Eye(Observer):
    def __init__(self, name):
        self.name = name
        self.openObserver = Eye.OpenObserver(self)
        self.closeObserver = Eye.CloseObserver(self)

    # An inner class for observing openings:
    class OpenObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, arg):
            print("Eye " + self.outer.name + " is observing")

    # Another inner class for closings:
    class CloseObserver(Observer):
        def __init__(self, outer):
            self.outer = outer
        def update(self, observable, arg):
            print("Eye " + self.outer.name + " stopped observing")

f = Window("Window1")
ba = Eye("Eye1")
bb = Eye("Eye2")
f.openNotifier.addObserver(ba.openObserver)
f.openNotifier.addObserver(bb.openObserver)
# A change that interests observers:
f.open()
#f.open() # It's already open, no change.
#f.closeNotifier.deleteObserver(ba.closeObserver, ba.name)
f.close()
#f.close() # It's already closed; no change
f.openNotifier.deleteObservers()
f.open()
f.close()