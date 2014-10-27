# Class support for "Observer" pattern.
from Synchronization import *


class Observer(object):
    def __init__(self, name, elements):
        self.name = name
        self.elements = elements

    def update(observable, arg):
        """Called when the observed object is
        modified. You call an Observable object's
        notifyObservers method to notify all the
        object's observers of the change."""
        pass

class Observable(Synchronization):
    def __init__(self):
        """ Initializes list of Observable objects.
            Sets them to unchanged by default and
            synchronizes them.
            changed = 0 if unchanged """
        self.obs = []
        self.changed = 0
        #self.name = name
        Synchronization.__init__(self)

    def addObserver(self, observer):
        """ Adds new Observer of Observable """
        if observer not in self.obs:
            self.obs.append(observer)
            print("New Observer added.")
            #print self.obs

    def deleteObserver(self, observer, name):
        """ Deletes Observer """
        print("Observer " + name + " has been deleted.")
        self.obs.remove(observer)

    def notifyObservers(self, label = None, arg = None):
        """ If 'changed' indicates that this object
        has changed, notify all its observers, then
        call clearChanged(). Each Observer has its
        update() called with two arguments: this
        observable object and the generic 'arg'."""

        self.mutex.acquire()
        try:
            if not self.changed: return
            # Make a local copy in case of synchronous
            # additions of observers:
            localArray = self.obs[:]
            self.clearChanged()
        finally:
            self.mutex.release()
        # Updating is not required to be synchronized:
        for observer in localArray:
            print(str(observer) + " has been updated.")
            observer.update(self, arg)

    def deleteObservers(self):
        print("All observers have been deleted.")
        self.obs = []

    def setChanged(self):
        self.changed = 1

    def clearChanged(self):
        self.changed = 0

    def hasChanged(self):
        return self.changed

    def countObservers(self):
        return len(self.obs)

synchronize(Observable,
  "addObserver deleteObserver deleteObservers " +
  "setChanged clearChanged hasChanged " +
  "countObservers")
