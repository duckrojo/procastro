from Synchronization import *


class Observer(object):
    """Implement Observer object for Observer Pattern."""

    def __init__(self, name, elements):
        """
        :param name: String. Label or name of current Observer.
        :param elements: List. Elements to be observed by this Observer.
        :return: None
        """
        self.name = name
        self.elements = elements

    def update(self, obs, label, value):
        """Update observer.

        Called when the observed object is modified. You call an Observable object's
        C{notifyObserver} method to notify all the object's observers of the change.

        :param obs: ???
        :param label: String. Name of updated data element.
        :param value: New data value for label.
        :return: None"""
        if self.elements[label][1] == 1:
            self.elements[label][0] = value
            print('Observer ' + str(self.name) + ' has changed data ' + str(label))
        print self.elements
        pass

class Observable(Synchronization):
    """Implement Observable object for Observer Pattern."""

    def __init__(self):
        """ Initialize Observable object.

            Set them to unchanged by default and synchronize them. Changed = 0 if unchanged."""
        self.obs = []
        self.changed = 0
        #self.name = name
        Synchronization.__init__(self)

    def addObserver(self, observer):
        """ Add new Observer of Observable.
            :param observer: Observer
            :return: None"""
        if observer not in self.obs:
            self.obs.append(observer)
            print('New Observer added: ' + str(observer.name))
        else:
            print('Observer ' + str(observer.name) + ' already in observers list.')

    def deleteObserver(self, observer):
        """Delete Observer.
            :param observer: Observer to be deleted.
            :return: None"""
        print('Observer has been deleted.')
        try:
            self.obs.remove(observer)
        except ValueError:
            print(str(observer.name) + ' not in observers list. Cannot be removed.')

    def notifyObservers(self, label, value, arg=None):
        """Notify all Observers that a change has occurred.

        If 'changed' indicates that this object has changed, notify all its observers,
        then call clearChanged(). Each Observer has its update() called with two arguments:
        this Observable object and the generic 'arg'.

        :param label: String. Name of updated data element.
        :param value: new data element.
        :param arg: possible arguments for update().
        :return: None"""
        self.mutex.acquire()
        try:
            if not self.changed: return
            # Make a local copy in case of synchronous
            # additions of observers:
            localArray = self.obs[:]
            self.clearChanged()
        finally:
            self.mutex.release()
        for observer in localArray:
            if label in observer.elements:
                print('Observer has been updated.')
                print label, value
                observer.update(self, label, value)
            else:
                raise ValueError('No Observers to update.')


    def deleteObservers(self):
        """Delete all Observers from Observer.
            :return: None"""
        print("All observers have been deleted.")
        self.obs = []

    def setChanged(self):
        """Set Observable as changed.
            :return: None"""
        if self.changed != 1:
            self.changed = 1
        else:
            raise Exception('Error: Observable was already set as changed. Possible code error.')

    def clearChanged(self):
        """Set Observable as unchanged.

            Called after all Observers have been updated after a change.

            :return: None"""
        if self.changed != 0:
            self.changed = 0
        else:
            raise Exception('Error: Observable was already set as unchanged.'
                            'clearChanged() shouldn\'t have been called.')

    def hasChanged(self):
        """Check if Observable has changed.
            :return: None"""
        return self.changed

    def countObservers(self):
        """Return number of current Observers.
            :return: int """
        return len(self.obs)

synchronize(Observable,
  "addObserver deleteObserver deleteObservers " +
  "setChanged clearChanged hasChanged " +
  "countObservers")
