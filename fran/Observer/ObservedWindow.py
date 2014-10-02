__author__ = 'fran'


from Observer import Observer, Observable


class Data:
    """ This would be the original data/FITS/whatever to be "Observed".
        Aka the Observed or Observable Object """
    def __init__(self):
        """ isOpen: for now basic test for communication with Observers.
            Could mean "it's connected" """
        self.isOpen = 0
        # Notifiers for Observers! open (connect) & close (disconnect)
        self.openNotifier = Data.OpenNotifier(self)
        self.closeNotifier= Data.CloseNotifier(self)

    def open(self):
        """ "opens" data for connections
        """
        self.isOpen = 1
        self.openNotifier.notifyObservers() # Notify!
        self.closeNotifier.open() # Close the notifier

    def close(self):
        """ Close "connection"
        """
        self.isOpen = 0
        self.closeNotifier.notifyObservers()
        self.openNotifier.close()

    def closing(self):
        return self.closeNotifier

    # Notifier classes
    class OpenNotifier(Observable):
        def __init__(self, outer):
            Observable.__init__(self)
            self.outer = outer
            self.alreadyOpen = 0

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


class Eye:
    """ This would be the "windows" that show the original data in different views.
        They are alerted of changes in the original data window.
    """
    def __init__(self, name):
        self.name = name
        self.openObserver = Eye.OpenObserver(self)
        self.closeObserver = Eye.CloseObserver(self)

    class OpenObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, arg):
            print("Eye '" + self.outer.name + "' can see Window")

    class CloseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, arg):
            print("Eye '" + self.outer.name + "' can't see Window")


originalData = Data()
window1 = Eye("First view")
window2 = Eye("Second view")

# We add the observers to be notified when window "opens"
originalData.openNotifier.addObserver(window1.openObserver)
originalData.openNotifier.addObserver(window2.openObserver)

# same but for when window "closes"
originalData.closeNotifier.addObserver(window1.closeObserver)
originalData.closeNotifier.addObserver(window2.closeObserver)

# We "connect" the original data to the windows
originalData.open()
originalData.open() # It's already open, no change

# We close the window: the eyes now don't see it
#originalData.close()

# Open it again
#originalData.open()

# Delete an eye before closing it
#originalData.closeNotifier.deleteObserver(window1.closeObserver)
#originalData.open()

# Closing window: only the eye that's left notifies that doesn't see the window anymore
originalData.close()
