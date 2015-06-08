__author__ = 'fran'

from observer.Observable import Observable
from observer.Observer import Observer
import dataproc as dp
import matplotlib.pyplot as plt


class Subject(Observable):
    """Implement Observer pattern. A Subject is an Observable with data and elements to share."""

    def __init__(self, data, elements):
        """Initialize Subject (Observable).

        Data corresponds to the actual observed data. It can be an Astrofile, Scipy array,
        or anything really. Initialize as default Observable, added data, elements and open
        status.

        :param data: actual observed data.
        :param elements: dictionary with elements the Observable is willing to show.
        :return: None
        """
        Observable.__init__(self)
        self.data = data
        self.shared = elements  # Elements the Subject allows Observers to see
        self.isOpen = 0  # Subject is initialized as 'closed'

    def open(self):  # Makes Subject seen to Observers
        """Open Observable to be observed.
            :return: None"""
        if self.isOpen != 1:
            self.isOpen = 1
            print('Observable is open.')
        else:
            raise Exception('Observable is already open. Can\'t  be opened again!')

    def close(self):  # Makes Subject unseen to Observers
        """Closes Observable to Observers.
            :return: None"""
        if self.isOpen != 0:
            self.isOpen = 0
            print('Observable is closed.')
        else:
            raise Exception('Observable is already closed. Can\'t  be closed again!')

    def set_data(self, new_data):
        """Changes data of Subject. This will be applied when a mathematical change,
            filter, addition, etc is carried over the Observable image (Subject data).
            :param new_data: new data value.
            :return: None"""
        self.data = new_data
        print('Observable data has changed.')

        # Only if the Observable is open, will the Observers be notified of the data change
        if self.isOpen == 1:
            self.notify_observers('data', new_data)
        else:
            print('Observable is closed. Observers not be notified of data change.')

    def get_data(self):
        """ Returns current data state of the Subject.
            :return: self.data
        """
        if self.isOpen == 1:
                return self.data


class Eye(Observer):
    """Implement Observer pattern. An Eye is an Observer."""

    def __init__(self, obj, name, shared, elements):
        """ Initialize Eye (Observer). Import default and user-defined methods.

        :param name: String. Label or name of current Eye.
        :param shared: Dictionary with possible elements to be seen from Observable.
        :param elements: Elements to be observed by this Observer.
        :param commandsfile:
        :return: None
        """
        Observer.__init__(self)
        self.name = name
        self.elements = elements
        self.obj = obj
        self.shared = shared
        self.seen = {}
        self.default = __import__('defaultfuncs', globals(), locals(), [], -1)
        self.user = __import__('user', globals(), locals(), [], -1)

        # Preproceso diccionario + lista para registrar que quiero ver y que no
        try:
            for key, value in shared.iteritems():
                if key in elements:
                    self.seen[key] = [value, 1]  # Seen
                else:
                    self.seen[key] = [value, 0]  # Not seen
            #print self.seen
            # Para tomar los eventos definidos por el usuario
            self.events = self.setEvents(self.default, self.user)
            Observer.__init__(self, name, self.seen)
        except ValueError:
            print('Not possible to initialize Observer. Check shared elements.')

    def update(self):
        print("update")


    # Para empezar a "ver" nuevos parametros
    def see(self, new):
        """Add new elements to be seen by the Observer.

        A value of 1 on an element meens it's being seen by the Observer. A value of 0
        means the Observer won't be attending to changes on that element.

        :param new: String. Label or name of new element to be seen.
        :return: None"""
        try:
            for n in new:
                self.seen[n] = [self.shared[n], 1]
        except ValueError:
            print('Error encountered when adding to seen list. Check shared elements.')

    # Para dejar de ver parametros
    def unsee(self, new):
        """Stop seeing certain elements.

        :param new: Sting. Label or name of element to be unseen.
        :return: None"""
        for n in new:
            try:
                self.seen[n][1] = 0
            except ValueError:
                print(str(n) + ' cannot be unseen. Possibly not being seen already.')

    # Para listar los metodos/funciones que se pueden usar
    def setEvents(self, default, user):
        """Get events from default and user-defined packages and add
            them to list of available events to execute.
            :return: List of events (String of each event name).
        """
        import inspect

        events = []

        # defaultfuncs
        for name, data in inspect.getmembers(default, inspect.isfunction):
            if name == '__builtins__':
                continue
            events.append(name)

        # user
        for name, data in inspect.getmembers(user, inspect.isfunction):
            if name == '__builtins__':
                continue
            events.append(name)

        return events

    # Para aplicar eventos
    def apply(self, event, args=None):
        """Apply event, through Observer, to Observable.

        Possible events to apply are given by /default and /user package imports.
        All functions in those packages are available to be applied.

        :param event: String. Name of event.
        :return: Changed data.
        """
        if event in self.events:
            try:
                methodToCall = getattr(self.default, event)
            except AttributeError:
                methodToCall = getattr(self.user, event)
            methodToCall()
        else:
            raise ValueError('Method ' + event + ' not in available method list.')

    def listMethods(self, ini=None):
        if ini is None:
            for e in self.events: print(e)
        else:
            l = len(ini)
            for e in self.events:
                if e[:l] == ini:
                    print(e)

    def updateEye(self, data=False):
        print('in update')
        #if data == False:
        #    self.obj.canvas.draw()
        #else:
        #print(data)
        self.obj.set_data(data)
        plt.draw()

    def changeView(self, changes):
        try:
            for key, value in changes.iteritems():
                if self.seen[key][1] == 1:
                    self.obj.set(key=value)
                else:
                    raise ValueError('Value to be changed not seen by Observer')
            self.updateEye()
        except ValueError:
            print('Not possible to change Observer. Check items and values.')




# Diccionarios con elementos a compartir
# Dummy values
# label: [valor, on/off]
'''af = dp.AstroFile()
shared_elem = {'fulldata': 10, 'xlim': 5, 'ylim': 7, 'zoom': 50}
eye1_elem = ['zoom']
eye2_elem = ['ylim']

s = sharedObject(af, shared_elem)

#cfiles = open('commands', 'r')
e1 = Eye('Eye1', shared_elem, eye1_elem)
e1.listMethods()'''