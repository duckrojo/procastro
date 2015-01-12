__author__ = 'fran'

#import core as core
from Observer import Observer, Observable
import core.io_file as dp
import subprocess as sp


class sharedObject(Observable):
    """Implement Observer pattern. A sharedObject is an Observable."""

    def __init__(self, data, elements):
        """Initialize sharedObject (Observable).

        Data corresponds to the actual observed data. It can be an Astrofile, Scipy array,
        or anything really. Initialize as default Observable, added data, elements and open
        status.

        :param data: actual observed data.
        :param elements: dictionary with elements the Observable is willing to show.
        :return: None
        """
        """ Ojo: data puede ser un Astrofile o un Scipy array,
            por eso no especifico por separado el header + data.
            ELEMENTS son los elementos que el observable esta
            "dispuesto" a compartir; si un Eye quiere ver algo que
            el Observable no permite, no puede hacerlo.
        """
        Observable.__init__(self)
        self.data = data
        self.shared = elements  # Elementos que permite compartir
        self.isOpen = 0  # Se inicializa "cerrado"

    def open(self):  # Abre "conexion"
        """Open Observable to be observed.
            :return: None"""
        if self.isOpen != 1:
            self.isOpen = 1
            print('Observable is open.')
        else:
            raise Exception('Observable is already open. Can\'t  be opened again!')

    def close(self): # Cierra conexion
        """Closes Observable to Observers.
            :return: None"""
        if self.isOpen != 0:
            self.isOpen = 0
            print('Observable is closed.')
        else:
            raise Exception('Observable is already closed. Can\'t  be closed again!')

    def changeData(self, label, newdata):  # Cambia los datos
        """Changes data on Observable.

        :param label: label or name of data value to be changed.
        :param newdata: new data value.
        :return: None"""
        print("Observable data has changed.")
        print("Old data: " + str(self.shared))
        self.shared[label] = newdata
        print("New data: " + str(self.shared))
        # Solo si esta abierto, aviso a Observers
        if self.isOpen == 1:
            try:
                self.shared[label] = newdata
                self.notifyObservers(label, newdata)
            except ValueError:
                print('Data not changed. Error received when notifying Observers.')
        else:
            print('Observable is closed, Observers not notified of change!')

    def notifyObservers(self, label, value):
        """Notify all Observes of change.

        :param label: label or name of data value changed.
        :param value: new value for changed data.
        :return: None"""
        if self.isOpen == 1:
            print("Notification sent to Observers.")
            self.setChanged()  # Esto es para los mutex
            try:
                Observable.notifyObservers(self, label, value)
            except ValueError:
                print('Observers couldn\'t be notified of changes.')


class Eye(Observer):
    """Implement Observer pattern. A Eye is an Observer."""

    def __init__(self, name, shared, elements):
        """ Initialize Eye (Observer). Import default and user-defined methods.

        :param name: String. Label or name of current Eye.
        :param shared: Dictionary with possible elements to be seen from Observable.
        :param elements: Elements to be observed by this Observer.
        :param commandsfile:
        :return: None
        """
        self.shared = shared
        self.seen = {}
        self.default = __import__('defaultfuncs', globals(), locals(), [], -1)
        self.user = __import__('user', globals(), locals(), [], -1)

        # Preproceso diccionario + lista para registrar que quiero ver y que no
        try:
            for key, value in shared.iteritems():
                if key in elements:
                    self.seen[key] = [value, 1]
                else:
                    self.seen[key] = [value, 0]
            #print self.seen
            # Para tomar los eventos definidos por el usuario
            self.events = self.setEvents(self.default, self.user)
            Observer.__init__(self, name, self.seen)
        except ValueError:
            print('Not possible to initialize Observer. Check shared elements.')

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


# Diccionarios con elementos a compartir
# Dummy values
# label: [valor, on/off]
#af = dp.AstroFile()
#shared_elem = {'fulldata': 10, 'xlim': 5, 'ylim': 7, 'zoom': 50}
#eye1_elem = ['zoom']
#eye2_elem = ['ylim']

#s = sharedObject(af, shared_elem)

#cfiles = open('commands', 'r')
#e1 = Eye("Eye1", shared_elem, eye1_elem, 'commands')
#e1.apply('bar2')

