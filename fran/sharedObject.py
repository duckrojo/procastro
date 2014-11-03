__author__ = 'fran'

#from dataproc import core, combine
from Observer import Observer, Observable


class sharedObject(Observable):
    def __init__(self, data, elements):
        """ Ojo: data puede ser un Astrofile o un Scipy array,
            por eso no especifico por separado el header + data.
            ELEMENTS son los elementos que el observable esta
            "dispuesto" a compartir; si un Eye quiere ver algo que
            el Observable no permite, no puede hacerlo.
        """
        Observable.__init__(self)
        self.data = data
        self.shared = elements
        self.isOpen = 0  # Se inicializa "cerrado"
        # Creo los Notifiers para los distintos elementos del diccionario
        #self.changeNotifier = sharedObject.ChangeNotifier(self)
        self.openNotifier = sharedObject.OpenNotifier(self)

    def open(self):  # Abre "conexion"
        self.isOpen = 1
        self.openNotifier.notifyObservers()
        print("sharedObject is now sharing data.")
        #self.closeNotifier.open()

    def close(self): # Cierra conexion
        self.isOpen = 0
        #self.closeNotifier.notifyObservers()
        self.openNotifier.close()

    def changeData(self, label, newdata):  # Cambia los datos
        print("Observable data has changed.")
        print("Old data: " + str(self.shared))
        self.shared[label] = newdata
        print("New data: " + str(self.shared))
        #print self.obs
        #  Notifico a todos los Observers, ellos filtran si les sirve la info
        #self.changeNotifier.notifyObservers()
        self.notifyObservers(label, newdata)
        #for n in self.shareDict:
        #    self.shareDict[n].openNotifier.notifyObservers()
        #self.openNotifier.notifyObservers()

    #def closing(self): return self.closeNotifier

    # TODO OJO CON ESTO
    """class ChangeNotifier(Observable):
        def __init__(self, outer):
            Observable.__init__(self)
            self.outer = outer
            self.alreadyOpen = 0"""
    def notifyObservers(self, label, value):
        print("Notification sent to Observers.")
        #if self.outer.isOpen and not self.alreadyOpen:
        self.setChanged()
        #print self.obs
        Observable.notifyObservers(self, label, value)
        self.alreadyOpen = 1
        """def close(self):
            self.alreadyOpen = 0"""

    class OpenNotifier(Observable):
        def __init__(self, outer):
            Observable.__init__(self)
            self.outer = outer
            self.alreadyOpen = 0
        def notifyObservers(self):
            if self.outer.isOpen and \
            not self.alreadyOpen:
                self.setChanged()
                Observable.notifyObservers(self, 0, 0)
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
                Observable.notifyObservers(self, 0, 0)
                self.alreadyClosed = 1
        def open(self):
            alreadyClosed = 0



class Eye(Observer):
    def __init__(self, name, elements):
        Observer.__init__(self, name, elements)
        #self.elements = elements
        self.openObserver = Eye.OpenObserver(self)
        self.closeObserver = Eye.CloseObserver(self)
    # An inner class for observing openings:
    class OpenObserver(Observer):
        def __init__(self, outer):
            self.outer = outer
        def update(self, observable, arg):
            print("Eye " + self.outer.name + " is observing.")
    # Another inner class for closings:
    class CloseObserver(Observer):
        def __init__(self, outer):
            self.outer = outer
        def update(self, observable, arg):
            print("Eye " + self.outer.name + " stopped observing.")


# Diccionarios con elementos a compartir
# Dummy values
shared_elem = {'fulldata': 10, 'xlim': 5, 'ylim': 7}
eye_elem = {'xlim': 0}

s = sharedObject(1, shared_elem)
e1 = Eye("Eye1", eye_elem)
#s.openNotifier.addObserver(e1.openObserver)
s.open()
s.addObserver(e1)
s.changeData('ylim', 2)
print e1.elements
