__author__ = 'fran'

import core as core
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
        self.shared = elements  # Elementos que permite compartir
        self.isOpen = 0  # Se inicializa "cerrado"

    def open(self):  # Abre "conexion"
        self.isOpen = 1
        print("Observable is open.")

    def close(self): # Cierra conexion
        self.isOpen = 0
        print("Observable is closed.\n")

    def changeData(self, label, newdata):  # Cambia los datos
            print("Observable data has changed.")
            print("Old data: " + str(self.shared))
            self.shared[label] = newdata
            print("New data: " + str(self.shared))
            # Solo si esta abierto, aviso a Observers
            if self.isOpen == 1:
                self.notifyObservers(label, newdata)
            else:
                print("Observable is closed, Observers not notified of change!")

    def notifyObservers(self, label, value):
        if self.isOpen == 1:
            print("Notification sent to Observers.")
            self.setChanged()  # Esto es para los mutex
            Observable.notifyObservers(self, label, value)


class Eye(Observer):
    def __init__(self, name, shared, elements):
        self.shared = shared
        self.seen = {}
        # Preproceso diccionario + lista para registrar que quiero ver y que no
        for key, value in shared.iteritems():
            if key in elements:
                self.seen[key] = [value, 1]
            else:
                self.seen[key] = [value, 0]
        print self.seen
        Observer.__init__(self, name, self.seen)

    # Para empezar a "ver" nuevos parametros
    def see(self, new):
        for n in new:
            self.seen[n] = [self.shared[n], 1]

    # Para dejar de ver parametros
    def unsee(self, new):
        for n in new:
            self.seen[n][1] = 0



# Diccionarios con elementos a compartir
# Dummy values
# label: [valor, on/off]
shared_elem = {'fulldata': 10, 'xlim': 5, 'ylim': 7, 'zoom': 50}
eye1_elem = ['zoom']
eye2_elem = ['ylim']

s = sharedObject(1, shared_elem)
e1 = Eye("Eye1", shared_elem, eye1_elem)
e2 = Eye("Eye2", shared_elem, eye2_elem)

#s.openNotifier.addObserver(e1.openObserver)
s.open()
s.addObserver(e1)
s.addObserver(e2)
#s.close()
#s.changeData('ylim', 2)
s.changeData('xlim', 4)
e1.see(['xlim'])
s.changeData('xlim', 3)
s.changeData('zoom', 25)
e1.unsee(['zoom'])
s.changeData('zoom', 20)
#s.deleteObserver(e1)
#s.close()
#s.open()
#s.changeData('xlim',3)
