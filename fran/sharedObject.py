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
            self.shared[label][0] = newdata
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
    def __init__(self, name, elements):
        Observer.__init__(self, name, elements)
        #self.openObserver = Eye.OpenObserver(self)
        #self.closeObserver = Eye.CloseObserver(self)



# Diccionarios con elementos a compartir
# Dummy values
# label: [valor, on/off]
shared_elem = {'fulldata': [10, 1], 'xlim': [5, 1], 'ylim': [7, 1]}
eye1_elem = {'xlim': 0}
eye2_elem = {'ylim': 0}

s = sharedObject(1, shared_elem)
e1 = Eye("Eye1", eye1_elem)
e2 = Eye("Eye2", eye2_elem)

#s.openNotifier.addObserver(e1.openObserver)
s.open()
s.addObserver(e1)
s.addObserver(e2)
#s.close()
s.changeData('ylim', 2)
#s.deleteObserver(e1)
#s.close()
#s.open()
#s.changeData('xlim',3)
print s.countObservers()
