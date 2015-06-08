from __future__ import print_function

import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends import qt4_compat
import matplotlib.pyplot as plt
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE

import pyfits as pf

if use_pyside:
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

import subject as so


class MyPopup(QWidget):
    def __init__(self):
        QWidget.__init__(self)


class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent) # Esta bien, QMainWindow NO debe ser Observer, es la ventana general
        # Estas cosas de Observable y Observer yo creo que no van aqui
        self.data, min, max = self.load_data()
        self.observable = so.Subject(self.data, {'min': min, 'max': max})
        #self.observer = so.Eye('zoomwindow', {'min': min, 'max': max}, ['min', 'max'])
        self.observable.open()
        # Inicializa la ventana
        self.create_main_frame()
        self.popups = []

    def create_main_frame(self):
        self.observableWindow = QWidget()

        self.observableFig = Figure((4.0, 4.0), dpi=100)
        self.canvas = FigureCanvas(self.observableFig)
        self.canvas.setParent(self.observableWindow)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Showing initial data on Observable Window
        self.observableFig.clear()
        self.observableAxes = self.observableFig.add_subplot(1, 1, 1)
        min, max = self.observable.get_data('min'), self.observable.get_data('max')
        self.observableAxes.imshow(self.data, vmin=min, vmax=max, origin='lower')

        # Buttons for Observable Window
        hbox = QHBoxLayout()
        self.create_button = QPushButton("&Create Observer")
        #self.connect(self.draw_button, SIGNAL('clicked()'), self.on_create)
        observerCount = 0
        self.create_button.clicked.connect(lambda: self.on_create(observerCount))
        self.scale_button = QPushButton("&Change scale")
        self.connect(self.scale_button, SIGNAL('clicked()'), self.on_scale)
        hbox.addWidget(self.create_button)
        hbox.addWidget(self.scale_button)

        # Para agregar subplot
        self.filter_button = QPushButton("&Apply filter")
        self.connect(self.filter_button, SIGNAL('clicked()'), self.on_filter)
        hbox.addWidget(self.filter_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)  # the matplotlib canvas
        # Barra de herramientas de mpl
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.observableWindow)
        vbox.addWidget(self.mpl_toolbar)

        vbox.addLayout(hbox)

        self.observableWindow.setLayout(vbox)
        self.setCentralWidget(self.observableWindow)

    def load_data(self):
        tfits = pf.getdata('test.fits')
        import core as ma
        min, max = ma.misc_arr.zscale(tfits)
        return tfits, min, max

    def on_create(self, i):
        global observerCount
        self.popups.append(MyPopup())
        print(self.popups, len(self.popups), i)
        self.observerFig = Figure((4.0, 4.0), dpi=100)
        self.canvas = FigureCanvas(self.observerFig)
        self.canvas.setParent(self.popups[i])

        self.observerAxes = self.observerFig.add_subplot(1, 1, 1)
        min, max = self.observable.get_data('min'), self.observable.get_data('max')
        self.observerAxes.imshow(self.data, vmin=min, vmax=max, origin='lower')
        self.observer = so.Eye(self.observerFig, 'Observer 1', {'min': min, 'max': max}, ['min', 'max'])
        self.observable.add_observer(self.observer)

        self.popups[i].show()

        # El zoom
        '''x, y = 0, 0
        def onmove(event):
            if event.button != 1:
                return
            x, y = event.xdata, event.ydata
            self.zoomw.set_xlim(x-100, x+100)
            self.zoomw.set_ylim(y-100, y+100)
            min, max = self.observable.get_data('min'), self.observable.get_data('max')
            self.zoomw.imshow(self.observable.get_data('data'), vmin=min, vmax=max, origin='lower')
            self.canvas.draw()

        self.observableFig.canvas.mpl_connect('button_press_event', onmove)'''

    def on_scale(self):
        min, max = self.observable.get_data('min'), self.observable.get_data('max')
        # Esto deberia ser un cambio en el Eye del axes correspondiente
        #self.observable.set_data('min', min*0.75)
        #self.observable.set_data('max', max*2)
        self.observer.changeView({'min': min*0.75, 'max': max*2})
        #min, max = self.observable.get_data('min'), self.observable.get_data('max')
        # Esto tambien deberia ser parte del Eye
        #self.axes2.imshow(self.data, vmin=min, vmax=max, origin='lower')
        #self.canvas.draw()
        self.observer.update()

    def on_filter(self):
        # Esto esta bien, el cambio de data debe hacerse aca y avisarse a los Eyes
        self.observable.set_data('data', self.data/2)
        # Esto esta mal, el redraw deberia hacerse en un llamado a alguna funcion de Eye
        #min, max = self.observable.get_data('min'), self.observable.get_data('max')
        #self.axes.imshow(self.observable.get_data('data'), vmin=min, vmax=max, origin='lower')
        #self.axes2.imshow(self.observable.get_data('data'), vmin=min, vmax=max, origin='lower')
        #self.canvas.draw()

        # Ya no es necesario llamar al update, se hace solo en observable.set_data
        #self.observer.update()

    def on_click(self, event):
        #print('x=%d, y=%d, xdata=%f, ydata=%f' % (event.x, event.y, event.xdata, event.ydata))
        pass


def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()