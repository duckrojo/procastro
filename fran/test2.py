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
        QMainWindow.__init__(self, parent)
        self.data, self.min, self.max = self.load_data()
        self.observable = so.Subject(self.data)
        self.observable.open()
        self.create_main_frame()  # Starts the main (Observable) window
        self.popups = []  # New windows for Observers will be kept here
        self.observer_main = []

    def create_main_frame(self):
        self.observable_window = QWidget()

        self.observable_fig = Figure((4.0, 4.0), dpi=100)
        self.canvas = FigureCanvas(self.observable_fig)
        self.canvas.setParent(self.observable_window)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Showing initial data on Observable Window
        self.observable_fig.clear()
        self.observable_axes = self.observable_fig.add_subplot(1, 1, 1)
        min, max = self.min, self.max
        self.observable_main = self.observable_axes.imshow(self.data, vmin=min, vmax=max, origin='lower')

        # Buttons for Observable Window
        hbox = QHBoxLayout()
        self.create_button = QPushButton("&Create Observer")
        self.create_button.clicked.connect(lambda: self.on_create(len(self.observable.observers)))
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
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.observable_window)
        vbox.addWidget(self.mpl_toolbar)

        vbox.addLayout(hbox)

        self.observable_window.setLayout(vbox)
        self.setCentralWidget(self.observable_window)

    def load_data(self):
        tfits = pf.getdata('test.fits')
        import core as ma
        min, max = ma.misc_arr.zscale(tfits)
        return tfits, min, max

    def on_create(self, i):
        self.popups.append(MyPopup())
        observer_fig = Figure((4.0, 4.0), dpi=100)
        observer_canvas = FigureCanvas(observer_fig)
        observer_canvas.setParent(self.popups[i])

        observer_axes = observer_fig.add_subplot(1, 1, 1)
        min, max = self.min, self.max
        self.observer_main.append(observer_axes.imshow(self.data, vmin=min, vmax=max, origin='lower'))
        new_observer = so.Eye(observer_fig, 'Observer ' + str(i), {'min': min, 'max': max}, ['min', 'max'])
        self.observable.add_observer(new_observer)

        mpl_toolbar = NavigationToolbar(observer_canvas, self.popups[i])
        vbox = QVBoxLayout()
        vbox.addWidget(observer_canvas)
        vbox.addWidget(mpl_toolbar)
        self.popups[i].setLayout(vbox)

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
        self.observable.set_data(self.data/2)
        new_data = self.observable.get_data()
        self.observable_main.set_data(new_data)
        self.observable_fig.canvas.draw()

        for i in range(len(self.observable.observers)):
            self.observer_main[i].set_data(new_data)
            self.observable.observers[i].obj.canvas.draw()

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