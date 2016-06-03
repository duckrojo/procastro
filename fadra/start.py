__author__ = 'fran'

import sys
from PyQt4 import QtCore, QtGui
from test import Ui_MainWindow

class StartQT4(QtGui.QMainWindow):
    def file_dialog(self):
	    self.ui.textEdit.setText('aaaaaaaaaa')

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # here we connect signals with our slots
        QtCore.QObject.connect(self.ui.pushButton, QtCore.SIGNAL("clicked()"), self.file_dialog)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = StartQT4()
    myapp.show()
    sys.exit(app.exec_())