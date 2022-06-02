#光学测试软件Alpha v0.1







import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from bost_UI import Ui_MainWindow
from tiaoping import Ui_Tiaoping

# Press the green button in the gutter to run the script.




class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.Tiaoping_Dialog=Tiaoping_Window()
        self.flat_button.clicked.connect(self.show_Tiaoping_test)


    def show_Tiaoping_test(self):
        self.Tiaoping_Dialog.show()




class Tiaoping_Window(QMainWindow, Ui_Tiaoping):
    #定义信号
    _signal = QtCore.pyqtSignal(str)
    def __init__(self):
        super(Tiaoping_Window, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
   # MainWindow =QtWidgets.QMainWindow()
    MainWindow = MainWindow()
    #Child = ChildWindow()
    tiaoping = Tiaoping_Window()
   # ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())






