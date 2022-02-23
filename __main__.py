from .GUI import MyWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(sys.argv)
    app = QApplication(sys.argv)
    Window = QMainWindow()
    ui = MyWindow()
    ui.setupUi(Window)
    Window.show()

    sys.exit(app.exec_())
