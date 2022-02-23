from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog


# TODO move this funciton to separate file and add sum arguments
def openFileNameDialog():
    W = QWidget()
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    # TODO change the file things
    fileName, _ = QFileDialog.getOpenFileName(W,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
    if fileName:
        return fileName
    return ""
    
    
def create_Figure_canvas():
    canvas = FigureCanvasQTAgg()
    ax = canvas.figure.add_subplot()
    toolbar = NavigationToolbar2QT(canvas,canvas)
    return canvas, ax, toolbar
