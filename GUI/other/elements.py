import PyQt5.QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog


def openFileNameDialog():
    W = QWidget()
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    # TODO change the file things
    fileName, _ = QFileDialog.getOpenFileName(W,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
    if fileName:
        return fileName
    return ""
    
    
def create_Figure_canvas(layout: PyQt5.QtWidgets.QVBoxLayout):
    """
    printing a matplotlib canvas on a vertical QVBoxLayout
    returning the figure, the axis and the toolbar (dont know if one really needs toolbar)


    :param layout: QVerticalLayout
    :return:
    """
    canvas = FigureCanvasQTAgg()
    # todo add arguments to funciton to create a figure with multiple plots
    #  in that case, make ax a list to return
    ax = canvas.figure.add_subplot()
    toolbar = NavigationToolbar2QT(canvas,canvas)

    layout.addWidget(canvas)
    layout.addWidget(toolbar)

    return canvas, ax, toolbar
