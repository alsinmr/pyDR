import PyQt5.QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog


def openFileNameDialog(**kwargs):
    W = QWidget()
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    # TODO change the file things

    if kwargs.get("filetypes"):
        filetypes = kwargs['filetypes']
    else:
        filetypes = "All Files (*)"
        #"All Files (*);;Python Files (*.py)"
    fileName, _ = QFileDialog.getOpenFileName(W,"QFileDialog.getOpenFileName()", "",filetypes, options=options)
    if fileName:
        if kwargs.get("target"):
            target = kwargs["target"]
            if hasattr(target, "setText"):
                target.setText(fileName)
        else:
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

    return canvas
