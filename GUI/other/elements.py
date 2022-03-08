import PyQt5.QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog, QMainWindow


def get_mainwindow(widget):
    while hasattr(widget,"parent"):
        widget = widget.parent()
        if isinstance(widget, QMainWindow):
            return widget
    assert 0, "Widget is not connected to Main Window!"
    
def get_workingproject(widget: QWidget):
    """

    :param widget:
    :return:            a pointer to the actual working project
    """
    while not hasattr(widget, "get_project"):
        if hasattr(widget, "parent"):
            widget = widget.parent()
        else:
            assert 0, "No Project!"
    return widget.get_project()

def openFileNameDialog(**kwargs) -> str:
    """
    :param kwargs:
        folder:         returns a folder name if set to true
        filetypes:      set listed filetypes
        target:         if the target, for example a QLabel has the Attribute "setText", then the result will be insert
                        there
        title:          the title of the dialog window
    :return:            filename (or foldername if folder=True)
    """
    W = QWidget()
    windowtitle = kwargs.get("title") if kwargs.get("title") else ""

    if kwargs.get("folder"):
        #if you only want to open a folder, set the options for this and use another Dialog
        options = QFileDialog.Options(QFileDialog.DirectoryOnly)
        fileName = QFileDialog.getExistingDirectory(W, windowtitle, options=options)
    else:
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        if kwargs.get("filetypes"):
            filetypes = kwargs['filetypes']
        else:
            filetypes = "All Files (*)"
            #"All Files (*);;Python Files (*.py)"

        fileName, _ = QFileDialog.getOpenFileName(W, windowtitle, "",
                                                  filetypes, options=options)

    if fileName:
        if kwargs.get("target"):
            target = kwargs["target"]
            if hasattr(target, "setText"):
                target.setText(fileName)
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
    canvas.figure.add_subplot()
    toolbar = NavigationToolbar2QT(canvas,canvas)

    layout.addWidget(canvas)
    layout.addWidget(toolbar)

    return canvas
