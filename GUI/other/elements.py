import PyQt5.QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import __version__ as mplvers
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

#def fill_combobox_with_items(box, listofitems: list):
#    for item in listofitems:
#        if not item in box:
#            box.add

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
    target = kwargs.get("target")

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
        if target is not None:
            if hasattr(target, "setText"):
                target.setText(fileName)
            elif hasattr(target, "addItem"):
                target.addItem(fileName)
        return fileName
    return ""
    
    
def create_Figure_canvas(layout: PyQt5.QtWidgets.QVBoxLayout):
    """
    printing a matplotlib canvas on a vertical QVBoxLayout
    returning the figure, the axis and the toolbar (dont know if one really needs toolbar)


    :param layout: QVerticalLayout
    :return:
    """
    if mplvers == '3.3.4':
        #todo find out which version needs that, right now it is my plt at home that is suffering from that -K
        fig = plt.figure()
        canvas = FigureCanvasQTAgg(fig)
    else:
        fig = plt.figure()
        # working computer hat 3.4.3
        canvas = FigureCanvasQTAgg(fig)
    # todo add arguments to function to create a figure with multiple plots
    #  in that case, make ax a list to return
    canvas.figure.add_subplot()
    toolbar = NavigationToolbar2QT(canvas, canvas)

    layout.addWidget(canvas)
    layout.addWidget(toolbar)
    return canvas
