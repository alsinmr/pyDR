import matplotlib.pyplot
from pyDR.GUI.designer2py.data_widget import Ui_Data
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas, get_mainwindow, get_workingproject
from pyDR.IO import read_file, readNMR, isbinary
import numpy as np
import matplotlib.pyplot as plt

class Ui_Data_final(Ui_Data):
    def retranslateUi(self, Data: QWidget) -> None:
        super().retranslateUi(Data)
        # important: connect parent!
        self.parent = Data.parent()
        self.load_from_working_project()
        # connect a function to a button with clicked.connect
        # target is a label, of which the text will be overwritten by the function
        self.loadfileButton.clicked.connect(lambda: openFileNameDialog(target=self.label_filename))

        # create a plot by passing a predefined layout to the function
        self.plot = create_Figure_canvas(self.layout_plot)
        self.working_project.add_fig(self.plot.figure)
        self.pushButton_plotdata.clicked.connect(self.plot_data)
        self.pushButton_clear.clicked.connect(self.clear_button)

    def plot_data(self) -> None:
        """
        plotting data of a Data object onto the figure of self.plot
        :return: None
        """
        assert len(self.working_project.titles), "No data available, please append data"
        self.working_project = get_workingproject(self.parent)
        style = self.comboBox_plotstyle.currentText()
        errorbars = self.checkBox_errorbars.checkState()
        self.working_project[self.listWidget_dataobjects.currentIndex().row()].plot(style=style, errorbars=errorbars)
        self.plot.draw()

    def clear_button(self):
        self.working_project.close_fig('all')
        self.plot.figure = plt.figure()
        self.working_project.add_fig(self.plot.figure)
        self.plot.draw()

    def load_from_working_project(self):
        self.working_project = get_workingproject(self.parent)
        for title in self.working_project.titles:
            self.listWidget_dataobjects.addItem(title)


