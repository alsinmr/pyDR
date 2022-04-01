import matplotlib.pyplot
from pyDR.GUI.designer2py.data_widget import Ui_Data
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas, get_mainwindow, get_workingproject
from pyDR.chimeraX.CMXRemote import CMXRemote
from time import time, sleep
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
        # todo add function, that the appended data is going to land in the project as well as in the listwidget

        # create a canvas by passing a predefined layout to the function
        self.canvas = create_Figure_canvas(self.layout_plot)
        self.working_project.add_fig(self.canvas.figure)
        self.pushButton_plotdata.clicked.connect(self.plot_data)
        self.pushButton_clear.clicked.connect(self.clear_button)
        self.pushButton_plotchimerax.clicked.connect(self.open_chimerax)

    def plot_data(self) -> None:
        """
        plotting data of a Data object onto the figure of self.canvas
        :return: None
        """
        assert len(self.working_project.titles), "No data available, please append data"
        self.working_project = get_workingproject(self.parent)
        style = self.comboBox_plotstyle.currentText()
        errorbars = self.checkBox_errorbars.checkState()
        self.working_project[self.listWidget_dataobjects.currentIndex().row()].plot(style=style, errorbars=errorbars)
        self.canvas.draw()

    def clear_button(self) -> None:
        """
        closing the figure of the canvas and creating a new one
        there is probably a better way to do it, just clearing the figure won't work
        :return none:
        """
        self.working_project.close_fig('all')
        self.canvas.figure = plt.figure()
        self.working_project.add_fig(self.canvas.figure)
        self.canvas.draw()

    def load_from_working_project(self) -> None:
        self.working_project = get_workingproject(self.parent)

        for title in self.working_project.titles:
            self.listWidget_dataobjects.addItem(title)
            if hasattr(self.working_project[title][0], "select") and getattr(self.working_project[title][0],"select") is not None:
                self.comboBox_selectpdb.addItem(self.working_project[title][0].select.molsys.topo)
        #todo load pdbs and add to combobox
        #todo load selection for pdb and add a combobox

    def open_chimerax(self) -> None:
        dataset = self.listWidget_dataobjects.currentIndex().row()
        scale = self.doubleSpinBox_scaling.value()
        self.working_project[dataset].chimera(scaling=scale)

