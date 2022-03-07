from pyDR.GUI.designer2py.data_widget import Ui_Data
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas, get_mainwindow, get_workingproject
from pyDR.IO import read_file, readNMR, isbinary


class Ui_Data_final(Ui_Data):
    def retranslateUi(self, Data: QWidget) -> None:
        super().retranslateUi(Data)
        # important: connect parent!
        self.parent = Data.parent()

        # connect a function to a button with clicked.connect
        # target is a label, of which the text will be overwritten by the function
        self.loadfileButton.clicked.connect(lambda: openFileNameDialog(target=self.label_filename))

        # create a plot by passing a predefined layout to the function
        self.plot = create_Figure_canvas(self.layout_plot)
        self.pushButton_plotdata.clicked.connect(self.plot_data)

    def plot_data(self) -> None:
        """
        plotting data of a Data object onto the figure of self.plot
        :return: None
        """
        self.working_project = get_workingproject(self.parent)
        filename =self.label_filename.text()
        if filename == "filename":
            return

        self.working_project.append_data(filename)


        for ax in self.plot.figure.get_axes():
            self.plot.figure.delaxes(ax)

        self.working_project[-1].plot(fig=self.plot.figure)
        #todo something is broken here when I try to open another datafiile in the project

        self.plot.draw()

