from pyDR.GUI.designer2py.data_widget import Ui_Data
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas
from pyDR.IO import read_file, readNMR, isbinary


class Ui_Data_final(Ui_Data):
    def retranslateUi(self, Data: QWidget) -> None:
        super().retranslateUi(Data)
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
        filename =self.label_filename.text()
        if filename is "filename":
            return
        data=read_file(filename) if isbinary(filename) else readNMR(filename)
        assert data.source, "Test data source thing"
        for ax in self.plot.figure.get_axes():
            self.plot.figure.delaxes(ax)

        data.plot(fig=self.plot.figure)
        self.plot.draw()

