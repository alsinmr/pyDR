import pyDR.IO
from pyDR.GUI.designer2py.data_widget import Ui_Data
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog

import numpy as np
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas


from pyDR.IO import read_file,readNMR,isbinary

        



class Ui_Data_final(Ui_Data):
    def retranslateUi(self, Data):
        super().retranslateUi(Data)
        # connect a function to a button with this
        self.loadfileButton.clicked.connect(lambda:openFileNameDialog(target=self.label_filename))
        # be careful if you want to pass arguments, then it should be this way:
        # lambda e, arg = 1 : print(arg)
        # the first argument for the clicked funtion will alway be the click event
        #self.addMiauToListButton.clicked.connect(lambda e, text="miau": add_miau(text))
       
       
        #self.plot = FigureCanvasQTAgg()
        #ax = self.plot.figure.add_subplot(111)
       
       
        #toolbar = NavigationToolbar2QT(self.plot,self.plot)
       
        self.plot, ax, toolbar = create_Figure_canvas(self.layout_plot)
        ax.plot(np.sin(np.arange(0,1,0.01)))

        self.pushButton_plotdata.clicked.connect(self.plot_data)

    def plot_data(self):
        filename =self.label_filename.text()
        if filename is "filename":
            return
        print(filename)
#        data = pyDR.IO.read_file(filename)
        data=read_file(filename) if isbinary(filename) else readNMR(filename)
        for ax in self.plot.figure.get_axes():
            self.plot.figure.delaxes(ax)

        data.plot(fig = self.plot.figure)
        self.plot.figure.tight_layout()
        self.plot.draw()

