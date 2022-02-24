from pyDR.GUI.designer2py.data_widget import Ui_Data
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QFileDialog

import numpy as np
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas


        



class Ui_Data_final(Ui_Data):
    def retranslateUi(self, Data):
       super().retranslateUi(Data)
       def load_file():
           self.label_filename.setText(openFileNameDialog())
           
       # connect a function to a button with this
       self.loadfileButton.clicked.connect(lambda:load_file())
       # be careful if you want to pass arguments, then it should be this way:
       # lambda e, arg = 1 : print(arg)
       # the first argument for the clicked funtion will alway be the click event
       #self.addMiauToListButton.clicked.connect(lambda e, text="miau": add_miau(text))
       
       
       #self.plot = FigureCanvasQTAgg()
       #ax = self.plot.figure.add_subplot(111)
       
       
       #toolbar = NavigationToolbar2QT(self.plot,self.plot)
       
       self.plot, ax, toolbar = create_Figure_canvas(self.layout_plot)
       ax.plot(np.sin(np.arange(0,1,0.01)))

       
