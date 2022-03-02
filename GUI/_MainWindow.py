import sys
from os import system
system("python3 _convert_design.py")
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QListWidgetItem, QFileDialog, QWidget

from pyDR.GUI.designer2py.mymainwindow import Ui_MainWindow

from pyDR.GUI._Data import Ui_Data_final
from pyDR.GUI._Sensitivity import Ui_Sensitivity_final
from pyDR.GUI._Detectors import Ui_Detectors_final
from pyDR.GUI._iRed import Ui_iRed_final
from pyDR.GUI._Frames import Ui_Frames_final
from pyDR.GUI._Selection import Ui_Selection_final

from pyDR.GUI.other.elements import openFileNameDialog

class MyWindow(Ui_MainWindow):
    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)
       
        # general procedure to connect a Tab of the main_window with the widgets for every page
        # first step: create the object
        # second step: run setup and pass the "tab"
        self.data_tab = Ui_Data_final()
        self.data_tab.setupUi(self.tab_data)

       
        self.sensitivity_tab = Ui_Sensitivity_final()
        self.sensitivity_tab.setupUi(self.tab_sensitivity)
       
        self.detectors_tab = Ui_Detectors_final()
        self.detectors_tab.setupUi(self.tab_detectors)
       
        self.ired_tab = Ui_iRed_final()
        self.ired_tab.setupUi(self.tab_ired)
       
        self.frames_tab = Ui_Frames_final()
        self.frames_tab.setupUi(self.tab_frames)
        
        self.selection_tab = Ui_Selection_final()
        self.selection_tab.setupUi(self.tab_selection)
       
        self.actionQuitProgram.triggered.connect(MainWindow.close)   # this connects the 'action' to QAction!

        self.actionLoad_Project.triggered.connect(lambda: openFileNameDialog())
        #todo add function for save project

if __name__ =="__main__":
    app = QApplication(sys.argv)
    Window = QMainWindow()
    ui = MyWindow()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())
        
