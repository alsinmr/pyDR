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

import pyDR

class QMainWindow(QMainWindow):
    # just a little Method added to the QMainWindow to make the project accessible to all tabs of the Window
    # since the idea is that it is shouldnt be possible to work with the gui without a project open, this throws an
    # assertion if no project is available
    def get_project(self):
        if self.working_project is not None:
            return self.working_project
        assert 0, "No Project available"


class MyWindow(Ui_MainWindow):
    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)

        #todo think about creating GUI AFTER creating a new or loading existing project
        self.parent = MainWindow
        MainWindow.working_project = pyDR.Project.Project("testproject", create=True)
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

        self.actionLoad_Project.triggered.connect(self.load_project)
        #todo add function for save project





    def load_project(self):
        filename = openFileNameDialog()
        self.label_projectname.setText(f"Project: {filename}")

    def create_new_project(self):
        self.parent.working_project = pyDR.Project.Project()




if __name__ =="__main__":
    app = QApplication(sys.argv)
    Window = QMainWindow()
    ui = MyWindow()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())
        
