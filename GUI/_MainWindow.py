import sys
from os import system
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QListWidgetItem, QFileDialog, QWidget
from pyDR.GUI.designer2py.mymainwindow import Ui_MainWindow
from pyDR.GUI._Data import Ui_Data_final
from pyDR.GUI._Sensitivity import Ui_Sensitivity_final
from pyDR.GUI._Detectors import Ui_Detectors_final
from pyDR.GUI._iRed import Ui_iRed_final
from pyDR.GUI._Frames import Ui_Frames_final
from pyDR.GUI._Selection import Ui_Selection_final
from pyDR.GUI._MDSimulation import Ui_MDSimulation_final

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
    def setupUi(self, MainWindow, project):
        # to run the GUI, the project has to be initialized, either by opening an existing one, or by creating a new
        # one. The Project shall then be accessible from all tabs equally
        MainWindow.working_project = project

        super().setupUi(MainWindow)

    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)
        #todo think about creating GUI AFTER creating a new or loading existing project
        self.parent = MainWindow
        self.load_project(init=True)
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

        self.mdsimulation_tab = Ui_MDSimulation_final()
        self.mdsimulation_tab.setupUi(self.tab_md)
       
        self.actionQuitProgram.triggered.connect(MainWindow.close)   # this connects the 'action' to QAction!

        self.actionLoad_Project.triggered.connect(self.load_project)
        self.actionSave_Project.triggered.connect(self.save_project)
        #todo I am not sure if it might be useful to just load another project. Setting this up might be a little
        # complicated because of the hard connections to the gui
        # maybe it is just more useful to run two windows of pyDR

        #todo add function for save project



    def save_project(self):
        self.parent.working_project.save()

    def load_project(self, init=False):
        if not init:
            filename = openFileNameDialog()
            self.label_projectname.setText(f"Project: {filename}")
        else:
            self.label_projectname.setText(f"Project: {self.parent.working_project.name}")


    def create_new_project(self):
        self.parent.working_project = pyDR.Project.Project()



if __name__ =="__main__":
    app = QApplication(sys.argv)
    Window = QMainWindow()
    ui = MyWindow()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())
        
