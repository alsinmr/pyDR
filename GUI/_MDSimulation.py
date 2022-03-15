from pyDR.GUI.designer2py.mdsimulation_widget import Ui_MDSimulation
from pyDR.GUI.other.elements import openFileNameDialog, get_workingproject

"""
This Tab should have the following functions:

- adding .xtc files that are available for your project and listing them
- selecting bonds/dihedral angles to observe during the simulations
- calculate total Correlation functions for bonds

"""


class Ui_MDSimulation_final(Ui_MDSimulation):
    def retranslateUi(self, MDSimulation):
        super().retranslateUi(MDSimulation)
        self.parent = MDSimulation.parent()
        self.working_project = get_workingproject(self.parent)
        self.load_from_project()
        self.pushButton_addxtc.clicked.connect(
            lambda e, t=self.listWidget_xtcs: openFileNameDialog(filetypes="*.xtc", title="open .xtc file", target=t))
        self.pushButton_addpdb.clicked.connect(
            lambda e, t=self.listWidget_pdbs: openFileNameDialog(filetypes="*.pdb", title="open .pdb file", target=t))

    def load_from_project(self):
        pass
        #todo load already to the project attached xtc and pdb files
        # idk if there is already a functionality in the infos? -K

