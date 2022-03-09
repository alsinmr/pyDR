from pyDR.GUI.designer2py.mdsimulation_widget import Ui_MDSimulation
from pyDR.GUI.other.elements import openFileNameDialog

class Ui_MDSimulation_final(Ui_MDSimulation):
    def retranslateUi(self, MDSimulation):
        super().retranslateUi(MDSimulation)
        self.parent = MDSimulation.parent()
        self.pushButton_addsim.clicked.connect(
            lambda e, t=self.listWidget: openFileNameDialog(filetypes="*.xtc", title="open .xtc file", target=t))
