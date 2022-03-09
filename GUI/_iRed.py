from pyDR.GUI.designer2py.ired_widget import Ui_iRed
from pyDR.GUI.other.elements import get_workingproject

class Ui_iRed_final(Ui_iRed):
    def retranslateUi(self, iRed):
        super().retranslateUi(iRed)
        self.parent = iRed.parent()
        #
        #
       
    def load_from_working_project(self):
        self.working_project = get_workingproject(self.parent)
