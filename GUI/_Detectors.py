from pyDR.GUI.designer2py.detectors_widget import Ui_Detectors
from pyDR.GUI.other.elements import get_workingproject

class Ui_Detectors_final(Ui_Detectors):
    def retranslateUi(self, Detectors):
        super().retranslateUi(Detectors)
        self.parent = Detectors.parent()
        #
        ##
        
        #
    def load_from_working_project(self):
        self.working_project = get_workingproject(self.parent)