from pyDR.GUI.designer2py.frames_widget import Ui_Frames
from pyDR.GUI.other.elements import get_workingproject

class Ui_Frames_final(Ui_Frames):
    def retranslateUi(self,Frames):
        super().retranslateUi(Frames)
        self.parent = Frames.parent()
        #
        #
        #
    def load_from_working_project(self):
        self.working_project = get_workingproject(self.parent)