from pyDR.GUI.designer2py.selection_widget import Ui_Selection
from pyDR.GUI.other.elements import openFileNameDialog

class Ui_Selection_final(Ui_Selection):
    def retranslateUi(self, Selection):
        super().retranslateUi(Selection)
        self.pushButton_loadpdb.clicked.connect( lambda: openFileNameDialog(target=self.label_pdbpath, filetypes="*.pdb"))
        #
        #
        #
