from pyDR.GUI.designer2py.sensitivity_widget import Ui_Sensitivity
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas

class Ui_Sensitivity_final(Ui_Sensitivity):
    def retranslateUi(self, Sensitivity):
        super().retranslateUi(Sensitivity)
        #
        #
        #
        self.plot, ax, toolbar  = create_Figure_canvas()
        self.layout_plot.addWidget(self.plot)
        self.layout_plot.addWidget(toolbar)
        
        
        for i in ["15N","13C","CO","CH3","CHD2"]:
           self.comboBox_nuctype.addItem(i)
