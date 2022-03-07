from pyDR.GUI.designer2py.sensitivity_widget import Ui_Sensitivity
from pyDR.GUI.other.elements import openFileNameDialog, create_Figure_canvas
from os import listdir
from os.path import join, abspath

def get_defaults() -> list:
    path = abspath(__file__).rsplit("/", 1)[0]

    default_entries = []

    with open(join(path, "..", "Sens", "NMR_defaults.txt")) as f:
        for line in f:
            if "BEGIN DEFAULT" in line:
                entry = []
            elif "END DEFAULT" in line:
                default_entries.append(entry)
            elif len(line.strip()):
                entry.append(line.strip())
    #for entry in default_entries:
    #    print(entry)
    return default_entries



class Ui_Sensitivity_final(Ui_Sensitivity):
    def retranslateUi(self, Sensitivity) -> None:
        super().retranslateUi(Sensitivity)
        self.parent = Sensitivity.parent()
        self.plot  = create_Figure_canvas(self.layout_plot)
        self.layout_plot.layout().setContentsMargins(0,0,0,0)

        for entry in get_defaults():
            self.comboBox_nuctype.addItem(entry[0].split(" ")[0])
        #for i in ["15N","13C","CO","CH3","CHD2"]:
        #   self.comboBox_nuctype.addItem(i)
