from pyDR.GUI.designer2py.selection_widget import Ui_Selection
from pyDR.GUI.other.elements import openFileNameDialog
import MDAnalysis as MDA
from PyQt5.QtWidgets import QLabel, QSpacerItem, QSizePolicy, QPushButton
from PyQt5.QtCore import Qt, QRect

class Ui_Selection_final(Ui_Selection):
    def retranslateUi(self, Selection):
        def load_pdb():
            self.listWidget_pdbresidues.clear()
            pdbpath = openFileNameDialog(filetypes="*.pdb")
            self.label_pdbpath.setText(pdbpath)
            Uni = MDA.Universe(pdbpath)
            for res in Uni.residues:
                self.listWidget_pdbresidues.addItem(f"{res.resname}-{res.resid} in Segment({res.segid})")

        super().retranslateUi(Selection)
        self.pushButton_loadpdb.clicked.connect(lambda: load_pdb())
        #
        def change_label(label):
            label.setText(self.listWidget_pdbresidues.selectedItems()[0].text())

        self.nmrlabels = []
        self.pdbassign = []
        for signal in [f"Res{x}" for x in range(30)]:  #  todo exchange this list by a function that reads out NMRfile

            self.nmrlabels.append(QLabel(parent=Selection, text=signal))
            self.verticalLayout.addWidget(self.nmrlabels[-1])
            self.nmrlabels[-1].setFixedHeight(30)

            self.pdbassign.append(QLabel(parent=Selection, text=signal))
            self.verticalLayout_2.addWidget(self.pdbassign[-1])
            self.pdbassign[-1].setFixedHeight(30)


            # create a button for every set of Labels which will assign the selected value of the selected item in the box
            button = QPushButton(parent=Selection, text ="<")
            button.setFixedWidth(20)
            button.setFixedHeight(self.nmrlabels[-1].size().height()) #todo get the value from the height of the labels

            button.clicked.connect(lambda e, label = self.pdbassign[-1]: change_label(label))
            self.verticalLayout_3.addWidget(button)

        self.verticalLayout.addItem(QSpacerItem(20, 40, QSizePolicy.MinimumExpanding,QSizePolicy.Expanding))
        self.verticalLayout_2.addItem(QSpacerItem(20, 40, QSizePolicy.MinimumExpanding,QSizePolicy.Expanding))
        self.verticalLayout_3.addItem(QSpacerItem(20, 40, QSizePolicy.MinimumExpanding, QSizePolicy.Expanding))
        #
        #
