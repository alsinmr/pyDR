from pyDR.GUI.designer2py.selection_widget import Ui_Selection
from pyDR.GUI.other.elements import openFileNameDialog
import MDAnalysis as MDA
from PyQt5.QtWidgets import QLabel, QSpacerItem, QSizePolicy, QPushButton
from PyQt5.QtCore import Qt, QRect

class Ui_Selection_final(Ui_Selection):
    def retranslateUi(self, Selection):

        def get_segments_of_pdb():
            pass

        def get_residues_of_segment():
            pass

        def load_pdb():
            self.listWidget_pdbresidues.clear()
            pdbpath = openFileNameDialog(filetypes="*.pdb")
            self.label_pdbpath.setText(pdbpath)
            Uni = MDA.Universe(pdbpath)
            for res in Uni.residues:
                self.listWidget_pdbresidues.addItem(f"{res.resname}-{res.resid} in Segment({res.segid})")

        super().retranslateUi(Selection)
        self.parent = Selection.parent()
        self.pushButton_loadpdb.clicked.connect(lambda: load_pdb())
        self.nmrlabels = []



        def insert(label):
            index  = self.nmrlabels.index(label)
            additional_label = QLabel(parent=Selection, text= self.nmrlabels[index].text()+"+")
            self.nmrlabels.insert(index+1, additional_label)
            self.verticalLayout_signals.insertWidget(index+1, additional_label)
            additional_label.setFixedHeight(30)
            button = QPushButton(parent=Selection, text ="+")
            button.setFixedWidth(20)
            button.setFixedHeight(additional_label.size().height())
            button.clicked.connect(lambda e, label = additional_label: insert(label))
            self.verticalLayout_copysignal.insertWidget(index+1,button)

        for signal in [f"Res{x}" for x in range(5)]:  #  todo exchange this list by a function that reads out NMRfile
            #create a QLabel for every signal and a an additional label to assign a bond of the pdb
            nmrlabel = QLabel(parent=Selection, text=signal)
            self.nmrlabels.append(nmrlabel)
            self.verticalLayout_signals.addWidget(nmrlabel)
            nmrlabel.setFixedHeight(30)

            #todo
            # add layout for segment seleciton
            # add layout for residue selection
            # add layout for atom1 selection
            # add layout for atom2 selection

            # create a button for every set of Labels which will assign the selected value of the selected item in the box
            button = QPushButton(parent=Selection, text ="+")
            button.setFixedWidth(20)
            button.setFixedHeight(self.nmrlabels[-1].size().height()) #todo get the value from the height of the labels

            #button.clicked.connect(lambda e, label = self.pdbassign[-1]: change_label(label))
            #button.clicked.connect(lambda e, label = self.pdbassign[-1]: label.setText(self.listWidget_pdbresidues.selectedItems()[0].text()))

            button.clicked.connect(lambda e, label = nmrlabel: insert(label))
            self.verticalLayout_copysignal.addWidget(button)


        # add QSpacerItems so the labels and buttons will always stick to the top of the vertical layouts
        for layout in [self.verticalLayout_signals, self.verticalLayout_copysignal, self.verticalLayout_assignatom1,
                self.verticalLayout_assignatom2, self.verticalLayout_assignresidue, self.verticalLayout_assignsegment]:

            layout.addItem(QSpacerItem(20, 40, QSizePolicy.MinimumExpanding,QSizePolicy.Expanding))
        #
        #
