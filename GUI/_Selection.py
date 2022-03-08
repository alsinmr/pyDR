from pyDR.GUI.designer2py.selection_widget import Ui_Selection
from pyDR.GUI.other.elements import openFileNameDialog
import MDAnalysis as mda
from PyQt5.QtWidgets import QLabel, QSpacerItem, QSizePolicy, QPushButton, QComboBox
from PyQt5.QtCore import Qt, QRect


class Ui_Selection_final(Ui_Selection):
    def retranslateUi(self, Selection):
        self.universe = None

        def get_segments_of_pdb(*args):
            """
            when a pdb and nmr signals are available, this will
            :param
            :return:
            """
            for entry in args:
                entry["segment_box"].clear()
                try:
                    entry["segment_box"].currentIndexChanged.disconnect()
                except TypeError:
                    print("No connection available")
                    # I couldnt find a way to check if a function is connected to that, so i put in this
                    # it is clumsy but it works
                for seg in self.universe.segments:
                    entry["segment_box"].addItem(seg.segid)

                entry["segment_box"].currentIndexChanged.connect(lambda a, e=entry: get_residues_of_segment(e))
                get_residues_of_segment(entry)

        def get_residues_of_segment(entry):
            entry["residue_box"].clear()
            try:
                entry["residue_box"].currentIndexChanged.disconnect()
            except TypeError:
                print("No connection available")
                # same as above

            for res in self.universe.segments[entry["segment_box"].currentIndex()].residues:
                entry["residue_box"].addItem(f"{res.resname}-{res.resid}")
            entry["residue_box"].currentIndexChanged.connect(lambda a, e=entry: get_atoms_of_residue(e))

            get_atoms_of_residue(entry)

        def get_atoms_of_residue(entry):
            if not len(entry["residue_box"].currentText()):
                #for some reason this is called in moments when no entries are available and throws an error
                return
            entry["atom1_box"].clear()
            # todo think about making another function that looks for connected atoms as atom2 and fill the combobox separately
            #  for atom 1 only make heavy atoms available?
            entry["atom2_box"].clear()
            resid = int(entry["residue_box"].currentText().split("-")[-1])
            for atom in self.universe.residues[resid-1].atoms:
                entry["atom1_box"].addItem(atom.name)
                entry["atom2_box"].addItem(atom.name)
            #todo add function that adds the atom numbers to the dicitonary

        def load_pdb():
            #self.listWidget_pdbresidues.clear()
            pdbpath = openFileNameDialog(filetypes="*.pdb")
            self.label_pdbpath.setText(pdbpath)
            self.universe = mda.Universe(pdbpath)
            get_segments_of_pdb(*self.entries)
            #for res in self.universe.residues:
            #    self.listWidget_pdbresidues.addItem(f"{res.resname}-{res.resid} in Segment({res.segid})")

        super().retranslateUi(Selection)
        self.parent = Selection.parent()
        self.pushButton_loadpdb.clicked.connect(lambda: load_pdb())
        self.entries = []

        def add_entry(entry, insert=False):
            """

            :param entry:   is either an already existing entry, if insert=True, or a string
            :param insert:  bool
            :return:        none
            """
            new_entry = {}  # todo refactor new_entry
            if insert:
                index  = self.entries.index(entry)
                new_entry["label"] = QLabel(parent=Selection, text=entry["label"].text() + "+")
                new_entry["add_button"] = QLabel(parent=Selection, text="")
            else:
                index = len(self.entries)
                new_entry["label"] = QLabel(parent=Selection, text=entry)
                new_entry["add_button"] = QPushButton(parent=Selection, text ="+")
                new_entry["add_button"].clicked.connect(lambda e, r_entry=new_entry: add_entry(r_entry, insert=True))

            self.verticalLayout_signals.insertWidget(index + 1, new_entry["label"])
            new_entry["label"].setFixedHeight(30)

            new_entry["segment_box"] = QComboBox(parent=Selection)
            self.verticalLayout_assignsegment.insertWidget(index + 1, new_entry["segment_box"])
            new_entry["segment_box"].setFixedWidth(40)

            new_entry["residue_box"] = QComboBox(parent=Selection)
            self.verticalLayout_assignresidue.insertWidget(index + 1, new_entry["residue_box"])
            new_entry["residue_box"].setFixedWidth(90)

            new_entry["atom1_box"] = QComboBox(parent=Selection)
            self.verticalLayout_assignatom1.insertWidget(index + 1, new_entry["atom1_box"])

            new_entry["atom2_box"] = QComboBox(parent=Selection)
            self.verticalLayout_assignatom2.insertWidget(index + 1, new_entry["atom2_box"])


            new_entry["add_button"].setFixedWidth(20)
            new_entry["add_button"].setFixedHeight(new_entry["label"].size().height())
            self.verticalLayout_copysignal.insertWidget(index + 1, new_entry["add_button"])

            self.entries.insert(index, new_entry)
            if self.universe:
                get_segments_of_pdb(new_entry)

        for signal in [f"Res{x}" for x in range(5)]:  #  todo exchange this list by a function that reads out NMRfile
            add_entry(signal)

        # add QSpacerItems so the labels and buttons will always stick to the top of the vertical layouts
        for layout in [self.verticalLayout_signals, self.verticalLayout_copysignal, self.verticalLayout_assignatom1,
                self.verticalLayout_assignatom2, self.verticalLayout_assignresidue, self.verticalLayout_assignsegment]:
            layout.addItem(QSpacerItem(20, 40, QSizePolicy.MinimumExpanding,QSizePolicy.Expanding))

