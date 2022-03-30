#!usr/bin/python3
from pyDR.GUI.designer2py.selection_widget import Ui_Selection
from pyDR.GUI.other.elements import openFileNameDialog, get_workingproject
import MDAnalysis as mda
from PyQt5.QtWidgets import QLabel, QSpacerItem, QSizePolicy, QPushButton, QComboBox, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt, QRect
import pyDR
import re


class Ui_Selection_final(Ui_Selection):
    def retranslateUi(self, Selection):
        self.universe = None

        def get_segments_of_pdb(*args):
            """
            when a pdb and nmr signals are available, this will
            :param
            :return:
            """
            for id, entry in enumerate(args):
                entry["segment_box"].clear()
                entry["segment_box"].addItem("")
                for seg in self.universe.segments:
                    entry["segment_box"].addItem(seg.segid)
                get_residues_of_segment(entry)

        def get_residues_of_segment(entry):
            entry["residue_box"].clear()
            entry["residue_box"].addItem("")
            seg = self.universe.segments[self.universe.segments.segids==entry["segment_box"].currentText()]
            if not len(seg):
                return
            for res in seg.residues:
                entry["residue_box"].addItem(f"{res.resname}-{res.resid}")
            if not len(entry["segment_box"].currentText()):
                return
            ### assuming the residue number is somewhere written in the label of the signal, we try to extract the
            # number and search for it in the available residues of the segment
            resnum = re.sub('[^0-9]','',entry["label"].text())
            if len(resnum):
                resnum = int(resnum)
                for index, res in enumerate(seg.residues):
                    if resnum == res.resid:
                        entry["residue_box"].setCurrentIndex(index+1)

            get_atoms_of_residue(entry)

        def get_atoms_of_residue(entry):
            if not len(entry["residue_box"].currentText()):
                #for some reason this is called in moments when no entries are available and throws an error
                return
            entry["atom1_box"].clear()
            # todo think about making another function that looks for connected atoms as atom2 and fill the combobox separately
            #  for atom 1 only make heavy atoms available?
            entry["atom2_box"].clear()
            seg = entry["segment_box"].currentText()
            seg = self.universe.segments[self.universe.segments.segids==seg][0]
            resid = int(entry["residue_box"].currentText().split("-")[-1])
            res = seg.residues[seg.residues.resids==resid][0]
            for atom in res.atoms:#self.universe.residues[resid-1].atoms:
                entry["atom1_box"].addItem(atom.name)
                entry["atom2_box"].addItem(atom.name)
            #todo add function that adds the atom numbers to the dicitonary

        def load_pdb():
            pdbpath = self.comboBox_pdb.currentText()#openFileNameDialog(filetypes="*.pdb")
            print(pdbpath)
            self.universe = mda.Universe(pdbpath)
            get_segments_of_pdb(*self.entries)

        super().retranslateUi(Selection)
        self.parent = Selection.parent()
        self.load_from_working_project()
        self.pushButton_loadpdb.clicked.connect(lambda: load_pdb())

        zero_entry = {"label":QLabel(""),
            "segment_box":self.comboBox,
                      "residue_box": self.comboBox_2,
                      "atom1_box": self.comboBox_3,
                      "atom2_box": self.comboBox_4}
        zero_entry["segment_box"].currentIndexChanged.connect(lambda a, e=zero_entry: get_residues_of_segment(e))
        zero_entry["residue_box"].currentIndexChanged.connect(lambda a, e=zero_entry: get_atoms_of_residue(e))
        zero_entry["residue_box"].setMaxVisibleItems(15)
        zero_entry["residue_box"].setStyleSheet("combobox-popup: 0;")
        self.entries = [zero_entry]

        def apply_to_all_signals(box : str):
            """
            Applying the selection of the zeroentry to all signals below, if it is possible
            :param box:  str -> segment_box or residue_box or atom1_box or atom2_box
            :return:  none
            """
            val = self.entries[0][box].currentText()
            for entry in self.entries:
                entry[box].setCurrentIndex(entry[box].findText(val))

        self.pushButton.clicked.connect(lambda:apply_to_all_signals("segment_box"))
        self.pushButton_2.clicked.connect(lambda:apply_to_all_signals("residue_box"))
        self.pushButton_3.clicked.connect(lambda:apply_to_all_signals("atom1_box"))
        self.pushButton_4.clicked.connect(lambda:apply_to_all_signals("atom2_box"))

        def clear_entries():
            for entry in self.entries[1:]:
                while entry["layout"].count():
                    child = entry["layout"].takeAt(0)#entry["layout"].count()-1)
                    if child.widget():
                        child.widget().deleteLater()
                entry["widget"].deleteLater()
            while len(self.entries)>1:
                self.entries.pop()

        def add_entry(label, insert=False):
            """

            :param label:   is either an already existing entry, if insert=True, or a string
            :param insert:  bool
            :return:        none
            """
            new_entry = {}  # todo refactor new_entry
            if insert:
                index  = self.entries.index(label)
                new_entry["label"] = QLabel(parent=Selection, text=label["label"].text())
                new_entry["label"].setToolTip(label["label"].text())
                new_entry["add_button"] = QLabel(parent=Selection, text="")
            else:
                index = len(self.entries)
                new_entry["label"] = QLabel(parent=Selection, text=label)
                new_entry["add_button"] = QPushButton(parent=Selection, text ="+")
                new_entry["add_button"].clicked.connect(lambda e, r_entry=new_entry: add_entry(r_entry, insert=True))

                new_entry["label"].setToolTip(label)
            new_entry["label"].setContentsMargins(0,0,0,0)
            #self.comboBox_dataset.setToolTip(_translate("Selection", "tesst"))

            new_entry["widget"] = QWidget()
            new_entry["widget"].setContentsMargins(0,0,0,0)
            new_entry["layout"] = QHBoxLayout(new_entry["widget"])
            new_entry["layout"].addWidget(new_entry["label"])
            new_entry["label"].setFixedWidth(60)

            new_entry["segment_box"] = QComboBox(parent=Selection)
            #self.verticalLayout_assignsegment.insertWidget(index + 1, new_entry["segment_box"])
            new_entry["layout"].addWidget(new_entry["segment_box"])
            new_entry["segment_box"].setFixedWidth(40)
            new_entry["segment_box"].currentIndexChanged.connect(lambda a, e=new_entry: get_residues_of_segment(e))

            new_entry["residue_box"] = QComboBox(parent=Selection)
            #self.verticalLayout_assignresidue.insertWidget(index + 1, new_entry["residue_box"])
            new_entry["layout"].addWidget(new_entry["residue_box"])
            new_entry["residue_box"].setFixedWidth(90)
            new_entry["residue_box"].currentIndexChanged.connect(lambda a, e=new_entry: get_atoms_of_residue(e))
            new_entry["residue_box"].setMaxVisibleItems(15)
            new_entry["residue_box"].setStyleSheet("combobox-popup: 0;")
            # this stylesheet is a little ugly inbetween, but the only possibility I see to limit the number of items

            new_entry["atom1_box"] = QComboBox(parent=Selection)
            #self.verticalLayout_assignatom1.insertWidget(index + 1, new_entry["atom1_box"])
            new_entry["layout"].addWidget(new_entry["atom1_box"])
            new_entry["atom1_box"].setFixedWidth(70)

            new_entry["atom2_box"] = QComboBox(parent=Selection)
            #self.verticalLayout_assignatom2.insertWidget(index + 1, new_entry["atom2_box"])
            new_entry["layout"].addWidget(new_entry["atom2_box"])
            new_entry["atom2_box"].setFixedWidth(70)



            new_entry["add_button"].setFixedWidth(20)
            new_entry["add_button"].setFixedHeight(new_entry["label"].size().height())
            #self.verticalLayout_copysignal.insertWidget(index + 1, new_entry["add_button"])
            new_entry["layout"].addWidget(new_entry["add_button"])

            new_entry["layout"].setContentsMargins(0,0,0,0)
            new_entry["layout"].setSpacing(0)   #todo the spacing is not applied properly
                                                #  find out why and fix -K


            self.verticalLayout.insertWidget(index+insert,new_entry["widget"])

            self.entries.insert(index+1, new_entry)
            if self.universe:
                get_segments_of_pdb(new_entry)

        # add QSpacerItems so the labels and buttons will always stick to the top of the vertical layouts
        #for layout in [self.verticalLayout_signals, self.verticalLayout_copysignal, self.verticalLayout_assignatom1,
        #        self.verticalLayout_assignatom2, self.verticalLayout_assignresidue, self.verticalLayout_assignsegment]:
        self.verticalLayout.addItem(QSpacerItem(20, 40, QSizePolicy.MinimumExpanding,QSizePolicy.Expanding))
        #functinality for dataset selection and pdb selection

        def load_signals_from_data():
            clear_entries()
            title = self.comboBox_dataset.currentText()
            for label in self.working_project[title][0].label:
                add_entry(label)
            load_pdb()

        self.comboBox_dataset.currentIndexChanged.connect(lambda a: self.update_pdb_combobox())
        self.comboBox_dataset.currentIndexChanged.connect(lambda a: load_signals_from_data())
        #self.comboBox_pdb.currentIndexChanged.connect(lambda a: load_pdb())

        self.pushButton_select.clicked.connect(lambda a: self.collect_selection())

    def collect_selection(self):
        """when clicking the select button, reading out all comboboxes for all signals and extract the atomnumbers of
        the corresponding bond
        if the dataset is already connected to a a MolSelect object, sel1, sel2 and label of this object will be over-
        written. If no dataset is provided, a new one will be created and attached"""
        #todo make a checkbox if it shall take original labels from nmr or create new ones form pdb
        dataset = self.working_project[self.comboBox_dataset.currentText()][0]
        sel1_list = []
        sel2_list = []
        signal_list= []
        for entry in self.entries[1:]: #first element is defaults
            signal_name = entry["label"].text()
            seg = entry["segment_box"].currentText()
            if not len(seg):
                continue
            seg = self.universe.segments[self.universe.segments.segids==seg][0]
            res =entry["residue_box"].currentText()
            if len(res)<2:
                continue
            resname, resid = res.split("-")
            resid = int(resid)
            res = seg.residues[seg.residues.resids==resid]
            sel1 = entry["atom1_box"].currentText()
            sel1 = res.atoms[res.atoms.names==sel1][0]
            sel2 = entry["atom2_box"].currentText()
            sel2 = res.atoms[res.atoms.names==sel2][0]
            sel1_list.append(sel1)
            sel2_list.append(sel2)
            signal_list.append(signal_name)

        sel1 = mda.AtomGroup(sel1_list)
        sel2 = mda.AtomGroup(sel2_list)
        if not hasattr(dataset,"select") or getattr(dataset,"select") is None:
            sys = pyDR.MolSys(self.comboBox_pdb.currentText())
            sel = pyDR.MolSelect(sys)
            dataset.select = sel
        else:
            sel = dataset.select

        sel.sel1 =sel1
        sel.sel2 = sel2
        sel.label = signal_list

    def update_pdb_combobox(self):
        # I am not sure, should we provide the possibility to have more than one pdb on a single dataset?
        # if yes: TODO loop over pdbs
        title = self.comboBox_dataset.currentText()
        self.comboBox_pdb.clear()
        if self.working_project[title][0].select.molsys.topo:
            self.comboBox_pdb.addItem(self.working_project[title][0].select.molsys.topo)

    def load_from_working_project(self):
        self.working_project = get_workingproject(self.parent)
        self.comboBox_dataset.addItem("")
        for title in self.working_project.titles:
            print(title)
            self.comboBox_dataset.addItem(title)

        self.update_pdb_combobox()