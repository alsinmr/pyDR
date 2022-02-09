import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt,QRect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
import numpy as np
import pyDIFRATE as DR
from pyDIFRATE.data.load_nmr import load_NMR
import MDAnalysis as MDA


class Home(QWidget):
    name = "Home"
    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        layout = QHBoxLayout()
        label = QLabel("Das ist pyDIFRATE",parent=self)
        layout.addWidget(label)
        self.setLayout(layout)


class NMR_available(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        print(parent.name)
        layout = QVBoxLayout()
        label = QLabel("Das ist pyDIFRATE, im NMR Widget",parent=self)
        layout.addWidget(label)
        for fold in listdir("nmr"):
            b = QLabel(fold)
            layout.addWidget(b)

            for f in listdir(join("nmr",fold)):
                b = QPushButton(f)
                b.setFixedWidth(200) #todo unfix this, bzw make it dynamic
                b.clicked.connect(lambda s, t=fold, fi=f: parent.detect(join(t,fi)))
                layout.addWidget(b)
        self.setLayout(layout)


class NMR_detectors(QWidget):
    def __init__(self, parent: QWidget, filename:str):
        super().__init__(parent=parent)
        layout = QVBoxLayout()
        het = load_NMR(join("nmr",filename))
        n_dets = 4
        het.detect.r_auto3(int(n_dets), inclS2=True, Normalization='MP')
        fit = het.fit()
        #todo rearrange the plot functionality, gives some problems with the GUI on linux -K
        self.plot = FigureCanvasQTAgg()
        fit.plot_rho(fig=self.plot.figure, style="bar", errorbars=True)

        self.plot.figure.suptitle(filename)
        toolbar = NavigationToolbar2QT(self.plot,self)
        layout.addWidget(self.plot)
        layout.addWidget(toolbar)
        b = QPushButton("new plot")
        b.clicked.connect(self.other_plot)
        layout.addWidget(b)

        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)

    def other_plot(self):
        self.ax.cla()




class NMR(QWidget):
    name = "NMR"


    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        self.layout = QHBoxLayout(self)
        left = NMR_available(self)
        self.layout.addWidget(left)
        self.setLayout(self.layout)

    def detect(self, filename):
        if hasattr(self,"detectorframe"):
            self.detectorframe.close()
        self.detectorframe = NMR_detectors(self, filename)
        self.layout.addWidget(self.detectorframe)


class MDDetail(QWidget):
    def __init__(self, parent: QWidget, mdpath):
        super().__init__(parent=parent)
        mylayout = QVBoxLayout(self)

        uni = MDA.Universe(join("xtcs/",mdpath))

        for info in [mdpath,
                     "length:\t {}".format(len(uni.trajectory)),
                     "dt:\t{}".format(uni.trajectory.dt)]:
            mylayout.addWidget(QLabel(info))

        mylayout.setAlignment(Qt.AlignmentFlag.AlignTop)


class MD(QWidget):
    name = "MD"
    def __init__(self, parent: QWidget):
        super().__init__(parent=parent)
        self.mylayout = QHBoxLayout(self)

        MD_List = QWidget()
        MD_List_layout = QVBoxLayout(MD_List)
        for xtc in listdir("xtcs"):
            if xtc.endswith(".xtc"):
                b = QPushButton(xtc[:-4])
                b.setFixedWidth(100)
                b.clicked.connect(lambda s,t=xtc:self.set_MD(t))
                MD_List_layout.addWidget(b)

        MD_List.setLayout(MD_List_layout)
        self.mylayout.addWidget(MD_List)

    def set_MD(self, mdpath):
        print(mdpath)
        if hasattr(self,"MDdetails"):
            self.MDdetails.close()

        self.MDdetails = MDDetail(self,mdpath)
        self.mylayout.addWidget(self.MDdetails)
        self.mylayout.setAlignment(Qt.AlignmentFlag.AlignLeft)



class Header(QWidget):
    def __init__(self,parent:QWidget):
        super().__init__(parent=parent)

        layout = QHBoxLayout()
        for body in [Home,NMR,MD]:
            button = QPushButton(body.name, parent=self)
            button.clicked.connect(lambda s, b=body:parent.set_body(b))  #important, the s argument has to be there, no idea why
            layout.addWidget(button)#, alignment=Qt.AlignTop)

        self.setLayout(layout)
        self.setFixedWidth(350)
        self.setFixedSize(350,40)



class pyDR_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(QRect(100,100,1280,720))
        self.layout = QVBoxLayout()

        H = Header(self)  #creating the Header, this one should stay in program whole time
        self.layout.addWidget(H)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)

    def set_body(self, body):
        if hasattr(self,"body"):
            self.body.close()
        self.body = body(self)
        self.layout.addWidget(self.body)

    '''
    def handleButton(self):
        modifiers = QApplication.keyboardModifiers()

        if modifiers == Qt.ShiftModifier:
            print('Shift+Click')
        elif modifiers == Qt.ControlModifier:
            print('Ctrl+Click')
        elif modifiers == (Qt.ShiftModifier | Qt.ControlModifier):
            print('Ctrl+Shift+Click')
        else:
            print('Mouse Click')
    '''

if __name__ == '__main__':
    app = QApplication(sys.argv)

    demo = pyDR_GUI()
    demo.show()

    sys.exit(app.exec_())
