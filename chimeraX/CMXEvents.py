#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:49:56 2021

@author: albertsmith
"""
from PyQt5.QtWidgets import QMouseEventTransition, QApplication
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

class Hover():
    def __init__(self, cmx):
        self.cmx = cmx
        self.session= cmx.session
        self.cursor = self.session.ui.mouse_modes.graphics_window.cursor()
        for win in self.session.ui.allWindows():  #todo set session.ui.main_window
            if win.objectName() == "MainWindowClassWindow":
                self.win1 = win
                break
        self.win2 = self.session.ui.allWindows()[-1]
        self.win_size=self.session.view.window_size
        self.cont = True

    def get_mouse_pos(self):
        return self.cursor.pos().x() - self.win1.position().x() - self.win2.position().x(),\
               self.cursor.pos().y() - self.win1.position().y() - self.win2.position().y()

    def __call__(self):
        mx, my = self.get_mouse_pos()
        ob = self.session.main_view.picked_object(mx, my)
        if not hasattr(self,"hover"):
            if hasattr(ob,"atom"):
                self.hover = ob
                ob.atom.radius += 1
        elif hasattr(ob,"atom"):
            if self.hover.atom.name != ob.atom.name:
                print(ob.atom.name)
                if self.hover:
                    self.hover.atom.radius -= 1
                ob.atom.radius += 1
                self.hover = ob

    def cleanup(self):
        if hasattr(self,"hover"):
            self.hover.atom.radius -= 1

class Click(Hover):
    def __init__(self,cmx):
        super().__init__(cmx)
        self.mod =QApplication.mouseButtons

    def __call__(self, *args, **kwargs):
        mod = self.mod()
        mx, my = self.get_mouse_pos()
        ob = self.session.main_view.picked_object(mx, my)
        if hasattr(ob,"atom"):
            if mod == Qt.MouseButton.LeftButton:
                ob.atom.radius += 1
            elif mod== Qt.MouseButton.RightButton:
                if ob.atom.radius >2:
                    ob.atom.radius -= 2

        '''
        : Qt.MouseButton
        Out[42]: PyQt5.QtCore.Qt.MouseButton
        
        Qt.MouseButton()
        Out[43]: 0
        
        Qt.MouseButton()
        Out[44]: 0'''

class Detectors(Hover):
    def __init__(self,cmx):
        Hover.__init__(self,cmx)
        self.model = self.session.models[0]
        self.open_detector()

    def __call__(self):
        mx, my = self.get_mouse_pos()
        mx = mx/self.win2.size().width()*2-1
        my = -(my/self.win2.size().height()*2-1)
        for i,label in enumerate(self.labels):
            geo = label.geometry_bounds()
            if geo.contains_point([mx,my,0]):
                if not label.selected:
                    label.selected=True
                    self.commands[i]()
            else:
                label.selected=False
            label.label.update_drawing()

    def open_detector(self):
        def get_index(residue, atom):
            for j, a in enumerate(residue.atoms):
                if a.name == atom:
                    return j
        import numpy as np
        from matplotlib.pyplot import get_cmap
        cmap = lambda ind: (np.array(get_cmap("tab10")(i))*255).astype(int)#get_cmap("tab10")
        res_nums = []
        atom_names = []
        det_responses = []
        print("open detector file")
        with open("det2.txt") as f:
            for line in f:
                l = line.strip()
                l = l[:-1]
                res,  responses = l.split(":")
                res_num,atom_name = res.split("-")
                responses = responses.split(";")
                res_nums.append(int(res_num))
                atom_names.append(atom_name)
                det_responses.append(np.array(responses).astype(float))

        res_nums = np.array(res_nums)
        det_responses = np.array(det_responses)
        self.model.residues[res_nums-1].atoms.displays=True
        last = 0
        atom_nums = []  #TODO i think this could almost get oneline, but low priority -K
        for i,res in enumerate(self.model.residues[res_nums-1]):
            atom_nums.append(last+get_index(res,atom_names[i]))
            last += len(res.atoms)
        atom_nums = np.array(atom_nums)
        def set_radius(atoms,R, color):
            #todo check if R has a value and is greater than 0, otherwise you can get problems
            R/=R.max()
            R*=5# I dont understand why, but atoms.radii = R/R.min() will not work
            #TODO decide how to display which response
            atoms.radii = R
            atoms.colors = color

        self.labels = []
        self.commands = []
        from chimerax.label.label2d import label_create
        #todo one can set the label text to the correlation time
        for i in range(len(responses)):
            label = label_create(self.session,"det{}".format(i), text="ρ{}".format(i)
                         , xpos=.95,ypos=.9-i*.075,
                         color=cmap(i), outline=1)
            self.labels.append(self.session.models[-1])
            self.commands.append(lambda atoms = self.model.residues[res_nums-1].atoms[atom_nums],
                                        R = det_responses.T[i],
                                        color=cmap(i)
                                         :set_radius(atoms, R, color))
