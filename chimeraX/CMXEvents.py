#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:49:56 2021

@author: albertsmith
"""

from pkg_resources import working_set
if 'pyqt5-commercial' in [pkg.key for pkg in working_set]:
    # from PyQt5.QtWidgets import QMouseEventTransition, QApplication
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    # from PyQt5 import QtGui
else:
    # from PyQt6.QtWidgets import QMouseEventTransition, QApplication
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    # from PyQt6 import QtGui


import numpy as np
from matplotlib.pyplot import get_cmap
from RemoteChimeraFuns import set_color_radius,set_color_radius_CC,DetFader
from chimerax.label.label2d import label_create

class DetectorFader():
    def __init__(self,cmx,*args):
        self.cmx=cmx
        self.session=cmx.session
        self.fader=DetFader(self.session.models[-1],*args)
        
    def __call__(self):
        self.fader.set_color_radius()
        
class TimescaleIndicator():
    def __init__(self,cmx,tau):
        self.cmx=cmx
        self.session=cmx.session
        self.model=self.session.models[-1]
        self.tau=tau
        self.i=-1
        self.label=label_create(self.session,'timescale',  #Create a 2d label
                                text='1 s: {:.0f} ps'.format(self.tau[1]*1e3),
                                xpos=0.02,ypos=.05,size=50)   
        
        
    def __call__(self):
        i=self.model.active_coordset_id-1
        if i!=self.i:
            tau=self.tau[i]*1e3
            if i==0:
                text='1 s: {:.0f} ps'.format(self.tau[1]*1e3)
            elif tau<1e3:
                text='1 s: {:.0f} ps'.format(tau)
            elif tau<1e6:
                text='1 s: {:.0f} ns'.format(tau/1e3)
            elif tau<1e9:
                text='1 s: {:.0f} μs'.format(tau/1e6)
            else:
                text='1 s: {:.0f} ms'.format(tau/1e9)
            self.label.text=text
            self.label.update_drawing()
            self.i=i
        

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
        self.mod = QApplication.mouseButtons

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
    def __init__(self, cmx, *args):
        # Hover.__init__(self,cmx)
        super().__init__(cmx)
        for k in range(len(self.session.models),0,-1):
            if hasattr(self.session.models[k-1],'atoms'):
                self.model=self.session.models[k-1]
                break
        # self.model = self.session.models[0]   #todo be careful here, the index is not always the right model!!!!
        ids = args[0].get("ids")
        R = args[0].get("R")
        rho_index=args[0].get('rho_index')
        self.color=args[0]['color'] if (R.shape[1]==1 and 'color' in args[0]) else None
        self.open_detector(ids,R,rho_index)

    def __call__(self):
        mx, my = self.get_mouse_pos()
        mx = mx/self.win2.size().width()*2-1
        my = -(my/self.win2.size().height()*2-1)
        for i,label in enumerate(self.labels):
            geo = label.geometry_bounds()
            if geo.contains_point([mx,my,0]):
                if not label.selected:
                    label.selected=True
                    self.commands[i]()  #HERE IS WHERE THE COMMAND GETS CALLED!!!!!!!!
            else:
                label.selected=False
            label.label.update_drawing()

    def open_detector(self, ids, R,rho_index):
        cmap =lambda ind: (np.array(get_cmap("tab10")(ind%10)) * 255).astype(int) \
            if self.color is None else self.color  # get_cmap("tab10")
        # def set_color_radius(atoms,R, color,ids):
        #     #todo check if R has a value and is greater than 0, otherwise you can get problems
        #     # R/=R.max()
        #     # R*=5# I dont understand why, but atoms.radii = R/R.min() will not work
        #     #TODO decide how to display which response
        #     colors=color_calc(R,colors=[[210,180,140,255],color])
        #     for id0,R0,color in zip(ids,R,colors):
        #         atoms[id0].radii = 0.8+4*R0
        #         atoms[id0].colors = color
        
        
        self.labels = []
        self.commands = []
        
        #todo one can set the label text to the correlation time
        if len(rho_index)==1:
            "If only one rho_index, then just set the radii initially (no label)"
            set_color_radius(self.model.atoms,R.T[rho_index[0]],cmap(rho_index[0]),ids)
        else:
            for i in range(R.T[rho_index].shape[0]):
                # label = label_create(self.session,"det{}".format(i), text="ρ{}".format(rho_index[i])
                label = label_create(self.session,"det{}".format(np.random.randint(1000000)), text="ρ{}".format(rho_index[i])
                             , xpos=.95,ypos=.9-i*.075,
                             color=cmap(rho_index[i]),bg_color=np.array([255,255,255,0],dtype=int))
                self.labels.append(self.session.models[-1])
                #TODO It seems that a pdb with multiple structures not loaded as a coordset might not have model.atoms ??
                self.commands.append(lambda atoms = self.model.atoms,#residues[res_nums-1].atoms[atom_nums],
                                            R = R.T[rho_index[i]],
                                            color=cmap(rho_index[i]),ids=ids
                                             :set_color_radius(atoms, R, color,ids))
        return
        '''
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
                                         :set_radius(atoms, R, color))'''


class DetCC(Detectors):
    def open_detector(self, ids, R,rho_index):
        
        cmap =lambda ind: (np.array(get_cmap("tab10")(ind%10)) * 255).astype(int)  # get_cmap("tab10")
        
        self.labels = []
        self.commands = []
        
        #todo one can set the label text to the correlation time
        if len(rho_index)==1:
            "If only one rho_index, then just set the radii initially (no label)"
            set_color_radius_CC(self.model.atoms,R.T[rho_index[0]],cmap(rho_index[0]),ids)
        else:
            for i in range(R.T[rho_index].shape[0]):
                # label = label_create(self.session,"det{}".format(i), text="ρ{}".format(rho_index[i])
                label = label_create(self.session,"det{}".format(np.random.randint(1000000)), text="ρ{}".format(rho_index[i])
                             , xpos=.95,ypos=.9-i*.075,
                             color=cmap(rho_index[i]),bg_color=np.array([255,255,255,0],dtype=int))
                self.labels.append(self.session.models[-1])
                #TODO It seems that a pdb with multiple structures not loaded as a coordset might not have model.atoms ??
                self.commands.append(lambda atoms = self.model.atoms,#residues[res_nums-1].atoms[atom_nums],
                                            R = R.T[rho_index[i]],
                                            color=cmap(rho_index[i]),ids=ids
                                             :set_color_radius_CC(atoms, R, color,ids))
        return

def color_calc(x,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]]):
    """
    Calculates color values for a list of values in x (x ranges from 0 to 1).
    
    These values are linear combinations of reference values provided in colors.
    We provide a list of N colors, and a list of N x0 values (if x0 is not provided,
    it is set to x0=np.linspace(0,1,N). If x is between the 0th and 1st values
    of x0, then the color is somewhere in between the first and second color 
    provided. Default colors are blue at x=0, tan at x=0.5, and red at x=1.
    
    color_calc(x,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]])
    """
    
    colors=np.array(colors,dtype='uint8')
    N=len(colors)
    if x0 is None:x0=np.linspace(0,1,N)
    x=np.array(x)
    if x.min()<x0.min():
        print('Warning: x values less than min(x0) are set to min(x0)')
        x[x<x0.min()]=x0.min()
    if x.max()>x0.max():
        print('Warning: x values greater than max(x0) are set to max(x0)')
        x[x>x0.max()]=x0.max()

    i=np.digitize(x,x0)
    i[i==len(x0)]=len(x0)-1
    clr=(((x-x0[i-1])*colors[i].T+(x0[i]-x)*colors[i-1].T)/(x0[i]-x0[i-1])).T
    return clr.astype('uint8')