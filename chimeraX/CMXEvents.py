#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:49:56 2021

@author: albertsmith
"""

class Hover():
  def __init__(self,cmx):
     self.cmx = cmx
     self.session=cmx.session
     self.cursor = self.session.ui.mouse_modes.graphics_window.cursor()
     for win in self.session.ui.allWindows():
       if win.objectName() == "MainWindowClassWindow":
         self.win1 = win
         break
     self.win2 = self.session.ui.allWindows()[-1]
     self.win_size=self.session.view.window_size
     self.cont=True

  def get_mouse_pos(self):
      mx = self.cursor.pos().x() - self.win1.position().x() - self.win2.position().x()
      my = self.cursor.pos().y() - self.win1.position().y() - self.win2.position().y()
      return mx,my

  def __call__(self):
      mx, my = self.get_mouse_pos()
      ob = self.session.main_view.picked_object(mx, my)
      if not hasattr(self,"hover"):
        if hasattr(ob,"atom"):
          self.hover = ob
          ob.atom.radius+=1
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


class Detectors(Hover):
    def __init__(self,cmx):
        Hover.__init__(self,cmx)
        self.model = self.session.models[0]
        self.open_detector()
        from chimerax.label.label2d import LabelModel
        from time import sleep
        self.labels = []  #todo make this to a dictionary where the functions of the buttons are stored, too
        sleep(1) #todo hate this but is needed because else it could be the labels are not fully initialized
        try:
            for mdl in self.session.models:
                if isinstance(mdl,LabelModel):
                    self.labels.append(mdl)
            print("got the label(s)")
        except:
            print("no label here")

    def __call__(self):
        mx,my = self.get_mouse_pos()
        mx= mx/self.win2.size().width()*2-1
        my=(my/self.win2.size().height()*2-1)*-1
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
        def get_index(res,atom):
            for i,a in enumerate(res.atoms):
                if a.name==atom:
                    return i
        import numpy as np
        from matplotlib.pyplot import get_cmap
        cmap = get_cmap("tab10")
        res_nums = []
        atom_names = []
        det_responses = []
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
            atom_nums.append(last+get_index(res,atom_names[i]))#)targets[res.name.lower()])
            last += len(res.atoms)
        atom_nums = np.array(atom_nums)
        def set_radius(atoms,R, color):
            #todo check if R has a value and is greater than 0, otherwise you can get problems
            R/=R.min()  # I dont understand why, but atoms.radii = R/R.min() will not work
            #TODO decide how to display which response
            atoms.radii = R
            atoms.colors = color

        self.commands = []
        from chimerax.label.label2d import label_create

        for i in range(len(responses)):
            label_create(self.session,"testlabel", text="testmeifucan", xpos=.2,ypos=.2)
            self.cmx.send_command("2dlabels text det{} size 25 x 0.9 y {}".format(i,str(0.9-i*0.075)))
            self.commands.append(lambda atoms = self.model.residues[res_nums-1].atoms[atom_nums],
                                        R = det_responses.T[i],
                                        color=(np.array(cmap(i))*255).astype(int)
                                         :set_radius(atoms,R,color))