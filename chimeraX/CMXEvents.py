#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:49:56 2021

@author: albertsmith
"""

class Hover():
  def __init__(self,cmx):
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
      self.hover.atom.radius -= 1


class Hover_over_2DLabel(Hover):
    def __init__(self,cmx):
        Hover.__init__(self, cmx)
        self.cmx = cmx
        from chimerax.model_panel.tool import ModelPanel
        from chimerax.label.label2d import LabelModel
        from time import sleep
        self.open_detector()

        #getting Panel where Models are stored in chimera (down right side)
        for tool in self.session.tools:
            if isinstance(tool,ModelPanel):
                break
        self.labels = []  #todo make this to a dictionary where the functions of the buttons are stored, too
        sleep(1) #todo hate this but is needed because else it could be the labels are not fully initialized
        try:
            for mdl in tool.models:
                if isinstance(mdl,LabelModel):
                    self.labels.append(mdl)
            print("got the label(s)")
        except:
            print("no label here")

    def open_detector(self):
        import numpy as np
        print("open det")
        res_nums = []
        res_names = []
        det_responses = []
        with open("det.txt") as f:
            for line in f:
                l = line.strip()
                l = l[:-1]
                res,  responses = l.split(":")
                res_num,res = res.split("-")
                responses = responses.split(";")
                res_nums.append(int(res_num))
                res_names.append(res.lower())
                det_responses.append(np.array(responses).astype(float))

        res_nums = np.array(res_nums)
        det_responses = np.array(det_responses)
        cmd = "show :"
        for res in res_nums:
            cmd += str(res)
            cmd += ","
        cmd = cmd[:-1]
        self.cmx.send_command(cmd)
        atoms = []

        def set_all(det_num, atoms,resnames,R):
            targets = {"ile": "CD",
                       "ala": "CB",
                       "val": "CG2",
                       "leu": "CD2"}
            for j in range(len(atoms)):
                print("set",atoms[j],resnames[j], R[j])
                targ = targets[resnames[j]]
                self.cmx.send_command("setattr :{}@{} atom radius {}".format(atoms[j],targ, R[j]/min(R)))
                self.cmx.send_command("color :{}@{} {} target a".format(atoms[j],targ, "blue" if det_num==0 else "orange"
                                                                        if det_num==1 else "green" if det_num==2 else "red"))
        self.commands = []
        for i in range(len(responses)):
            #todo label naming should contain the correlation time of the detector, so this information might be needed
            #todo in the detector file -K
            self.cmx.send_command("2dlabels text det{} size 25 x 0.9 y {}".format(i,str(0.9-i*0.1)))
            self.commands.append(lambda det=i,
                                        atoms = res_nums,
                                        resnames=res_names,
                                        R=det_responses[:,i]
                                        :
                                        set_all(det,atoms,resnames,R))


    def __call__(self):
        #here I would say one should iterate over the existing labels, get the geo events
        #calculating the mouse positions might be fine so far
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

    def cleanup(self):
        pass
