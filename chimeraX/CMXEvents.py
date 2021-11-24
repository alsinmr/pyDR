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
        from chimerax.model_panel.tool import ModelPanel
        from chimerax.label.label2d import LabelModel

        #todo get_button_information from cmx
        #todo create_buttons

        #getting Panel where Models are stored in chimera (down right side)
        for tool in self.session.tools:
            if isinstance(tool,ModelPanel):
                break
        self.labels = []  #todo make this to a dictionary where the functions of the buttons are stored, too
        try:
            for mdl in tool.models:
                if isinstance(mdl,LabelModel):
                    self.labels.append(mdl)
            print("got the label(s)")
        except:
            print("no label here")

    def __call__(self):
        #here I would say one should iterate over the existing labels, get the geo events
        #calculating the mouse positions might be fine so far
        mx,my = self.get_mouse_pos()
        mx= mx/self.win2.size().width()*2-1
        my=(my/self.win2.size().height()*2-1)*-1
        for label in self.labels:
            geo = label.geometry_bounds()
            if geo.contains_point([mx,my,0]):
                label.label.text="True"
            else:
                label.label.text="False"
            label.label.update_drawing()

    def cleanup(self):
        pass
