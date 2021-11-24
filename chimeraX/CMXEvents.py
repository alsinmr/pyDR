#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:49:56 2021

@author: albertsmith
"""

class Hover():
  def __init__(self,cmx):
     session=cmx.session
     super().__init__()
     self.session = session
     self.cursor = self.session.ui.mouse_modes.graphics_window.cursor()
     for win in self.session.ui.allWindows():
       print(win.objectName())
       if win.objectName() == "MainWindowClassWindow":
         self.win1 = win
         break
     self.win2 = self.session.ui.allWindows()[-1]
     self.win_size=self.session.view.window_size
     self.cont=True
     
  def __call__(self):
      mx = self.cursor.pos().x()-self.win1.position().x()-self.win2.position().x()
      my = self.cursor.pos().y()-self.win1.position().y()-self.win2.position().y()
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
          