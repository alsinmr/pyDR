#!/usr/bin/chimerax
from chimerax.core.commands import run as rc
import os
import numpy as np
from chimerax.geometry import Place
from threading import Thread
from time import sleep





mdl=session.open_command.open_data("/Volumes/My Book/HETs/HETs_3chain.pdb")[0]
session.models.add(mdl)
#session.view.camera.set_position(Place(np.load('pos.npy')))
atoms=session.models[0].atoms
rc(session,"ribbon")
rc(session,"ui tool show Shell")
rc(session,"display")
rc(session,"style ball")
rc(session,"set bgColor white")
rc(session,"graphics silhouettes true")
rc(session,"lighting soft")


class MyThread(Thread):
  def __init__(self,session):
     super().__init__()
     self.session = session
     self.cursor = session.ui.mouse_modes.graphics_window.cursor()
     for win in session.ui.allWindows():
       print(win.objectName())
       if win.objectName() == "MainWindowClassWindow":
         self.win1 = win
         break
     self.win2 = session.ui.allWindows()[-1]
     self.win_size=session.view.window_size
     
  def is_session_alive(self):
     return 1
     #TODO

     
  def run(self):
    while self.is_session_alive():
      sleep(0.3)
      mx = self.cursor.pos().x()-self.win1.position().x()-self.win2.position().x()
      my = self.cursor.pos().y()-self.win1.position().y()-self.win2.position().y()
      ob = self.session.main_view.picked_object(mx, my)
      if not hasattr(self,"hover"):
        if hasattr(ob,"atom"):
          rc(session,"set bgColor black")
          self.hover = ob
          ob.atom.radius+=1
      elif hasattr(ob,"atom"):
        if self.hover.atom.name != ob.atom.name:
          print(ob.atom.name)
          if self.hover:
            self.hover.atom.radius -= 1
          ob.atom.radius += 1
          self.hover = ob
      

T = MyThread(session)
T.start()
