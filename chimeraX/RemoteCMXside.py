#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:25:09 2021

@author: albertsmith
"""


from multiprocessing.connection import Client
from chimerax.core.commands import run
import chimerax
from threading import Thread
from time import sleep


class StartThread(Thread):
    def __init__(self,listener):
        super().__init__()
        self.listener=listener
    def run(self):
        self.conn=self.listener.accept()    

class ListenExec(Thread):
    def __init__(self,cmx):
        super().__init__()
        self.cmx=cmx
        self.args=None
        
    def run(self):
        try:
            self.args=self.cmx.client.recv()
            self.cmx.wait4command()     #If we successfully receive, start the process again   
        except:     #If the connection closes, we'll close this side as well
            self.cmx.client.close()
            
class Hover(Thread):
  def __init__(self,cmx):
     super().__init__()
     self.cmx=cmx
     self.session = cmx.session
     self.cursor = session.ui.mouse_modes.graphics_window.cursor()
     for win in session.ui.allWindows():
       print(win.objectName())
       if win.objectName() == "MainWindowClassWindow":
         self.win1 = win
         break
     self.win2 = session.ui.allWindows()[-1]
     self.win_size=session.view.window_size
     self.cont=True
     
  def is_session_alive(self):
     return self.cont
     #TODO

     
  def run(self):
    while self.is_session_alive():
      sleep(0.3)
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
    else:
      self.hover.atom.radius -= 1
          

class CMXReceiver():
    def __init__(self,session,port):
        self.session=session
        self.port=port
        self.tr=None
        self.is_closing=False
        
        try:
            self.client=Client(('localhost',port),authkey=b"pyDIFRATE2chimeraX")
        except:
            if hasattr(self,'client'):self.client.close()
            run(self.session,'exit')
            return

        self.wait4command()
        
    def wait4command(self):
        if self.tr and self.tr.args:
            fun,*args=self.tr.args
            if hasattr(self,fun):
                count=0
                try:
                    count+=1
                    getattr(self,fun)(*args)
                    count+=2
                except:
                    print(count)
                    print('Execution of {} failed'.format(fun))    
        self.tr=ListenExec(self)
#        self.tr.isDaemon=True
        self.tr.start()
    
    def command_line(self,string):
        run(self.session,string)
        
        
    def phone_home(self):
        self.client.send('still here')
        
    def Exit(self):
        try:
            self.client.close()
        except:
            pass
        run(self.session,'exit')
        
    def get_sel(self):
        sel=list()
        for k,mdl in enumerate(self.session.models):
            if mdl.selected:
                sel.append({"model":k})
                b0,b1=mdl.bonds[mdl.bonds.selected].atoms
                sel[-1]['b0']=b0.coord_indices
                sel[-1]['b1']=b1.coord_indices
                sel[-1]['a']=mdl.atoms[mdl.atoms.selected].coord_indices
        self.client.send(sel)
        
    def hover_on(self):
        self.hover=Hover(self)
        self.hover.start()
    
    def hover_off(self):
        if hasattr(self,'hover'):
            self.hover.cont=False
            
        
