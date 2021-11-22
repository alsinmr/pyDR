#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:25:09 2021

@author: albertsmith
"""


from multiprocessing.connection import Listener
from chimerax.core.commands import run
from time import time,sleep
from threading import Thread



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
        
    def run(self):
        while True:
            try:
                fun,*args=self.cmx.conn.recv()
                if hasattr(self.cmx,fun):
                    try:
                        getattr(self.cmx,fun)(*args)
                    except:
                        pass
            except:
                pass

class CMXReceiver():
    def __init__(self,session,port):
        self.session=session
        self.port=port
        self.listener=Listener(('localhost',port),authkey=b"pyDIFRATE2chimeraX")
        tr=StartThread(self.listener)
        t0=time()
        tr.start()
        while time()-t0<10:  
            if not(tr.isAlive()):
                self.conn=tr.conn
                break
            sleep(.05)
        else:
            self.listener.close()
            run(self.session,'2dlabel create label0 text "Failed to connect to pyDIFRATE" xpos .2 ypos .2')
            sleep(2)
            run(self.session,'exit')
            return
        
        self.tr=ListenExec(self)
        self.tr.isDaemon=True
        self.tr.start()
    
    def command_line(self,string):
        run(self.session,string)
        
