#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:02:59 2021

@author: albertsmith
"""


import numpy as np
import os
import sys
from multiprocessing.connection import Listener,Client
from pyDR.chimeraX.chimeraX_funs import get_path,py_line,WrCC,chimera_path,run_command
from threading import Thread
from time import time,sleep

#%% Functions to run in pyDR
class CMXRemote():
    ports=list()
    PIDs=list()
    
    @classmethod
    @classmethod
    def launch(cls):
        if True in cls.closed:
            ID=np.argwhere(cls.closed)[0,0]
        else:
            ID=len(cls.ports)
            cls.PIDs.append(None)
            cls.closed.append(True)
            cls.ports.append(cls.port0+ID) 
            

        with File(ID) as f:
            py_line(f,run_command())
            WrCC(f,'remotecontrol rest start port {0}'.format(cls.ports[ID]))
        
        cls.PIDs[ID]=os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),cls.full_path(ID))
        cls.closed[ID]=False
        
        return ID
    
        
class File():
    def __init__(self,ID):
        self.filename=CMXRemote.full_path(ID)
    def __enter__(self):
        self.file=open(self.filename,'w')
        return self.file
    def __exit__(self,exception_type, exception_value, traceback):
        self.file.close()

#%% Functions to run in chimeraX
class StartThread(Thread):
    def __init__(self,listener):
        super().__init__()
        self.listener=listener
    def run(self):
        self.conn=self.listener.accept()    

class CMXReceiver():
    def __init__(self,port):
        self.port=port
        self.listener=Listener(('localhost',port),authkey=b"pyDIFRATE2chimeraX")
        tr=StartThread(self.listener)
        t0=time()
        while time()-t0<10:
            tr.start()
            if not(tr.isAlive()):
                self.conn=tr.conn
                break
        else:
            self.listener.close()
#            from chimerax.core.commands import run
#            run('2dlabel create label0 text "Failed to connect to pyDIFRATE" xpos .2 ypos .2')
#            sleep(2)
#            run('exit')
            pass
        

        