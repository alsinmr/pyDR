#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:02:59 2021

@author: albertsmith
"""


import numpy as np
import os
from multiprocessing.connection import Listener,Client
from pyDR.chimeraX.chimeraX_funs import get_path,py_line,WrCC,chimera_path,run_command
from threading import Thread
from time import time,sleep

#%% Functions to run in pyDR
class CMXRemote():
    ports=list()
    PIDs=list()
    path=os.path.dirname(os.path.abspath(__file__))
    port0=7000
    clients=list()
    closed=list()
    
    @classmethod
    def launch(cls):
        if True in cls.closed:
            ID=np.argwhere(cls.closed)[0,0]
        else:
            ID=len(cls.ports)
            cls.PIDs.append(None)
            cls.clients.append(None)
            cls.closed.append(True)
            cls.ports.append(cls.port0+ID) 
            

        with File(ID) as f:
            py_line(f,'import sys')
            py_line(f,'sys.path.append("{}")'.format(cls.path))
            py_line(f,'from RemoteCMXside import CMXReceiver as cmxr')
            py_line(f,'cmxr(session,{})'.format(cls.ports[ID]))
        
        cls.PIDs[ID]=os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),cls.full_path(ID))
        
        tr=StartThread(cls.ports[ID])
        tr.start()
        while True:
            if not(tr.isAlive()):
                cls.clients[ID]=tr.client
                break
            
        
        cls.closed[ID]=False
        
        return ID
    
    @classmethod
    def full_path(cls,ID):
        return get_path('chimera_script{0:02d}.py'.format(ID))     #Location to write out chimera script 
    
    @classmethod
    def command_line(cls,ID,string):
        cls.clients[ID].send(('command_line',string))
        
 
class StartThread(Thread):
    def __init__(self,port):
        super().__init__()
        self.port=port
    def run(self):
        t0=time()
        while time()-t0<10:
            try:
                self.client=Client(('localhost',self.port),authkey=b'pyDIFRATE2chimeraX')
                break
            except:
                sleep(.1)
        else:
            self.client=None
        
class File():
    def __init__(self,ID):
        self.filename=CMXRemote.full_path(ID)
    def __enter__(self):
        self.file=open(self.filename,'w')
        return self.file
    def __exit__(self,exception_type, exception_value, traceback):
        self.file.close()


        