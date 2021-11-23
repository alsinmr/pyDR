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
    conn=list()
    closed=list()
    
    @classmethod
    def launch(cls):
        if True in cls.closed:
            ID=np.argwhere(cls.closed)[0,0]
        else:
            ID=len(cls.ports)
            cls.PIDs.append(None)
            cls.conn.append(None)
            cls.closed.append(True)
            cls.ports.append(cls.port0+ID) 
            

        with File(ID) as f:
            py_line(f,'import sys')
            py_line(f,'sys.path.append("{}")'.format(cls.path))
            py_line(f,'from RemoteCMXside import CMXReceiver as cmxr')
            py_line(f,'out=cmxr(session,{})'.format(cls.ports[ID]))
        
        cls.listener=Listener(('localhost',cls.ports[ID]),authkey=b'pyDIFRATE2chimeraX')
        
        cls.PIDs[ID]=os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),cls.full_path(ID))
        
        cls.tr=StartThread(cls.listener)
        cls.tr.start()
        t0=time()
        while time()-t0<10:
            if not(cls.tr.is_alive()):
                cls.conn[ID]=cls.tr.conn
                break     
        else:
            cls.conn[ID]=None
            print('Failed to establish connection with ChimeraX')
        

        
        cls.closed[ID]=False     
        print('update1')
        return ID
    
    @classmethod
    def full_path(cls,ID):
        return get_path('chimera_script{0:02d}.py'.format(ID))     #Location to write out chimera script 
    
    @classmethod
    def command_line(cls,ID,string):
        cls.conn[ID].send(('command_line',string))

        
 
class StartThread(Thread):
    def __init__(self,listener):
        super().__init__()
        self.listener=listener
    def run(self):
        self.conn=self.listener.accept()
        
class File():
    def __init__(self,ID):
        self.filename=CMXRemote.full_path(ID)
    def __enter__(self):
        self.file=open(self.filename,'w')
        return self.file
    def __exit__(self,exception_type, exception_value, traceback):
        self.file.close()


        