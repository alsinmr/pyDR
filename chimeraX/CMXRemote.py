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
    rc_port0=60958
    conn=list()
    closed=list()
    
    @classmethod
    def launch(cls,new_instance=True):
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
            py_line(f,run_command())
            WrCC(f,'remotecontrol rest start port {0}'.format(ID+cls.rc_port0))
            py_line(f,'sys.path.append("{}")'.format(cls.path))
            py_line(f,'from RemoteCMXside import CMXReceiver as CMXR')
            py_line(f,'cmxr=CMXR(session,{})'.format(cls.ports[ID]))
            WrCC(f,'ui mousemode right select')
        
        cls.listener=Listener(('localhost',cls.ports[ID]),authkey=b'pyDIFRATE2chimeraX')
        
        if new_instance:
            cls.PIDs[ID]=os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),cls.full_path(ID))
        else:
            cls.PIDs[ID]=0
        
        cls.tr=StartThread(cls.listener)
        cls.tr.start()
        t0=time()
        while time()-t0<10:     #Timeout connection after 10s
            if not(cls.tr.is_alive()):
                cls.conn[ID]=cls.tr.conn 
                cls.listener.close()    #Once the connection is made, we close the listener
                break     #Connection successfully made if we reach this point
        else:
            cls.conn[ID]=None
            cls.listener.close()
            print('Failed to establish connection with ChimeraX')
            cls.kill(ID)
        

        
        cls.closed[ID]=False     
        print('update1')
        return ID
    
    @classmethod
    def full_path(cls,ID):
        return get_path('chimera_script{0:02d}.py'.format(ID))     #Location to write out chimera script 
    
    @classmethod
    def py_command(cls,ID,string):
        with File(ID) as f:
            py_line(f,string)
        cls.send_file(ID)
    
    
    @classmethod
    def command_line(cls,ID,string):
        with File(ID) as f:
            py_line(f,run_command())
            WrCC(f,string)
        cls.send_file(ID)        
#        cls.conn[ID].send(('command_line',string))
    
    @classmethod
    def send_file(cls,ID):
        cls.send_command(ID,'open {0}'.format(cls.full_path(ID)))
#        cls.conn[ID].send(('command_line','open {}'.format(cls.full_path(ID))))
    
    @classmethod
    def hover(cls,ID,hover=True):
        cls.conn[ID].send(('hover_on' if hover else 'hover_off',))
            
        
    @classmethod
    def kill(cls,ID):
        if ID=='all':
            for k,P in enumerate(cls.PIDs):
                os.system('kill {0}'.format(P))
        elif cls.PIDs[ID]!=0:
            os.system('kill {0}'.format(cls.PIDs[ID]))
            cls.closed[ID]=True
            cls.conn[ID].close()
    
#    @classmethod
#    def send_command(cls,ID,string):
#        cls.command_line(ID,string)   
    @classmethod
    def send_command(cls,ID,string):
        string=string.replace(' ','+')
        return os.system('curl http://127.0.0.1:{0}/run?command={1}'.format(cls.rc_port0+ID,string))
    
    @classmethod
    def close(cls,ID):
        if ID=='all':
            for k in range(len(cls.ports)):
                if not(cls.closed[k]):
                    cls.close(k)
        else:
            if not(cls.closed[ID]):
                if not(cls.conn[ID].closed):
                    cls.command_line(ID,'exit')
                    cls.conn[ID].close()
                cls.closed[ID]=True

    @classmethod
    def kill_unconnected(cls):
        for k in range(len(cls.ports)):
            if not(cls.closed[k]):
                if not(cls.isConnected(k)):
                    cls.kill(k)
                    print('Killing ID #{0}'.format(k))
                    
    @classmethod
    def isConnected(cls,ID):
        try:
            cls.conn[ID].send(('phone_home',))
        except:
            return False
        tr=Listen(cls.conn[ID])
        tr.start()
        t0=time()
        while time()-t0<1:
            if tr.response=='still here':
                return True
        return False
    
    @classmethod
    def get_sel(cls,ID):
        try:
            cls.conn[ID].send(('get_sel',))
        except:
            print('Connection failed')
            return None
        tr=Listen(cls.conn[ID])
        tr.start()
        t0=time()
        while time()-t0<1:
            if tr.response:
                return tr.response
        
class Listen(Thread):
    def __init__(self,conn):
        super().__init__()
        self.conn=conn
        self.response=None
    def run(self):
        self.response=self.conn.recv()
        
 
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


        