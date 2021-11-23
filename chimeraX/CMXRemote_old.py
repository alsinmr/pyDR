#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:11:14 2021

@author: albertsmith
"""

"""
Reference:
https://www.cgl.ucsf.edu/chimerax/docs/user/commands/remotecontrol.html
"""
import numpy as np
import os
import sys
sys.path.append('/Users/albertsmith/Documents/GitHub')
from pyDR.chimeraX.chimeraX_funs import get_path,py_line,WrCC,chimera_path,run_command
from multiprocessing.connection import Listener,Client
from time import sleep
from threading import Thread


class CMXRemote():
    ports=list()
    PIDs=list()
    closed=list()
    port0=60958
    
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
    
    @classmethod
    def full_path(cls,ID):
        return get_path('chimera_script{0:06d}.py'.format(ID))     #Location to write out chimera script        
    
    @classmethod
    def send_command(cls,ID,string):
        string=string.replace(' ','+')
        return os.system('curl http://127.0.0.1:{0}/run?command={1}'.format(cls.port0+ID,string))
        
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
    
    @classmethod
    def send_file(cls,ID):
        cls.send_command(ID,'open {0}'.format(cls.full_path(ID)))
    
    @classmethod
    def send_object(cls,ID,obj):
        with File(ID) as f:
            py_line(f,'from multiprocessing.connection import Listener,Client')
            py_line(f,'listener=Listener(("localhost",{0}),authkey=b"pyDIFRATE password")'.format(6000+ID))
#            py_line(f,'client=Client(("localhost",{0}),authkey=b"pyDIFRATE password")'.format(7000+ID))
            py_line(f,'conn=listener.accept()')
            py_line(f,'while True:')
            py_line(f,'obj=conn.recv()',1)
            py_line(f,'if hasattr(fun,"__call__"):',1)
            py_line(f,'conn.close()',2)
            py_line(f,'break',2)
            py_line(f,'client.send("received")')
            py_line(f,'client.close()')
            py_line(f,'fun()')
        cls.send_file(ID)
        client=Client(('localhost',6000+ID),authkey=b'pyDIFRATE password')
#        listener=Listener(('localhost',7000+ID),authkey=b'pyDIFRATE password')
#        conn=listener.accept()
#        while conn.recv()!='received':
        client.send(obj)
        client.close()
     
    @classmethod
    def get_sel(cls,ID):
        listener=Listener(('localhost',7000+ID),authkey=b'pyDIFRATE password')
        with File(ID) as f:
            py_line(f,'from multiprocessing.connection import Listener,Client')
            py_line(f,'from time import time')
            py_line(f,'t0=time()')
            py_line(f,'while time()-t0<10:')
            py_line(f,'try:',1)
            py_line(f,'client=Client(("localhost",{0}),authkey=b"pyDIFRATE password")'.format(7000+ID),2)
            py_line(f,'break',2)
            py_line(f,'except:',1)
            py_line(f,'pass',2)
            py_line(f,'sel=list()')
            py_line(f,'for k,mdl in enumerate(session.models):')
            py_line(f,'if mdl.selected:',1)
            py_line(f,'sel.append({"model":k})',2)
            py_line(f,'a0,a1=mdl.bonds[mdl.bonds.selected].atoms',2)
            py_line(f,'sel[-1]["b0"]=a0.coord_indices',2)
            py_line(f,'sel[-1]["b1"]=a1.coord_indices',2)
            py_line(f,'sel[-1]["a"]=mdl.atoms[mdl.atoms.selected].coord_indices',2)
            py_line(f,'client.send(sel)')
            py_line(f,'client.close()')
        out=Thread(target=cls.send_file,args=(ID,))     #Why does this need to be a separate thread?
        out.start()
        try:
            conn=listener.accept()
            out=conn.recv()
            conn.close()
            listener.close()
            return out
        except:
            pass
        finally:
            conn.close()
            listener.close()
    
    
    @classmethod
    def kill(cls,ID):
        if ID=='all':
            for k,P in enumerate(cls.PIDs):
                os.system('kill {0}'.format(P))
        else:
            if not(cls.closed[ID]):
                os.system('kill {0}'.format(cls.PIDs[ID]))
    
    @classmethod
    def close(cls,ID):
        if ID=='all':
            for k in range(len(cls.ports)):
                if not(cls.closed[k]):
                    cls.send_command(k,'exit')
                    cls.closed[k]=True
        else:
            if not(cls.closed[ID]):
                cls.send_command(ID,'exit')
                cls.closed[ID]=True
    
class File():
    def __init__(self,ID):
        self.filename=CMXRemote.full_path(ID)
    def __enter__(self):
        self.file=open(self.filename,'w')
        return self.file
    def __exit__(self,exception_type, exception_value, traceback):
        self.file.close()
    
if __name__=='__main__':
    CMXRemote.launch()
    