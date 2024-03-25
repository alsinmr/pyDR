#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:02:59 2021

@author: albertsmith
"""


import numpy as np
import os
from multiprocessing.connection import Listener
from pyDR.chimeraX.chimeraX_funs import get_path,py_line,WrCC,chimera_path,run_command
from threading import Thread
from time import time,sleep
from platform import platform
from subprocess import Popen,check_output,DEVNULL
from pyDR import clsDict

#%% Functions to run in pyDR
class CMXRemote():
    ports=list()
    PIDs=list()
    path=os.path.dirname(os.path.abspath(__file__))
    port0=7000
    rc_port0=60958
    conn=list()
    # listeners=list()
    closed=list()
    
    
    #%% Open and close chimeraX instances
    @classmethod
    def launch(cls,commands=None):
        cls.cleanup()
        if True in cls.closed:
            ID=np.argwhere(cls.closed)[0,0]
        else:
            while True:
                ID=len(cls.ports)
                cls.PIDs.append(None)
                cls.conn.append(None)
                # cls.listeners.append(None)
                cls.closed.append(True)
                cls.ports.append(cls.port0+ID)
                
                try:
                    """So, I am not super happy about this. Problem is this: the curl 
                    port does not work if another terminal has also launched a chimera 
                    session with the same ID. What actually happens is that the commands
                    sent from here end up at the first session. So, what we do is
                    try to send a command to the curl port. If we're successful, we 
                    DO NOT use this port and move on to the next one. If an error
                    is thrown, we end up in the exception, which breaks the
                    while loop. -A.S.
                    """
                    _=check_output('curl http://127.0.0.1:{0}/run?command={1}'.format(ID+cls.rc_port0,' '),shell=True,
                                        stderr=DEVNULL)
                except:
                    break
                    
            

        with File(ID) as f:
            py_line(f,'import sys')
            py_line(f,run_command())
            WrCC(f,'remotecontrol rest start port {0}'.format(ID+cls.rc_port0))
            py_line(f,'sys.path.append("{}")'.format(cls.path))
            py_line(f,f'sys.path.append("{os.path.split(cls.path)[0]}")')
            py_line(f,'from RemoteCMXside import CMXReceiver as CMXR')
            py_line(f,'import RemoteCMXside')
            py_line(f,'cmxr=CMXR(session,{},rc_port0={})'.format(cls.ports[ID],cls.rc_port0+ID))
            WrCC(f,'ui mousemode right select')
            if commands:
                if isinstance(commands,list):
                    for co in commands:
                        WrCC(f,co)
                elif isinstance(commands,str):
                    WrCC(f,commands)

        cls.listener=Listener(('localhost',cls.ports[ID]),authkey=b'pyDIFRATE2chimeraX')

        cls.PIDs[ID] = (os.spawnl(os.P_NOWAIT, chimera_path(), chimera_path(), cls.full_path(ID))
                        if not "Linux" in platform() else
                        Popen([chimera_path().strip(), cls.full_path(ID)]))

        cls.tr=StartThread(cls.listener)
        cls.tr.start()
        t0=time()
        while time()-t0<10:     #Timeout connection after 10s
            if not(cls.tr.is_alive()):
                cls.conn[ID]=cls.tr.conn 
                cls.listener.close()    #Once the connection is made, we close the listener
                # cls.listeners[ID]=NewListen(cls.conn[ID])
                # cls.listeners[ID].start()
                break     #Connection successfully made if we reach this point
        else:
            cls.conn[ID]=None
            cls.listener.close()
            print('Failed to establish connection with ChimeraX')
            cls.kill(ID)

        cls.closed[ID]=False
        return ID
    
    
    @classmethod
    def cleanup(cls):
        for k in range(len(cls.PIDs)):
            if cls.conn[k] is None or cls.conn[k].closed:
                cls.kill(k)
    
    @classmethod
    def kill(cls,ID):
        if ID=='all':
            for k,P in enumerate(cls.PIDs):
                os.system('kill {0}'.format(P))
        elif cls.PIDs[ID]!=0 and cls.PIDs[ID] is not None:
            os.system('kill {0}'.format(cls.PIDs[ID]))
            if len(cls.closed)>ID and cls.closed[ID] is not None:
                cls.closed[ID]=True
            if cls.conn[ID] is not None:
                cls.conn[ID].close()
                # cls.listeners[ID].stop()

    
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
                    # cls.listeners[ID].stop()
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
        if ID>=len(cls.PIDs):return False
        try:
            cls.conn[ID].send(('phone_home',))
        except:
            cls.kill(ID)
            cls.closed[ID]=True
            return False
        tr=Listen(cls.conn[ID])
        tr.start()
        t0=time()
        while time()-t0<5:
            if tr.response=='still here':
            # if cls.listeners[ID].response=='still here':
                return True
        return False
    
    
    
    @classmethod
    def full_path(cls,ID):
        return get_path('chimera_script{0:02d}.py'.format(ID))     #Location to write out chimera script 

    #%% Send commands    
    @classmethod
    def send_command(cls,ID:int,string:str):
        """
        Send a command via string to chimeraX via http and curl

        Parameters
        ----------
        ID : int
            ID of the ChimeraX instance.
        string : str
            Command to be executed on the ChimeraX command line.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # cls.conn[ID].send(('command_line',string))
        # return
        
        # string=string.replace(' ','+')
        
        #Entries in encoding were obtained at https://meyerweb.com/eric/tools/dencoder/. 
        #If additional symbols cause problems, please add them to the encoding dictionary
        
        #The % has to come first because all the other signs will introduce a %
        encoding={'%':'%25','@':'%40','#':'%23','$':'%24','^':'5E','&':'%26',
                  '[':'%5B',']':'%5D','{':'%7B','}':'%7D','"':'%22',"'":'%27',
                  ' ':'%20','+':'%2B','|':'%7C','(':'\(',')':'\)'}
        
        for k,v in encoding.items():
            string=string.replace(k,v)

        out = check_output('curl http://127.0.0.1:{0}/run?command={1}'.format(cls.rc_port0+ID,string),shell=True,
                            stderr=DEVNULL)

        return out
        # return os.system('curl http://127.0.0.1:{0}/run?command={1}'.format(cls.rc_port0+ID,string))
    
    @classmethod
    def py_command(cls,ID:int,string:str) -> None:
        """
        Send a command via str to by executed as a python command. Transfered via
        text file.

        Parameters
        ----------
        ID : int
            ID of the chimeraX instance.
        string : str
            Command to be executed in the ChimeraX python command line.

        Returns
        -------
        None.

        """
        with File(ID) as f:
            py_line(f,string)
        cls.send_file(ID)
    
    
    @classmethod
    def command_line(cls,ID:int,string:str) -> None:
        """
        Send a command to be executed on the ChimeraX command line, but via 
        run_command in the python interface.
        
        At the moment, this is necessary because using run() from within the
        event loop crashes chimeraX and using curl seems not to transfer certain
        characters correctly. Probably, the latter problem can be solved. I
        suspect the former cannot be– A.S.

        Parameters
        ----------
        ID : int
            ID of the chimeraX instance.
        string : str
           Command to be executed on the ChimeraX command line.

        Returns
        -------
        None.

        """
        # Let's try to phase out this functionality and replace with curl
        cls.send_command(ID,string)
        return
        
        with File(ID) as f:
            py_line(f,run_command())
            if isinstance(string,list):
                for s in string:WrCC(f,s)
            else:
                WrCC(f,string)
        cls.send_file(ID)        
#        cls.conn[ID].send(('command_line',string))
    
    @classmethod
    def send_file(cls,ID:int):
        """
        Send a file to chimeraX to be executed as a python script. Note that
        this file is always the file stored in cls.full_path(ID)

        Parameters
        ----------
        ID : int
            ID of the chimeraX instance.

        Returns
        -------
        None.

        """
        cls.send_command(ID,'open {0}'.format(cls.full_path(ID)))
#        cls.conn[ID].send(('command_line','open {}'.format(cls.full_path(ID))))


#%% Non-event actions
    @classmethod
    def show_sel(cls,ID:int,ids:np.ndarray,color:tuple=(1.,0.,0.,1.)):
        """
        Highlight a selection of atoms by id in ChimeraX. Provide the chimeraX
        session ID, an array of ids (numpy integer array), and optionally a
        color tuple (3 or 4 elements, 0 to 1 or 0 to 255)

        Parameters
        ----------
        ID : int
            ChimeraX session ID.
        ids : np.ndarray
            Selection ids.
        color : tuple, optional
            Color to use. The default is (1,0,0,1).

        Returns
        -------
        None.

        """
        if np.max(color)<=1 and not(isinstance(color[0],int)):
            color=[int(c*255) for c in color]
        else:
            color=[int(c) for c in color]
        if len(color)==3:
            color.append(255)
        cls.conn[ID].send(('show_sel',ids,color))
#%% Event handling
    @classmethod
    def add_event(cls,ID,name,*args):
        # print(name,args)
        cls.conn[ID].send(('add_event',name,*args))

    @classmethod
    def remove_event(cls,ID,name):
        cls.conn[ID].send(('remove_event',name))
        
    @classmethod
    def set_event_attr(cls,ID,name,attr_name,value):
        cls.conn[ID].send(('set_event_attr',name,attr_name,value))
            
    
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
            out=tr.response
            # out=cls.listeners[ID].response
            if out:
                return out
#%% Execute function (not in event loop)
    @classmethod
    def run_function(cls,ID,name,*args):
        cls.conn[ID].send(('run_function',name,*args))
        
#%% Various queries    
    @classmethod
    def how_many_models(cls,ID:int,w_atoms=True)->int:
        """
        Queries chimeraX to determine how many atom-containing models are 
        currently loaded in chimeraX.
    

        Parameters
        ----------
        ID : int
            ID of chimeraX session.

        Returns
        -------
        int
            How many models currently open in chimeraX

        """
        try:
            cls.conn[ID].send(('how_many_models',w_atoms))
        except:
            print('Connection failed')
            return None
        tr=Listen(cls.conn[ID])
        tr.start()
        t0=time()
        while time()-t0<1:
            # out=cls.listeners[ID].response
            out=tr.response
            if out:
                return out
        return 0
    
    @classmethod
    def valid_models(cls,ID:int)->list:
        """
        Queries chimeraX to determine what models with atoms are open.
    

        Parameters
        ----------
        ID : int
            ID of chimeraX session.

        Returns
        -------
        list
            List of valid model indices

        """
        try:
            cls.conn[ID].send(('valid_models',))
        except:
            print('Connection failed')
            return []
        tr=Listen(cls.conn[ID])
        tr.start()
        t0=time()
        while time()-t0<1:
            # out=cls.listeners[ID].response
            out=tr.response
            if out:
                return out if isinstance(out,list) else []
        return []
    
    @classmethod
    def how_many_atoms(cls,ID:int,mdl_num:int)->int:
        """
        Queries chimeraX to determine how many atoms are in a given model. 
        
        Returns -1 if the model does not exist or does not contain atoms
        
        returns session.models[mdl_num].atoms.__len__()

        Parameters
        ----------
        ID : int
            ID of chimeraX session.
        mdl_num : int
            Model number (session.models[mdl_num])

        Returns
        -------
        int
            DESCRIPTION.

        """
        try:
            cls.conn[ID].send(('how_many_atoms',mdl_num))
        except:
            print('Connection failed')
            return None
        tr=Listen(cls.conn[ID])
        tr.start()
        t0=time()
        while time()-t0<1:
            # out=cls.listeners[ID].response
            out=tr.response
            if out:
                return out
        return -2
        
    
#%% Thread handling        
class Listen(Thread):
    def __init__(self,conn):
        super().__init__()
        self.conn=conn
        self.response=None
    def run(self):
        self.response=self.conn.recv()
        
        
# class NewListen(Thread):
#     def __init__(self,conn):
#         super().__init__()
#         self.conn=conn
#         self.response=None
#         self.running=True
#     def run(self):
#         while self.running:
#             sleep(.1)
#             try:
#                 self.response=self.conn.recv()
#                 for k in range(30):
#                     if self.response is None:
#                         break
#                     sleep(.1)
                    
#                 self.response=None
#             except:
#                 self.response=None
#             if self.conn.closed:
#                 self.stop()
#     @property
#     def response(self):
#         out=self._response
#         self._response=None
#         return out
#     @response.setter
#     def response(self,response):
#         self._response=response
#     def stop(self):
#         self.running=False
            
        
 
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

from pyDR.misc.disp_tools import NiceStr
clsDict[0]=NiceStr('pyDIFRATE')
        