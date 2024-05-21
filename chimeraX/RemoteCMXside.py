#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:25:09 2021

@author: albertsmith
"""


from multiprocessing.connection import Client
from chimerax.core.commands import run
from threading import Thread
from time import sleep
import CMXEvents
import RemoteChimeraFuns as RCF
import os
import numpy as np

class ListenExec(Thread):
    """
    This is run as a thread. For most of the time it sits at the line
    self.args = self.cmx.client.recv() and does not proceed until a message comes
    in. If a message is received, then we execute wait4command in the CMXReceiver.
    
    CMXReceiver.wait4command() gets the commands via args stored here.
    """
    def __init__(self, cmx):
        super().__init__()
        self.cmx = cmx
        self.args = None
        
    def run(self):
        try:
            self.args = self.cmx.client.recv()
            self.cmx.wait4command()     #If we successfully receive, start the process again   
        except:     #If the connection closes, we'll close this side as well
            self.cmx.client.close()

class EventManager(Thread):
    def __init__(self, cmx):
        super().__init__()
        self.cmx = cmx
        self.running=False

    @property
    def is_session_alive(self):
        return self.cmx.isRunning   #Check if CMXReceiver has been terminated by other means
    
    def run(self):
        # print('Event manager started')
        while self.is_session_alive:
            self.running=True
            sleep(.03)
            if self.cmx.session.ui.main_window.isActiveWindow(): #This checks if chimera is terminated (chimera hangs on close without this)
                for name,f in self.cmx._events.copy().items():
                    try:
                        f()
                    except:
                        print('Warning: Event "{}" failed, removing from event loop'.format(name))
                        self.cmx._events.pop(name)
        else:
            self.running=False
            # print('Event manager stopped')


class CMXReceiver():
    def __init__(self,session,port,rc_port0):
        self.session=session
        self.port=port
        self.LE=None
        self.Start()
        self._events={}
        self.rc_port0 = rc_port0   #CURL port
        
        try:
            self.client=Client(('localhost',port),authkey=b"pyDIFRATE2chimeraX")
        except:
            self.__isRunning=False
            if hasattr(self,'client'):self.client.close()
            print('fail')
            run(self.session,'exit')
            return

        self.wait4command()
     
        session.ui.aboutToQuit.connect(self.Stop,no_receiver_check=False)
        session.ui.aboutToQuit.connect(self.client.close,no_receiver_check=False)
    
    @property
    def isRunning(self):
        "When False, event manager will halt"
        return self.__isRunning     #Get the status of the run flag
    
    def Stop(self):
        "Stop the event manager"
        self.__isRunning=False      #Set the run flag to False
        while self.EM.running:
            sleep(.03)
            pass
        # sleep(.5) #Give the event manager a good chance to register the stop (??)
    def Start(self):
        "Start the event manager"
        self.EM=EventManager(self)  #Create the new event manager
        self.__isRunning=True       #Set run flag to true
        self.EM.start()             #Start the event manager
    
    def wait4command(self):
        if self.LE and self.LE.args:
            fun, *args = self.LE.args
            if hasattr(self, fun): #If fun is in this class, then we run it with the provided args
                try:
                    # self.session.ui.thread_safe(getattr(self, fun),*args)
                    getattr(self,fun)(*args)
                except:
                    print('Execution of {} failed'.format(fun))
        if self.LE is None:
            self.LE = ListenExec(self)
            self.LE_1 = self.LE
        else:
            self.LE = ListenExec(self)
#        self.LE.isDaemon=True
        self.LE.start()
        # print(self.LE_1.is_alive())
    
    def command_line(self, string):
        '''running this from inside an event will cause a crash of chimerax'''
        # print("run command")
        self.session.ui.thread_safe(run,self.session,string)
        # run(self.session, string)
        
        
    def phone_home(self):
        self.client.send('still here')
        
    
    def get_atoms(self):
        """
        Returns a list of all atom groups in the current chimera session

        Returns
        -------
        None.

        """
        atoms=list()
        for m in self.session.models:
            if hasattr(m,'atoms'):
                atoms.append(m.atoms)
        return atoms
    
    def shift_position(self,index:int,shift=[0,0,0]):
        """
        Shifts the position of a selected atom group. Index corresponds to 
        which order it was added.

        Parameters
        ----------
        index : int
            DESCRIPTION.
        shift : TYPE, optional
            DESCRIPTION. The default is [0,0,0].

        Returns
        -------
        None.

        """
        if len(self.get_atoms())<=index:
            print('index exceeds number of atom groups')
            return
        self.get_atoms()[index].coords+=np.array(shift)
    
    def show_sel(self,ids:np.ndarray,color=(255,0,0,255)):
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
            Color to use. The default is (255,0,0,255).

        Returns
        -------
        None.

        """
        for k in range(len(self.session.models),0,-1):
            if hasattr(self.session.models[k-1],'atoms'):
                model=self.session.models[k-1]
                break
        model.atoms[ids].colors=color
    
    def play_traj(self,topo:str,traj:str):
        """
        Opens a trajecory in chimeraX and sets up the play settings

        Parameters
        ----------
        topo : str
            Topology file.
        traj : str
            Trajectory file.

        Returns
        -------
        None.

        """
        print("open '{0}' coordset true".format(topo))
        
        run(self.session,"open 2kj3")
        # run(self.session,"open '{0}' coordset true".format(topo))
        n=len(self.get_atoms())
        print("open '{0}' structureModel #{1}".format(traj,n))
        # run(self.session,"open '{0}' structureModel #{1}".format(traj,n))
        # run(self.session,"coordset slider #{0}".format(n))
        
    def Exit(self):
        try:
            self.client.close()
        except:
            pass
        run(self.session, 'exit')
        
    def get_sel(self):
        sel = list()
        for k, mdl in enumerate(self.session.models):
            if mdl.selected:
                sel.append({"model": k})
                b0,b1=mdl.bonds[mdl.bonds.selected].atoms
                sel[-1]['b0'] = b0.coord_indices
                sel[-1]['b1'] = b1.coord_indices
                sel[-1]['a'] = mdl.atoms[mdl.atoms.selected].coord_indices
        self.client.send(sel)
        
    def how_many_models(self,w_atoms=True):
        """
        Tells pyDR how many atom-containing models are open in ChimeraX

        Returns
        -------
        None.

        """
        count=0
        for k in range(len(self.session.models),0,-1):
            if w_atoms:
                if hasattr(self.session.models[k-1],'atoms'):count+=1
            else:
                count+=1
        
        self.client.send(count)
        
    def valid_models(self):
        """
        Tells pyDR the indices of valid models (#1,#2, etc.)
                                                
        Returns
        -------
        None.

        """
        counter=0
        mdls=list()
        for m in self.session.models:
            if hasattr(m,'atoms'):
                if m.parent is None or not(hasattr(m.parent,'atoms')):
                    counter+=1
                    mdls.append(counter)
            else:
                if m.parent is None:
                    counter+=1
        self.client.send(mdls)
        
    def how_many_atoms(self,mdl_num):
        """
        Tells how many atoms are in a model

        Parameters
        ----------
        mdn_num : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if len(self.session.models)<=mdl_num:
            self.client.send(-1)
            return
        if not(hasattr(self.session.models[mdl_num],'atoms')):
            self.client.send(-1)
            return
        self.client.send(len(self.session.models[mdl_num].atoms))
        
        
    def send_command(self,string):
        """todo I found out, that creating any Model with the command line interface inside the thread will cause a program
        crash. since I want to create the buttons inside chimera i needed to implement this funciton from the remote side here
        it is working, still i think its ugly
        the port should be saved somewhere in the class object"""
        string=string.replace(' ','+')
        return os.system('curl http://127.0.0.1:{0}/run?command={1}'.format(self.rc_port0,string))

    def add_event(self,name,*args):
        # todo adding the a second event will cause the event manager to tell me add_event failed, but actually
        # todo it is still working
        if not(hasattr(CMXEvents,name)):
            print('Unknown event "{}", available events:\n'.format(name))
            print([fun for fun in dir(CMXEvents) if fun[0]!="_"])
            return
        event=getattr(CMXEvents,name)
        if event.__class__ is type: #Event is a class. First initialize
            event=event(self,*args)
        if not(hasattr(event,'__call__')):
            print('Event "{}" must be callable'.format(name))
            return
        
        self.Stop() #Stop the event manager
        if name in self._events:
            k=0
            while f'{name}{k}' in self._events:
                k+=1
            name=f'{name}{k}'
            
            # name0=name+'0'
            # k=0
            # while name0 in self._events:
            #     k+=1
            #     name0=name+str(k)
            # name=name0
        self._events[name]=event #Add the event
        self.Start()
        
        
    def remove_event(self,name):
        if name in self._events:
            self.Stop()             #Stop the event manager
            event=self._events.pop(name) #Remove the event
            if hasattr(event,'cleanup'):event.cleanup()  
            self.Start()
            
    def run_function(self,name:str,*args):
        """
        Runs a function in ChimeraX that does not get added to the event loop.
        This is used for one-time operations in ChimeraX
        
        For example, tensor display is currently implemented only as a one-time
        event

        Parameters
        ----------
        name : str
            Name of the function to be run (should exist in RemoteChimeraFuns)
        *args : TYPE
            Arguments to be passed to the function.

        Returns
        -------
        None.

        """
        if hasattr(RCF,name):
            fun=getattr(RCF,name)
            self.session.ui.thread_safe(fun,self,*args)
            # getattr(RCF,name)(self,*args)
        else:
            print(f'No function "{name}" found in RemoteChimeraFuns')
        
        
        