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
import importlib
import RemoteCMXside


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

class EventManager(Thread):
    def __init__(self,cmx):
        super().__init__()
        self.cmx=cmx
    
    @property
    def is_session_alive(self):
        if len(self.cmx.session.ui.allWindows())==0:  #If windows are all gone, chimeraX is probably been closed by the user
            print('Closing the Event Manager')
            return False
        return self.cmx.isRunning   #Check if CMXReceiver has been terminated by other means
    
    def run(self):
        print('Event manager started')
        while self.is_session_alive:
            sleep(.05)
            for name,f in self.cmx._events.items():
                try:
                    f()
                except:
                    print('Warning: Event "{}" failed, removing from event loop'.format(name))
                    self.cmx._events.pop(name)
        else:
            print('Event manager stopped')
                    
    def remove_event(self,name):
        if name in self.cmx._events.keys():
            self.cmx._events.pop(name)
        else:
            print('Event "{}" not found'.format(name))
    
    def add_event(self,name,fun):
        if hasattr(fun,'__call__'):
            self.cmx._events[name]=fun
        else:
            print('Event "{}" cannot be added. fun must be callable'.format(name))


class CMXReceiver():
    def __init__(self,session,port):
        self.session=session
        self.port=port
        self.LE=None
        self.__isRunning=True
        self.EM=EventManager(self)
        self._events={}
        
        try:
            self.client=Client(('localhost',port),authkey=b"pyDIFRATE2chimeraX")
        except:
            self.__isRunning=False
            if hasattr(self,'client'):self.client.close()
            run(self.session,'exit')
            return

        self.wait4command()
        
    @property
    def isRunning(self):
        return self.__isRunning
    
    def Stop(self):
        self.__isRunning=False
        sleep(.2) #Give the event manager a good chance to register the stop (??)
    def Start(self):
        self.EM=EventManager(self)
        self.__isRunning=True
        self.EM.start()
    
    def wait4command(self):
        if self.LE and self.LE.args:
            fun,*args=self.LE.args
            if hasattr(self,fun):
                try:
                    getattr(self,fun)(*args)
                except:
                    print('Execution of {} failed'.format(fun))    
        self.LE=ListenExec(self)
#        self.LE.isDaemon=True
        self.LE.start()
    
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
        
#    def hover_on(self):
#        self.hover=Hover(self.session)
#        self.hover.start()

    def add_event(self,name):
        if not(hasattr(CMXEvents,name)):
            print('Unknown event "{}"'.format(name))
            return
        event=getattr(CMXEvents,name)
        if event.__class__ is type: #Event is a class. First initialize
            event=event(self)
        if not(hasattr(event,'__call__')):
            print('Event "{}" must be callable'.format(name))
            return
        self.Stop() #Stop the event manager
        self._events[name]=event #Add the event
        print('Event added')
        self.Start()
        
        
    def remove_event(self,name):
        if name in self._events:
            self.Stop()             #Stop the event manager
            event=self._events.pop(name) #Remove the event
            if hasattr(event,'cleanup'):event.cleanup()  #Running delete lets us clean up the object if desired.
            self.Start()
        
#    
#    def hover_off(self):
#        if hasattr(self,'hover'):
#            self.hover.cont=False
            
        
