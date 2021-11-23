#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:25:09 2021

@author: albertsmith
"""


from multiprocessing.connection import Client
from chimerax.core.commands import run
from time import sleep
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
        fun,*args=self.cmx.client.recv()
        if hasattr(self.cmx,fun):
            getattr(self.cmx,fun)(*args)

class CMXReceiver():
    def __init__(self,session,port):
        self.session=session
        self.port=port
        
        try:
            self.client=Client(('localhost',port),authkey=b"pyDIFRATE2chimeraX")
        except:
            if hasattr(self,'client'):self.client.close()
            run(self.session,'exit')
            return
        


        self.wait4command()
            # self.tr.isDaemon=True       
        self.commands=list()
        # self.tr.start()
        
    def wait4command(self):
        self.tr=ListenExec(self)
        self.tr.isDaemon=True
        self.tr.start()
    
    def command_line(self,string):
        self.commands.append(string)
        print(string)
        run(self.session,string)
        self.wait4command()
        
