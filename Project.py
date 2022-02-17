#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:50:19 2022

@author: albertsmith
"""

import os
from pyDR.IO import read_file
decode=bytes.decode

class DataInfo():
    def __init__(self,project):
        self.project=project
        self.data_objs=[None for _ in self.saved_files]
        self._hashes=[None for _ in self.saved_files]
        if not(os.path.exists(self.directory)):os.mkdir(self.directory)
    
    @property
    def directory(self):
        return os.path.join(self.project.directory,'data')
        
    @property
    def saved_files(self):
        files=os.listdir(self.directory)
        names=list()
        for fname in files:
            with open(os.path.join(self.directory,fname),'rb') as f:
                if decode(f.readline())[:-1]=='OBJECT:DATA':
                    names.append(fname)
        return names
                
    def load_file(self,name=None,index=None):
        if name is not None:
            assert name in self.saved_files,"{} not found in project".format(name)
            index=self.saved_files.index(name)
        
        self.data_objs[index]=read_file(os.path.join(self.directory,self.saved_files[index]))
        
    def __getitem__(self,i):
        assert i<len(self.data_objs),"Index exceeds number of data objects in this project"
        if self.data_objs[i] is None:
            self.load_file(index=i)
        return self.data_objs[i]
        

class Project():
    def __init__(self,directory,create=False):
        self._directory=os.path.abspath(directory)
        if not(os.path.exists(self.directory)) and create:
            os.mkdir(self.directory)
        assert os.path.exists(self.directory),'Project directory does not exist. Select an existing directory or set create=True'
               
        self.data=DataInfo(self)
        
    @property
    def directory(self):
        return self._directory
        


        