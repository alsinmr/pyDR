#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:50:19 2022

@author: albertsmith
"""

import os
from pyDR.IO import read_file,write_file
from pyDR import Defaults
ME=Defaults['max_elements']
decode=bytes.decode

class DataInfo():
    def __init__(self,project):
        self.project=project
        self.data_objs=[None for _ in self.saved_files]
        self._hashes=[None for _ in self.saved_files]
        self.__i=-1
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
                
    def load_data(self,filename=None,index=None):
        "Loads a saved file from the project directory"
        if filename is not None:
            assert filename in self.saved_files,"{} not found in project. Use 'append_data' for new data".format(filename)
            index=self.saved_files.index(filename)
        self.data_objs[index]=read_file(os.path.join(self.directory,self.saved_files[index]))
        self._hashes[index]=self.data_objs[index]._hash
        
    def append_data(self,filename=None,data=None):
        "Adds data to the project (either from file, set 'filename' or loaded data, set 'data')"
        if filename is not None:
            assert os.path.exists(filename),"File does not exist"
            self.data_objs.append(read_file(filename))
            self._hashes.append(None)   #We only add the hash value if data is saved
        elif data is not None:
            assert data._hash not in self._hashes,"Data already in project (index={})".format(self._hashes.index(data._hash))
            self.data_objs.append(data)
            self._hashes.append(None)   #We only add the hash value if data is saved
        
    
    def __getitem__(self,i):
        assert i<len(self.data_objs),"Index exceeds number of data objects in this project"
        if self.data_objs[i] is None:
            self.load_data(index=i)
        return self.data_objs[i]
    
    def __next__(self):
        self.__i+=1
        if self.__i<len(self):
            return self.data_objs[self.__i]
        else:
            raise StopIteration
            self.__i=-1
    
    def __len__(self):
        return len(self.data_objs)
    
    def __iter__(self):
        self.__i=-1
        return self
        
    @property
    def titles(self):
        return [d.title for d in self]
    
    @property
    def filenames(self):
        pass
        
    @property
    def sens_list(self):
        return [d.sens for d in self]
    
    @property
    def detect_list(self):
        return [d.detect for d in self]
    
    @property
    def saved(self):
        "Logical index "
        return [h==d._hash for h,d in zip(self._hashes,self)]
    
    def save(self,i,filename=None):
        """
        Save data object stored in the project by index, or set i to 'all' to
        save all data objects. Default is to derive the filename from the title,
        although one may also specify the filename
        """
        if i=='all':
            for i in len(self):
                if self[i].R.size>ME:
                    print('Skipping data object {0}. Size of data.R ({1}) exceeds default max elements ({2})'.format(i,self[i].R.size,ME))
                    continue
                self.save(i)
        else:
            assert i<len(self),"Index {0} to large for project with {1} data objects".format(i,len(self))
            if not(self.saved[i]):
                write_file(self.filenames[i],self[i])
                self._hashes[i]=self[i]._hash
        
        
    
        
         
        

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
        


        