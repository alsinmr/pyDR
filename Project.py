#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:50:19 2022

@author: albertsmith
"""

import os
import numpy as np
from pyDR.IO import read_file,write_file
from pyDR import Defaults
ME=Defaults['max_elements']
decode=bytes.decode

class DataInfo():
    def __init__(self,project):
        self.project=project
        if not(os.path.exists(self.directory)):os.mkdir(self.directory)
        self.data_objs=[None for _ in self.saved_files]
        self._hashes=[None for _ in self.saved_files]
        self.__i=-1
        
    
    @property
    def directory(self):
        return os.path.join(self.project.directory,'data')
        
    @property
    def saved_files(self):
        files=os.listdir(self.directory)
        names=list()
        for fname in files:
            # with open(os.path.join(self.directory,fname),'rb') as f:
            #     if decode(f.readline())[:-1]=='OBJECT:DATA':
            if fname[0]!='.':names.append(fname)
        return names
                
    def load_data(self,filename=None,index=None):
        "Loads a saved file from the project directory"
        if filename is not None:
            assert filename in self.saved_files,"{} not found in project. Use 'append_data' for new data".format(filename)
            index=self.saved_files.index(filename)
        self.data_objs[index]=read_file(os.path.join(self.directory,self.saved_files[index]))
        self._hashes[index]=self.data_objs[index]._hash
        self.data_objs[index].project=self
        
    def append_data(self,data):
        filename=None
        if isinstance(data,str):
            filename,data=data,None
        "Adds data to the project (either from file, set 'filename' or loaded data, set 'data')"
        if filename is not None:
            assert os.path.exists(filename),"{} does not exist".format(filename)
            data=read_file(filename)

        if data in self:
            print("Data already in project (index={})".format(self.data_objs.index(data)))
            return
            
        self.data_objs.append(data)
        self._hashes.append(None)   #We only add the hash value if data is saved
        self.data_objs[-1].project=self
        if data.src_data is not None:self.append_data(data=data.src_data) #Recursively append source data
    
    def __getitem__(self,i):
        assert i<len(self.data_objs),"Index exceeds number of data objects in this project"
        if self.data_objs[i] is None:
            self.load_data(index=i)
        return self.data_objs[i]
    
    def __next__(self):
        self.__i+=1
        if self.__i<len(self):
            return self[self.__i]
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
        """
        List of filenames for previously saved data
        """
        return [d.source.saved_filename for d in self]
    
    @property
    def save_name(self):
        """
        List of filenames used for saving data
        """
        names=list()
        for d in self:
            name=d.source.default_save_location
            if name in names:
                name=name[:-5]+'1'+name[-5:]
                k=2
                while name in names:
                    name=name[:-5]+'{}'.format(k)+name[-5:]
                    k+=1
            names.append(os.path.join(self.directory,name))
        return names
        
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
    
    def save(self,i='all'):
        """
        Save data object stored in the project by index, or set i to 'all' to
        save all data objects. Default is to derive the filename from the title.
        To save to a specific file, use data.save(filename='desired_name') instead
        of saving from the project.
        """
        if i=='all':
            for i in range(len(self)):
                if self[i].R.size>ME:
                    print('Skipping data object {0}. Size of data.R ({1}) exceeds default max elements ({2})'.format(i,self[i].R.size,ME))
                    continue
                self.save(i)
        else:
            assert i<len(self),"Index {0} to large for project with {1} data objects".format(i,len(self))
            if not(self.saved[i]):
                src_fname=None
                if self[i].src_data is not None:
                    k=np.argwhere([self[i].src_data==d for d in self])[0,0]
                    self.save(i=k)
                    src_fname=self.save_name[k]
                self[i].save(self.save_name[i],overwrite=True,save_src=False,src_fname=src_fname)
                self._hashes[i]=self[i]._hash #Update the hash so we know this state of the data is saved
          

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
    
    def append_data(self,data):
        self.data.append_data(data)
        


        