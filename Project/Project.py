#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:50:19 2022

@author: albertsmith
"""

import os
import numpy as np
from pyDR.IO import read_file
from pyDR import Defaults
ME=Defaults['max_elements']
decode=bytes.decode

class DataMngr():
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
        self.data_objs[-1].source.project=self
        if data.src_data is not None:
            if data.src_data in self:
                data.src_data=self[self.data_objs.index(data.src_data)]
            else:
                self.append_data(data=data.src_data) #Recursively append source data
    
    def remove_data(self,index,delete=False):
        """
        Remove a data object from project. Set delete to True to also delete 
        saved data in the project folder. Note that this can also delete 
        associated data. If delete is not set to True, saved data will re-appear
        upon the next project load even if removed here.
        """
            
        if isinstance(index,int):
            self.data_objs.pop(index)
            self._hashes.pop(index)
            if delete and self.filenames[index] is not None:
                os.remove(os.path.abspath(self.directory,self.filenames[index]))
        else:
            for i in np.sort(index)[::-1]:self.remove_data(i,delete=delete)
                
            
    
    def __getitem__(self,i):
        if isinstance(i,int):
            assert i<len(self.data_objs),"Index exceeds number of data objects in this project"
            if self.data_objs[i] is None:
                self.load_data(index=i)
            return self.data_objs[i]
        elif hasattr(i,'__len__'):
            return [self[i0] for i0 in i]
    
    
    def __len__(self):
        return len(self.data_objs)
    
    def __iter__(self):
        def gen():
            for k in range(len(self)):
                yield self[k]
        return gen()
        
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
                self[i].source.saved_filename=self.save_name[i]
                self._hashes[i]=self[i]._hash #Update the hash so we know this state of the data is saved
          

class Project():
    def __init__(self,directory,create=False):
        self._directory=os.path.abspath(directory)
        if not(os.path.exists(self.directory)) and create:
            os.mkdir(self.directory)
        assert os.path.exists(self.directory),'Project directory does not exist. Select an existing directory or set create=True'
               
        self.data=DataMngr(self)
        
    @property
    def directory(self):
        return self._directory
    
    def append_data(self,data):
        self.data.append_data(data)
        
    def save(self):
        self.data.save()
    
    @property
    def detectors(self):
        r=list()
        for r0 in self.data.detect_list:
            if r0 not in r:
                r.append(r0)
        return r
        
    def unify_detect(self,chk_sens_only=False):
        """
        Checks for equality among the detector objects and assigns all equal
        detectors to the same object. This allows one to only optimize one of
        the data object's detectors, and all other objects with equal detectors
        will automatically be optimized.
        
        Note that by default, two detectors with differently defined sensitivites
        will not be considered equal. However, one may set chk_sens_only=True in
        which case only the sensitivity of the detectors will be checked, so
        all detectors having the same initial sensitivity will be combined.
        """
        r=self.data.detect_list
        s=[r0.sens for r0 in r]
        for k,(s0,r0) in enumerate(zip(s,r)):
            if (s0 in s[:k]) if chk_sens_only else (r0 in r[:k]):                
                i=s[:k].index(s0) if chk_sens_only else r[:k].index(r0)
                self.data[k].sens=s[i]
                self.data[k].detect=r[i]
                
    def unique_detect(self,index=None):
        if index is None:
            for i in range(len(self.data)):self.unique_detect(i)
        else:
            d=self.data[index]
            sens=d.detect.sens
            d.detect=d.detect.copy()
            d.detect.sens=sens
    
    def __iter__(self):
        def gen():
            for k in range(len(self.data)):
                yield self.data[k]
        return gen()
    
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self,index):
        """
        Extract a data object or objects by index or title (returns one item) or
        by Type or status (returns a list).
        """
        if isinstance(index,int):
            assert index<self.__len__(),"index too large for project of length {}".format(self.__len__())
            return self.data[index]
        if isinstance(index,str):
            if index in self.Types:
                out=list()
                for k,t in enumerate(self.Types):
                    if index==t:out.append(self[k])
                return out
            elif index in self.statuses:
                out=list()
                for k,s in enumerate(self.statuses):
                    if index==s:out.append(self[k])
                return out
            elif index in self.titles:
                return self[self.titles.index(index)]
            return
        if hasattr(index,'__len__'):
            return [self[i] for i in index]
            
    
    @property
    def Types(self):
        return [d.source.Type for d in self]
    
    @property
    def statuses(self):
        return [d.source.status for d in self]
    
    @property
    def titles(self):
        return [d.title for d in self]
    
    def comparable(self,i,threshold=0.9,mode='auto',min_match=2):
        """
        Find objects that are recommended for comparison to a given object. 
        Provide either an index (i) referencing the data object's position in 
        the project or by providing the object itself. An index will be
        returned, corresponding to the data objects that should be compared to
        the given object
        
        Comparability is defined as:
            1) Having some selections in common. This is determined by running 
                self[k].source.select.compare(i,mode=mode)
            2) Having 
                a) Equal sensitivities (faster calculation, performed first)
                b) overlap of the object sensitivities above some threshold.
                This is defined by sens.overlap_index, where we require 
                at least min_match elements in the overlap_index (default=2)
        """
        if isinstance(i,int):i=self[i] #Get the corresponding data object
        print('Updated1')
        out=list()
        for s in self:
            if s.select.compare(i.select)[0].__len__()==0:
                out.append(False)
                continue
            out.append(s.sens.overlap_index(i.sens,threshold=threshold)[0].__len__()>=min_match)
        return np.argwhere(out)[:,0]
            
        
        
            
        


        