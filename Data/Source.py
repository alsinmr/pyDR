#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:22:53 2022

@author: albertsmith
"""

import os
from pyDR.IO import read_file

class Source():
    """
    Class to use with data objects to store information about the origin of
    the data.
    """
    def __init__(self,Type=None,src_data=None,select=None,filename=None,title=None,status='raw',saved_filename=None,n_det=None):
        self.Type=Type
        self._src_data=src_data
        self.select=select
        self.filename=filename
        self.saved_filename=saved_filename
        self.n_det=n_det
        self._title=title
        self.project=None
        assert status[0].lower() in ['r','n','p'],"Status should be 'raw','processed' or 'no-opt'"
        self._status=status
        
    @property
    def topo(self):
        if self.select is None:return None
        return self.select.molsys.file
    @property
    def traj(self):
        if self.select is None or self.select.molsys.traj is None:return None
        return self.select.molsys.traj.files
    @property
    def original_file(self):
        source=self
        while source.src_data is not None:
            source=source.src_data.source
        return source.filename
    
    @property
    def src_data(self):
        if isinstance(self._src_data,str):
            if self.project is not None:
                if self._src_data in self.project.filenames:    #Is the source data part of the project?
                    i=self.project.filenames.index(self._src_data)
                    self._src_data=self.project[i]      #Copy into self._src_data
                else:   #Data not in project
                    if os.path.exists(self._src_data):
                        self.project.append_data(self._src_data) #Append the data to the current project
                        self._src_data=self.project[-1] #Also copy into self._src_data
                    else:
                        print('Warning: source data not found at location {}'.format(self._src_data))
            else: #No project
                if os.path.exists(self._src_data):
                    self._src_data=read_file(self._src_data)
                else:
                    print('Warning: source data not found at location {}'.format(self._src_data))
        return self._src_data #Return the source data
    
    @property
    def status(self):
        if self._status[0].lower()=='r':return 'raw'
        if self._status[0].lower()=='n':return 'no-opt'
        if self._status[0].lower()=='p':return 'processed'
        assert 0,"source._status should be 'raw','processed', or 'no-opt'"

    @property
    def title(self):
        if self._title is not None:return self._title
        title=self.status[0]
        if self.n_det is not None:title+='{}'.format(self.n_det)
        title+=self.Type.capitalize()+':'
        if self.additional_info is not None:title+=self.additional_info+':'
        title+=self.short_file
        return title
        
    @property
    def short_file(self):
        """
        Returns an abbreviated version of the data stored in self.filename
        """        
        return os.path.split(self.filename[0] if isinstance(self.filename,list) else self.filename)[1]
        
    @property
    def additional_info(self):
        """
        Link additional fields that should be included in the title here
        """
        if hasattr(self,'frame_type'):return self.frame_type
        return None
    
    @property
    def default_save_location(self):
        if self.saved_filename is not None:return self.saved_filename
        disallowed=[c for c in '-#%^{}\/<>*. $!":@+`|='+"'"]
        filename=self.title
        for d in disallowed:
            filename=filename.replace(d,'_')
        return filename+'.data'
    
    def __setattr__(self,name,value):
        if name=='title':
            super().__setattr__('_title',value)
            return
        super().__setattr__(name,value)