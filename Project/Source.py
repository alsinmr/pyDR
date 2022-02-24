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
    def __init__(self,Type='',src_data=None,select=None,filename=None,title=None,status=None,
                 additional_info=None,saved_filename=None,n_det=None):
        self.Type=Type
        self._src_data=src_data
        self.select=select
        self.filename=filename #For record keeping. Store things like .xtc file names here, or text file with NMR data
        self.saved_filename=saved_filename
        self.n_det=n_det
        self._title=title
        self.project=None
        self.additional_info=additional_info
        self._status=status
        
        if self.src_data is not None:
            flds=['Type','select','filename']
            for f in flds:  #Pass the Type, selection, and original filename
                if getattr(self,f) is None:setattr(self,f,getattr(src_data.source,f))
            self.n_det=src_data.detect.rhoz.shape[0]
            self._status='n' if src_data.detect.opt_pars['Type']=='no_opt' else 'p'
            self.additional_info=src_data.source.additional_info

        if self._status is None: #Some default behavior for status
            
            if self.filename is None and self.n_det is None:
                self._status='empty'
            elif self.n_det is not None:
                self._status='proc'
            elif self.filename is not None:
                self._status='raw'
            
        assert self._status[0].lower() in ['r','n','p','e'],"Status should be 'raw','proc','no_opt', or 'empty'"
        
    @property
    def topo(self):
        if self.select is None:return None
        return self.select.molsys.file
    @property
    def traj(self):
        if self.select is None or self.select.molsys.traj is None:return None
        return self.select.molsys.traj.files
    @property
    def original_file(self):  #Returns the deepest
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
        if self._status[0].lower()=='n':return 'no_opt'
        if self._status[0].lower()=='p':return 'proc'
        if self._status[0].lower()=='e':return 'empty'
        assert 0,"source._status should be 'raw','proc', 'no_opt', or 'empty'"

    @property
    def title(self):
        if self._title is not None:return self._title
        title=self.status[0]
        if self.n_det is not None:title+='{}'.format(self.n_det)
        if self.Type is not None:title+=':'+self.Type.upper()
        if self.additional_info is not None:title+=':'+self.additional_info
        if self.short_file is not None:title+=':'+self.short_file
        return title
        
    @property
    def short_file(self):
        """
        Returns an abbreviated version of the data stored in self.filename
        """
        if self.filename is not None:          
            return os.path.split(self.filename[0] if isinstance(self.filename,list) else self.filename)[1]
        
    
    @property
    def default_save_location(self):
        if self.saved_filename is not None:return self.saved_filename
        disallowed=[c for c in '-#%^{}\/<>*. $!":@+`|='+"'"]
        filename=self.title
        for d in disallowed:
            filename=filename.replace(d,'_')
        return filename+'.data'
    
    def __setattr__(self,name,value):
        if name in ['title','status','src_data']:
            super().__setattr__('_'+name,value)
            return
        super().__setattr__(name,value)