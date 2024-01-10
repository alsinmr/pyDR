#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:22:53 2022

@author: albertsmith
"""

import os
from pyDR.IO import read_file
import numpy as np
from copy import copy

class Source():
    """
    Class to use with data objects to store information about the origin of
    the data.
    """
    def __init__(self,Type='',src_data=None,select=None,filename=None,title=None,status=None,
                 additional_info=None,saved_filename=None,n_det=None):
        self.project=None
        self.Type=Type
        self._src_data=src_data
        self.select=select
        self.filename=filename #For record keeping. Store things like .xtc file names here, or text file with NMR data
        self.saved_filename=saved_filename
        self.n_det=n_det
        self._title=title
        self.additional_info=additional_info
        self._status=status
        self.details=list()  #TODO use details to track data analysis
        
        if self._src_data is not None and not(isinstance(self._src_data,str)):
            #TODO work on making this less dependent on loading previous data sets
            flds=['Type','select','filename']
            for f in flds:  #Pass the Type, selection, and original filename
                if getattr(self,f) is None:setattr(self,f,getattr(src_data.source,f))
            self.n_det=src_data.detect.rhoz.shape[0] if 'n' in src_data.detect.opt_pars else 0
            if 'Type' in src_data.detect.opt_pars.keys():
                self._status='n' if src_data.detect.opt_pars['Type']=='no_opt' else 'p'
            else:
                self._status=src_data.source.status
            self.additional_info=src_data.source.additional_info

        if self._status is None: #Some default behavior for status
            
            if self.filename is None and self.n_det is None:
                self._status='empty'
            elif self.n_det is not None:
                self._status='proc'
            elif self.filename is not None:
                self._status='raw'
            
        assert self._status[0].lower() in ['r','n','p','e','o'],"Status should be 'raw','proc','no_opt', 'opt_fit', or 'empty'"
    
    def __copy__(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        for f in ['Type','_src_data','select','filename','saved_filename','n_det',
                  '_title','additional_info','_status','details']:
            setattr(out,f,copy(getattr(self,f)))
        return out
    
    def __hash__(self):
        
        out=np.sum([getattr(self,name).__hash__() for name in ['n_det','Type','short_file','additional_info','title','status']],dtype=int)
        
        if self.select is not None:
            out+=hash(self.select)%123456789
        return int(out)
            
    
    @property
    def topo(self) -> str:
        """
        Yields the location of the topology file for the selection object if a
        selection object is included in source.

        Returns
        -------
        str
            Path to the topology file.

        """
        if self.select is None:return None
        return self.select.molsys.topo
    @property
    def traj(self) -> list:
        """
        Yields a list containing the trajectory files in the selection object if
        the selection object is included in source and has an associated
        trajectory

        Returns
        -------
        list
            List of strings specifying the location of the trajectory files.

        """
        if self.select is None or self.select.molsys.traj is None:return None
        return self.select.molsys.traj.files
    
    @property
    def original_file(self) -> str:  
        """
        Iteratively descends through the source data (src_data) to find the 
        original raw data file. Returns None if no src_data in source

        Returns
        -------
        str
            String specifying the location of the original data.

        """
        source=self
        while source.src_data is not None:
            source=source.src_data.source
        return source.filename
    
    @property
    def src_data(self) -> object:
        """
        Returns the source data as a data object for the given source file. Note
        that when a data object is initially loaded, its src_data is stored 
        as a string specifying the file location, and will not be loaded unless
        specifically called by the user.

        Returns
        -------
        data
            Source data for the current processed data object.

        """
        if isinstance(self._src_data,str):
            if self.project is not None:
                if os.path.split(self._src_data)[1] in self.project.pinfo['filename']:    #Is the source data part of the project?
                    i=np.argwhere(self.project.pinfo['filename']==os.path.split(self._src_data)[1])[0,0]
                    self._src_data=self.project.data[i]      #Copy into self._src_data 
                else:   #Data not in project
                    if os.path.exists(self._src_data):
                        # self.project.append_data(self._src_data) #Append the data to the current project
                        # self._src_data=self.project[-1] #Also copy into self._src_data
                        self._src_data=read_file(self._src_data)
                    else:
                        print('Warning: source data not found at location {}'.format(self._src_data))
                        self._src_data=None
            else: #No project
                if os.path.exists(self._src_data):
                    self._src_data=read_file(self._src_data)
                else:
                    print('Warning: source data not found at location {}'.format(self._src_data))
        return self._src_data #Return the source data
    
    @property
    def status(self) -> str:
        """
        Returns the processing status of the current object.
        
        'raw': No detector processing applied
        'no_opt': Processed with unoptimized detectors (further processing allowed)
        'proc': Processed with detectors (further detector processing not recommended)
        'opt_fit': Processed with detectors followed by fit optimization
        'empty': No data stored

        Returns
        -------
        str
            String describing the current processing status.

        """
        keys={'r':'raw','n':'no_opt','p':'proc','o':'opt_fit','e':'empty'}
        if self._status.lower() in keys.values():return self._status.lower()
        if self._status[0].lower() in keys:return keys[self._status]
        self._status
        assert 0,"source.status should be '"+"','".join(keys.keys())+"', setting to ''"


    @property
    def title(self) -> str:
        """
        Returns the title for the data object. This is usually an automatically
        generated title, but the user may set the title manually.

        Returns
        -------
        str
            Title for the associated data set.

        """
        if self._title is not None:return self._title
        title=self.status[0]
        if self.n_det is not None:title+='{}'.format(self.n_det)
        if self.Type is not None:title+=':'+self.Type.upper()
        if self.additional_info is not None:title+=':'+self.additional_info
        if self.short_file is not None:title+=':'+self.short_file.rsplit('.',maxsplit=1)[0]
        return title
        
    @property
    def short_file(self) -> str:
        """
        Returns the filename of the first file in self.filename, excluding the
        filepath

        Returns
        -------
        str
            Filename without path

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
            name='_'+name
        
        super().__setattr__(name,value)
        
        if self.project is not None and \
            name in ['_title','_status','_src_data','additional_info','n_det','Type']:
            self.project.update_info()
                