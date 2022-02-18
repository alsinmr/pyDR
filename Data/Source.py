#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:22:53 2022

@author: albertsmith
"""

import os

class Source():
    """
    Info class to use with data objects to store information about the origin of
    the data.
    """
    def __init__(self,Type=None,src_data=None,select=None,filename=None,title=None,status='raw',saved_filename=None):
        self.Type=Type
        self.src_data=src_data
        self.select=select
        self.filename=filename
        self.saved_filename=saved_filename
        self._title=title
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
    def status(self):
        if self._status[0].lower()=='r':return 'raw'
        if self._status[0].lower()=='n':return 'no-opt'
        if self._status[0].lower()=='p':return 'processed'
        assert 0,"source._status should be 'raw','processed', or 'no-opt'"

    @property
    def title(self):
        if self._title is not None:return self._title
        title=self.status[0]+self.Type.capitalize()+':'
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
    
    def __setattr__(self,name,value):
        if name=='title':
            super().__setattr__('_title',value)
            return
        super().__setattr__(name,value)