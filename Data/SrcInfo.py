#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:22:53 2022

@author: albertsmith
"""

class SrcInfo():
    """
    Info class to use with data objects to store information about the origin of
    the data.
    """
    def __init__(self,Type=None,src_data=None,select=None,filename=None):
        self.Type=Type
        self.src_data=src_data
        self.select=select
        self.filename=filename
        
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
        src_info=self
        while src_info.src_data is not None:
            src_info=src_info.src_data.src_info
        return src_info.filename

    def add_storage(self,name,**kwargs):
        if hasattr(self,name):
            for key,value in kwargs.items():
                setattr(getattr(self,name),key,value)
        else:
            setattr(self,name,Storage(**kwargs))
        
class Storage():
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)