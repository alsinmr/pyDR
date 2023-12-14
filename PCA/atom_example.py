#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:00:51 2023

@author: albertsmith
"""

import numpy as np

class Test():
    def __init__(self,select):
        self.select=select
        self.select._mdmode=True
        self.uni=self.select.uni
        self._atoms0=None
        self._atoms=None
        self._sel1index=None
        self._sel2index=None
        
    @property
    def atoms(self):
        if self._atoms is None:
            
            out=self.uni.atoms[:0]
            if self._atoms0 is not None:
                out=out+self._atoms0
            if self.select.sel1 is not None:
                out=out+self.select.sel1
            if self.select.sel2 is not None:
                out=out+self.select.sel2
                
            self._atoms,i=np.unique(out,return_inverse=True)
            self._atoms=self._atoms.sum()
            self._sel1index=i[-(len(self.select.sel1+self.select.sel2)):-len(self.select.sel2)]
            self._sel2index=i[-len(self.select.sel2):]
            
        return self._atoms
    
    @atoms.setter
    def atoms(self,value):
        self._atoms=None
        
        if isinstance(value,str):
            self._atoms0=self.uni.select_atoms(value)
            return
        self._atoms0=value
        
    @property
    def sel1index(self):
        if self._sel1index is None:self._atoms
        return self._sel1index
    
    @property
    def sel2index(self):
        if self._sel2index is None:self._atoms
        return self._sel2index
            
        
        
            