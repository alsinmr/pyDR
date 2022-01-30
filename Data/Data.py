#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:58:40 2022

@author: albertsmith
"""

from pyDR.Defaults import Defaults
from pyDR.Sens import Detector
import numpy as np

dtype=Defaults['dtype']

class Data():
    def __init__(self,R=None,R_std=None,sens=None,src_data=None):
        """
        Initialize a data object. Optional inputs are R, the data, R_std, the 
        standard deviation of the data, sens, the sensitivity object which
        describes the data, and src_data, which provides the source of this
        data object.
        """
        
        if R is not None:R=np.array(R,dtype=dtype)
        if R_std is not None:R_std=np.array(R_std,dtype=dtype)
        
        "We start with some checks on the data sizes, etc"
        if R is not None and R_std is not None:
            assert R.shape==R_std.shape,"Shapes of R and R_std must match when initializing a data object"
        if R is not None and sens is not None:
            assert R.shape[1]==sens.rhoz.shape[0],"Shape of sensitivity object is not consistent with shape of R"
        
        self.R=R if R is not None else np.zeros([0,sens.rhoz.shape[0] if sens else 0])
        self.R_std=R_std if R_std is not None else np.zeros(self.R.shape)
        self.sens=sens
        self.__detect=None
        
        
    
    def __setattr__(self, name, value):
        if name=='sens':
            if self.__detect is not None:print('Warning: Resetting "sens" deletes the current detector object')
            self.__detect=None
            assert value.__len__()==1 or value.__len__()==self.R.shape[0],"Sensitivity object length does not match length of R"
        if name=='detect':
            assert self.sens is not None,"Define 'sens' before assigning a detector object"
            if not(value.sens is self.sens):
                assert value.sens.rhoz.shape==self.sens.rhoz.shape,"Shape of detector input sensitivity does not match sensitivity of the data object"
                assert value.sens.__len__()==self.sens.__len__(),"Length of detector input sensitivity does not match sensitivty of the data object"
            if not(value.sens==self.sens):
                print('Warning: Detector input sensitivity and data sensitivity are not equal')
        super().__setattr__(name, value)

        
    
    @property
    def n_data_pts(self):
        return self.R.shape[0]
    
    @property
    def ne(self):
        return self.R.shape[1]
    
    @property
    def detect(self):
        if self.sens.edited:
            if self.__detect is not None:print("Warning: Data's sensitivity object has been edited. Detector object will also be updated")
        if self.__detect is None:self.__detect=Detector(self.sens)
        return self.__detect
        
        