#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:58:40 2022

@author: albertsmith
"""

from pyDR.Defaults import Defaults
from pyDR.Sens import Detector
from pyDR.Fitting import fit
import numpy as np

dtype=Defaults['dtype']

class Data():
    def __init__(self,R=None,Rstd=None,sens=None,src_data=None):
        """
        Initialize a data object. Optional inputs are R, the data, R_std, the 
        standard deviation of the data, sens, the sensitivity object which
        describes the data, and src_data, which provides the source of this
        data object.
        """
        
        if R is not None:R=np.array(R,dtype=dtype)
        if Rstd is not None:Rstd=np.array(Rstd,dtype=dtype)
        
        "We start with some checks on the data sizes, etc"
        if R is not None and Rstd is not None:
            assert R.shape==Rstd.shape,"Shapes of R and R_std must match when initializing a data object"
        if R is not None and sens is not None:
            assert R.shape[1]==sens.rhoz.shape[0],"Shape of sensitivity object is not consistent with shape of R"
        
        self.R=R if R is not None else np.zeros([0,sens.rhoz.shape[0] if sens else 0],dtype=dtype)
        self.Rstd=Rstd if Rstd is not None else np.zeros(self.R.shape,dtype=dtype)
        self.sens=sens
        self.detect=Detector(sens) if sens is not None else None
        self.src_data=src_data
        
        
        
    
    def __setattr__(self, name, value):
        """Special controls for setting particular attributes.
        """
        if name=='sens' and value is not None and hasattr(self,'detect') and self.detect is not None:
            assert self.detect.sens==value,"Detector input sensitivities and data sensitivities should match"
            if self.detect.sens is not value:
                print("Warning: Detector object's input sensitivity does is not the same object as the data sensitivity.")
                print("Changes to the data sensitivity object will not be reflected in the detector behavior")
        if name=='detect' and value is not None and hasattr(self,'sens') and self.sens is not None:
            assert self.sens==value.sens,"Detector input sensitivities and data sensitivities should match"
            if self.sens is not value.sens:
                print("Warning: Detector object's input sensitivity does is not the same object as the data sensitivity.")
                print("Changes to the data sensitivity object will not be reflected in the detector behavior")
        super().__setattr__(name, value)

        
    
    @property
    def n_data_pts(self):
        return self.R.shape[0]
    
    @property
    def ne(self):
        return self.R.shape[1]
    
    
    def fit(self,bounds=True,parallel=True):
        return fit(self,bounds=bounds,parallel=parallel)
        
        