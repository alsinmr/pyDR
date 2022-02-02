#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:58:40 2022

@author: albertsmith
"""

from pyDR.Defaults import Defaults
from pyDR.Sens import Detector
import numpy as np
import multiprocessing as mp
from scipy.optimize import lsq_linear as lsq


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
                print("Warning: Detector object's input sensitivity is not the same object as the data sensitivity.")
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
        
#%% Functions for fitting the data
def fit(data,bounds=True,parallel=True):
    """
    Performs a detector analysis on the provided data object. Options are to
    include bounds on the detectors and whether to utilize parallelization to
    solve the detector analysis. 
    
    Note that instead of setting parallel=True, one can set it equal to an 
    integer to specify the number of cores to use for parallel processing
    (otherwise, defaults to the number of cores available on the computer)
    """
    detect=data.detect.copy()
    out=Data(sens=detect) #Create output data with sensitivity as input detectors
    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.src_data=data
    
    "Prep data for fitting"
    X=list()
    for k,(R,Rstd) in enumerate(zip(data.R,data.Rstd)):
        r0=detect[k] #Get the detector object for this bond
        UB=r0.rhoz.max(1)#Upper and lower bounds for fitting
        LB=r0.rhoz.min(1)
        R-=data.sens[k].R0 #Offsets if applying an effective sensitivity
        if 'inclS2' in r0.opt_pars['options']: #Append S2 if used for detectors
            R=np.concatenate((R,[data.S2[k]]))
            Rstd=np.concatenate((Rstd,[data.S2std[k]]))
        R/=Rstd     #Normalize data by its standard deviation
        r=(r0.r.T/Rstd).T   #Also normalize R matrix
        
        X.append((r,R,(LB,UB) if bounds else None,Rstd))
    
    "Perform fitting"
    if parallel:
        nc=parallel if isinstance(parallel,int) else mp.cpu_count()
        with mp.Pool(processes=nc) as pool:
            Y=pool.map(fit0,X)
    else:
        Y=[fit0(x) for x in X]
    
    "Extract data into output"
    out.R=np.zeros([len(Y),detect.r.shape[1]],dtype=dtype)
    out.R_std=np.zeros(out.R.shape,dtype=dtype)
    out.Rc=np.zeros([out.R.shape[0],detect.r.shape[0]],dtype=dtype)
    for k,y in enumerate(Y):
        out.R[k],out.R_std[k],Rc0=y
        out.R[k]+=detect[k].R0
        out.Rc[k]=Rc0*X[k][3]
        
    if 'inclS2' in detect.opt_pars['options']:
        out.S2c,out.Rc=out.Rc[:,-1],out.Rc[:,:-1]
    if 'R2ex' in detect.opt_pars['options']:
        out.R2,out.R=out.R[:,-1],out.R[:,:-1]
        out.R2std,out.Rstd=out.Rstd[:,-1],out.Rstd[:,:-1]
        
    return out
    
    

def fit0(X):
    """
    Used for parallel fitting of data. Single argument in the input should
    include data, the r matrix, and the upper and lower bounds
    """
    Y=lsq(X[0],X[1],bounds=X[2])
    rho=Y['x']
    Rc=Y['fun']+X[1]
    stdev=np.sqrt((np.linalg.pinv(X[0])**2).sum(1))
    return rho,stdev,Rc        