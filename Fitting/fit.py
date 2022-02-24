#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 14:56:22 2022

@author: albertsmith
"""


from pyDR import Defaults,clsDict
import numpy as np
import multiprocessing as mp
from ._fitfun import fit0

dtype=Defaults['dtype']

def fit(data,bounds=True,parallel=False):
    """
    Performs a detector analysis on the provided data object. Options are to
    include bounds on the detectors and whether to utilize parallelization to
    solve the detector analysis. 
    
    Note that instead of setting parallel=True, one can set it equal to an 
    integer to specify the number of cores to use for parallel processing
    (otherwise, defaults to the number of cores available on the computer)
    """
    detect=data.detect.copy()
    out=clsDict['Data'](sens=detect,src_data=data) #Create output data with sensitivity as input detectors
    out.label=data.label
    "I think the line below is now redundant..."
#    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.select=data.select
    
    
    "Prep data for fitting"
    X=list()
    for k,(R,Rstd) in enumerate(zip(data.R.copy(),data.Rstd)):
        r0=detect[k] #Get the detector object for this bond (detectors support indexing but not iteration)
        UB=r0.rhoz.max(1)#Upper and lower bounds for fitting
        LB=r0.rhoz.min(1)
        R-=data.sens[k].R0 #Offsets if applying an effective sensitivity
        if 'inclS2' in r0.opt_pars['options']: #Append S2 if used for detectors
            R=np.concatenate((R,[1-data.S2[k]]))
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
    out.Rstd=np.zeros(out.R.shape,dtype=dtype)
    out.Rc=np.zeros([out.R.shape[0],detect.r.shape[0]],dtype=dtype)
    for k,y in enumerate(Y):
        out.R[k],out.Rstd[k],Rc0=y
        out.R[k]+=detect[k].R0
        out.Rc[k]=Rc0*X[k][3]
        
    if 'inclS2' in detect.opt_pars['options']:
        out.S2c,out.Rc=out.Rc[:,-1],out.Rc[:,:-1]
    if 'R2ex' in detect.opt_pars['options']:
        out.R2,out.R=out.R[:,-1],out.R[:,:-1]
        out.R2std,out.Rstd=out.Rstd[:,-1],out.Rstd[:,:-1]
    
    if data.source.project is not None:data.source.project.append_data(out)
    return out
    