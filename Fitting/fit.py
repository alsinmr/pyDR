#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 14:56:22 2022

@author: albertsmith
"""


from pyDR import Defaults,clsDict
import numpy as np
import multiprocessing as mp
from ._fitfun import fit0,dist_opt
from copy import copy

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
    # out=clsDict['Data'](sens=detect,src_data=data) #Create output data with sensitivity as input detectors
    out=data.__class__(sens=detect,src_data=data) #Use same class as input (usually Data, can be Data_iRED)
    out.label=data.label
#    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.source.select=data.source.select
    
    
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
        out.S2c,out.Rc=1-out.Rc[:,-1],out.Rc[:,:-1]
    if 'R2ex' in detect.opt_pars['options']:
        out.R2,out.R=out.R[:,-1],out.R[:,:-1]
        out.R2std,out.Rstd=out.Rstd[:,-1],out.Rstd[:,:-1]
    
    
    
    out.details=data.details.copy()
    op=data.detect.opt_pars
    out.details.append('Fitted with {0} detectors (bounds:{1})'.format(op['n'],str(bounds)))
    out.details.append('Detector optimization type: {0}'.format(op['Type']))
    out.details.append('Normalizaton: {0}, NegAllow: {1}, Options:'.format(op['Normalization'],op['NegAllow'])+\
                       ', '.join(op['options']))
        
    if data.source.project is not None:data.source.project.append_data(out)
    
    return out

def opt2dist(data,rhoz_cleanup=False,parallel=False):
    """
    Forces a set of detector responses to be consistent with some given distribution
    of motion. Achieved by performing a linear-least squares fit of the set
    of detector responses to a distribution of motion, and then back-calculating
    the detectors from that fit. Set rhoz_cleanup to True to obtain monotonic
    detector sensitivities: this option eliminates unusual detector due to 
    oscilation and negative values in the detector sensitivities. However, the
    detectors are no longer considered "DIstortion Free".
                            

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    rhoz_cleanup : TYPE, optional
        DESCRIPTION. The default is False. If true, we use a threshold for cleanup
        of 0.1, although rhoz_cleanup can be set to a value between 0 and 1 to
        assign the threshold manually

    Returns
    -------
    data object

    """
    
    out=clsDict['Data'](sens=data.sens) #Create output data with sensitivity as input detectors
    out.label=data.label
#    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.select=data.select
    out.source=copy(data.source)
    out.source.status='opt_fit'
    out.Rstd=data.Rstd
    out.R=np.zeros(data.R.shape)
    
    
    nb=data.R.shape[0]
    
    if data.S2c is None:
        S2=np.zeros(nb)
    else:
        S2=data.src_data.S2
        
    sens=data.sens

    "data required for optimization"
    X=[(R,R_std,sens[k].rhoz,S2r) for k,(R,R_std,S2r) in enumerate(zip(data.R,data.Rstd,S2))]        
    
    if parallel:
        nc=parallel if isinstance(parallel,int) else mp.cpu_count()
            
        with mp.Pool(processes=nc) as pool:
            Y=pool.map(dist_opt,X)
    else:
        Y=[dist_opt(X0) for X0 in X]
    
    
    dist=list()
    for k,y in enumerate(Y):
        out.R[k]=y[0]
        dist.append(y[1])
    
    if rhoz_cleanup:
        threshold=rhoz_cleanup if not(isinstance(rhoz_cleanup,bool)) else 0.1
        if not(str(data.sens.__class__)==str(clsDict['Detector'])):
            print('rhoz_cleanup only permitted on detector responses (no raw data)')
            return
        if data.sens.opt_pars['Type']=='no_opt':
            print('rhoz_cleanup not permitted on un-optimized detectors')
            return
        
        rhoz_clean=list()
        for rhoz in data.sens.rhoz.copy():  
            below_thresh=rhoz<threshold*rhoz.max()
            ind0=np.argwhere(np.diff(below_thresh))[:,0]        
            if len(ind0)==1 and below_thresh[-1]: #One maximum at beginning
                ind=np.diff(np.diff(rhoz[ind0[-1]:])<0)>0
                if np.any(ind):
                    ind=np.argwhere(ind)[0,0]+ind0[0]+1
                    rhoz[ind:]=0
            elif len(ind0)==1 and below_thresh[0]:  #One maximum at the end
                ind=np.diff(np.diff(rhoz[:ind0[0]])<0)>0
                if np.any(ind):
                    ind=np.argwhere(ind)[-1,0]+1
                    rhoz[:ind]=0
            elif len(ind0)==2 and below_thresh[0] and below_thresh[-1]: #One maximum in the middle
                ind1=np.diff(np.diff(rhoz[ind0[-1]:])<0)>0
                ind2=np.diff(np.diff(rhoz[:ind0[0]])<0)>0
                if np.any(ind1):
                    ind=np.argwhere(ind1)[0,0]+ind0[-1]+1
                    rhoz[ind:]=0
                if np.any(ind2):
                    ind=np.argwhere(ind2)[-1,0]+1
                    rhoz[:ind]=0         
            rhoz[rhoz<0]=0 #Eliminate negative values
            rhoz_clean.append(rhoz)
            
        out.sens=copy(out.sens)
        rhoz_clean=np.array(rhoz_clean)
        out.sens._Sens__rho=rhoz_clean
        out.R=np.array(dist)@rhoz_clean.T
        
        # in0,in1=np.argwhere(rhoz<threshold*rhoz.max())[[0,-1],0]
        out.detect=clsDict['Detector'](out.sens)

    # Rc=list()
    # if 'inclS2' in sens.opt_pars['options']:
    #     for k in range(out.R.shape[0]):
    #         R0in=np.concatenate((sens.sens[k].R0,[0]))
    #         Rc0=np.dot(sens[k].r,out.R[k,:])+R0in
    #         Rc.append(Rc0[:-1])
    # else:
    #     for k in range(out.R.shape[0]):
    #         Rc.append(np.dot(sens[k].r,out.R[k,:])+sens.sens[k].R0)
    # out.Rc=np.array(Rc)
    if data.S2c is not None:
        out.S2c=np.array([d.sum() for d in dist])
        
        
    out.details=data.details.copy()
    out.details.append('Data fit optimized with opt2dist (rhoz_cleanup:{0})'.format(rhoz_cleanup))
    
    if data.source.project is not None:data.source.project.append_data(out)
    
    return out