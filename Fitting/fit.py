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
from pyDR.misc.tools import linear_ex
from ..misc import ProgressBar

dtype=Defaults['dtype']

def fit(data,bounds='auto',parallel=False):
    """
    Performs a detector analysis on the provided data object. Options are to
    include bounds on the detectors and whether to utilize parallelization to
    solve the detector analysis. 
    
    Note that instead of setting parallel=True, one can set it equal to an 
    integer to specify the number of cores to use for parallel processing
    (otherwise, defaults to the number of cores available on the computer)
    """
    detect=data.detect.copy()
    assert detect.opt_pars.__len__()>0,"Detector object must first be optimized (e.g. run data.detect.r_auto)"
    
    ct0=1
    if hasattr(bounds,'lower') and bounds.lower()=='auto':
        bounds=False if data.detect.opt_pars['Type']=='no_opt' else True
    elif np.issubdtype(np.array(bounds).dtype, float):
        ct0=bounds
    
    
    # out=clsDict['Data'](sens=detect,src_data=data) #Create output data with sensitivity as input detectors
    out=data.__class__(sens=detect,src_data=data) #Use same class as input (usually Data, can be Data_iRED)
    out.label=data.label
#    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.source.select=data.source.select
    
    
    simple_fit=np.all(data.Rstd[0]==data.Rstd) and len(detect)==1 and not(bounds)
    # simple_fit=False
    "simple_fit: All detectors can be fit with the same pinv matrix– so just calculate it once"
    if simple_fit:
        r0=detect
        R=data.R-data.sens.R0
        Rstd=data.Rstd[0]
        if 'inclS2' in r0.opt_pars['options']: #Append S2 if used for detectors
            R=np.concatenate((R,np.array([1-data.S2]).T))
            Rstd=np.concatenate((Rstd,[data.S2std]))
        r=(r0.r.T/Rstd).T
        R/=Rstd
        pinv=np.linalg.pinv(r)
        rho=(pinv@R.T).T
        stdev=np.sqrt((pinv**2).sum(1))
        Rc=(r@rho.T).T
        
        Y=[(rho0,stdev,Rc0) for rho0,Rc0 in zip(rho,Rc)]
        
    else:
        "Prep data for fitting"
        X=list()
        for k,(R,Rstd) in enumerate(zip(data.R.copy(),data.Rstd)):
            r0=detect[k] #Get the detector object for this bond (detectors support indexing but not iteration)
            UB=r0.rhoz.max(1)*ct0 #Upper and lower bounds for fitting
            LB=r0.rhoz.min(1)*ct0 
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
    # out.Rc=np.zeros([out.R.shape[0],detect.r.shape[0]],dtype=dtype)
    out.Rc=None
    for k,y in enumerate(Y):
        out.R[k],out.Rstd[k],Rc0=y
        out.R[k]+=detect[k].R0
        # out.Rc[k]=Rc0*(Rstd if simple_fit else X[k][3]) 
        
    # if 'inclS2' in detect.opt_pars['options']:
    #     out.S2c,out.Rc=1-out.Rc[:,-1],out.Rc[:,:-1]
    # if 'R2ex' in detect.opt_pars['options']:
    #     out.R2,out.R=out.R[:,-1],out.R[:,:-1]
    #     out.R2std,out.Rstd=out.Rstd[:,-1],out.Rstd[:,:-1]
    
    
    
    out.details=data.details.copy()
    op=data.detect.opt_pars
    out.details.append('Fitted with {0} detectors (bounds:{1})'.format(op['n'],str(bounds)))
    out.details.append('Detector optimization type: {0}'.format(op['Type']))
    out.details.append('Normalizaton: {0}, NegAllow: {1}, Options:'.format(op['Normalization'],op['NegAllow'])+\
                       ', '.join(op['options']))
        
    if data.source.project is not None:data.source.project.append_data(out)
    
    return out

def opt2dist(data,rhoz=None,rhoz_cleanup=False,parallel=False):
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
    data : data object
        Data object to be optimized.
    rhoz : np.array, optional
        Provide a set of functions to replace the detector sensitivities.
        These should ideally be similar to the original set of detectors,
        but may differ somewhat. For example, if r_target is used for
        detector optimization, rhoz may be set to removed residual differences.
        Default is None (keep original detectors)
    
    rhoz_cleanup : bool, optional
        Modifies the detector sensitivities to eliminate oscillations in the
        data. Oscillations larger than the a threshold value (default 0.1)
        are not cleaned up. The threshold can be set by assigning the 
        desired value to rhoz_cleanup. Note that rhoz_cleanup is not run
        if rhoz is defined.
        Default is False (keep original detectors)

    parallel : bool, optional
        Use parallel processing to perform optimization. Default is False.

    Returns
    -------
    data object

    """
    
    out=data.__class__(sens=data.sens) #Create output data with sensitivity as input detectors
    out.label=data.label
#    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.select=data.select
    out.source=copy(data.source)
    out.source.saved_filename=None
    out.source.status='opt_fit'
    out.Rstd=data.Rstd
    out.R=np.zeros(data.R.shape)
    
    
    nb=data.R.shape[0]
    
    if data.S2c is None:
        S2=np.zeros(nb)
    else:
        S2=data.S2c if data.src_data is None else data.src_data.S2
        
        
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
    
    if rhoz_cleanup or rhoz is not None:
        if rhoz is not None:
            assert rhoz.shape[1]==data.sens.rhoz.shape[1],f"The new detectors do not have the correct number correlation times (rhoz.shape[1]!={data.sens.rhoz.shape[1]})"
            rhoz_clean=np.concatenate((rhoz,data.sens.rhoz[rhoz.shape[0]:]))
            for k,(rhoz0,rhoz) in enumerate(zip(data.sens.rhoz,rhoz)):
                overlap=(rhoz0*rhoz).sum()/np.sqrt((rhoz0**2).sum()*(rhoz**2).sum())
                if overlap<0.9:
                    print(f"Warning: Overlap of the old and new detector {k} sensitivity  is less than 0.9 ({overlap:.2f})")
        else:
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
                if 'Normalization' in data.sens.opt_pars and data.sens.opt_pars['Normalization']=='I':
                    rhoz/=rhoz.sum()*data.sens.dz
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
        out.S2c=1-np.array([d.sum() for d in dist])
        # for k in range(len(out.R)):
        #     out.R[k]-=out.sens[k].rhoz[:,-1]*out.S2c[k]
        
    out.details=data.details.copy()
    out.details.append('Data fit optimized with opt2dist (rhoz_cleanup:{0})'.format(rhoz_cleanup))
    
    if data.source.project is not None:data.source.project.append_data(out)

    return out


def model_free(data,nz:int=None,fixz:list=None,fixA:list=None,Niter:int=None,include:list=None,nsteps=1000)->tuple:
    """
    Fits a data object with model-free (i.e. multi-exponential) analysis. 
    Procedure is iterative: we start by fitting all detectors with a single
    correlation time and amplitude, and then the second correlation time and
    amplitude fit the remaining detector intensity, and so on. This procedure
    is repeated several times (set Niter to control precision), where the latter
    correlation times are the included when performing the initial fit.

    Parameters may be fixed to certain input values (use fixz, fixA), where
    these entries should be lists of length nz. Usually, just one parameter is fixed.
    In this case, the parameter may be provided as a single entry. That is, for
    nz=3, the following are equivalent
    
    model_free(data,nz=2,fixA=8/9)
    model_free(data,nz=2,fixA=[8/9,None],fixz=[None,None])
    
    Note that it is also possible to specify different fixed values for different
    residues. If this behavior is desired, then a list *must* be used, and the
    elements of the list should also be lists (or iterable) with the same length
    as the number of residues being fitted.
    
    We may also opt not to fit all detector responses. This is controlled by
    an index (for example, if we have 6 detectors). Note that regardless, we will
    always back-calculate all detectors (not just those in include)
    
    include=[0,1,2]
    include=[True,True,True,False,False,False]
    
    Note that we return amplitudes of each correlation time, not the corresponding
    S2 values (or S values). To get parameters in the form:
    (1-Sf^2)exp(-t/tauf)+Sf^2(1-Ss^2)exp(-t/\taus), we'd need to calculate
    
    Sf^2=1-A[0], Ss=1-A[1]/Sf^2
    

    Parameters
    ----------
    data : data object
        Data to be fit.
    nz : int, optional
        Number of correlation times to include. The default is 1.
    fixz : list, optional
        List with length nz. Include the log-correlation time as a list element
        to fix to a given value. Include None in the list where values should
        not be fixed. The default is None.
    fixA : list, optional
        Fix amplitudes to some input values. The default is None.
    Niter : int, optional
        Number of iterations for fitting. Will be set internally depending on
        the number of correlation times to use. The default is None.
    include : list, optional
        Index to determine which detectors to fit. The default is None.
    nsteps : int, optional
        Number of steps taken to sweep over the correlation time. Default is 1000

    Returns
    -------
    tuple
        Contains z (log correlation times), A (amplitudes), the total error 
        (chi-squared) for each residue, and a data object which contains the 
        back-calculated detector responses.

    """
    
    "Loop over project or list if provided"
    if hasattr(data,'__getitem__'):
        out=[model_free(d,nz=nz,fixz=fixz,fixA=fixA,Niter=Niter,include=include) for d in data]
        return out
    
    nz=nz if nz is not None else 1
    
    fixz=[None for _ in range(nz)] if fixz is None else \
        (fixz if hasattr(fixz,'__len__') else [fixz,*[None for _ in range(nz-1)]])
    fixA=[None for _ in range(nz)] if fixA is None else \
        (fixA if hasattr(fixA,'__len__') else [fixA,*[None for _ in range(nz-1)]])
    
    nb,nd=data.R.shape    
    
    
    op_in_rho=not(data.source.Type=='NMR')  #Order parameter adds to detector responses (order parameter in rho)
    # op_in_rho=False
    if op_in_rho and np.any(data.sens.rhoz[:,-1]>0.99):
        op_loc=np.argwhere(data.sens.rhoz[:,-1]>0.99)[0,-1]
    else:
        op_loc=None
    # op_loc=np.argwhere(data.sens.rhoz[:,-1]>0.99)[0,-1] if op_in_rho else None
    if include is None:
        include=np.ones(nd,dtype=bool)
        if op_loc is not None:include[op_loc]=False

    z0,rhoz,R,Rstd=data.sens.z,data.sens.rhoz[include],data.R[:,include],data.Rstd[:,include]
        
    Niter=4**(nz-1) if Niter is None else Niter  #I actually have no idea what to put here. Should be 1, though, for nz=1
    
    z,A=list(),list()  #Set initial values
    for k in range(nz):
        if fixz[k] is None:
            z.append(np.ones(nb)*z0[0])
        else:
            z.append(np.array(fixz[k]) if hasattr(fixz[k],'__len__') else np.ones(nb)*fixz[k])
        if fixA[k] is None:
            A.append(np.zeros(nb))
        else:
            A.append(np.array(fixA[k]) if hasattr(fixA[k],'__len__') else np.ones(nb)*fixA[k])
                
    zswp=np.linspace(z0[0],z0[-1],nsteps);
    ProgressBar(0, Niter,prefix='Iterations',suffix=f' of {Niter} steps',length=30,decimals=0)
    for q in range(Niter):
        # print('{0} of {1} iterations'.format(q+1,Niter))
        for k in range(nz):
            R0=np.zeros(R.shape)
            for m in range(nz):
                if k!=m:
                    R0+=(linear_ex(z0,rhoz.T,z[m]).T*A[m]).T
            DelR=(R-R0)/Rstd
            
            if fixz[k] is None:
                if fixA[k] is None:
                    #No fixed parameters
                    err=list()
                    A00=list()
                    
                    for z00 in zswp:
                        m=linear_ex(z0,rhoz.T,z00)/Rstd
                        if np.abs(m).max()==0:
                            err.append(np.ones(nb)*1e10)
                            continue
                        pinv=((m**2).sum(1)**(-1))*m.T
                        A00.append((pinv.T*DelR).sum(1))
                        A00[-1][A00[-1]<0]=0
                        A00[-1][A00[-1]>1]=1

                        err.append(((DelR.T-m.T*A00[-1])**2).sum(0))
                    err=np.array(err)
                    i=err.argmin(0)

                    A00=np.array(A00).T
                    A[k]=np.array([A00[k][i0] for k,i0 in enumerate(i)])
                    z[k]=zswp[i]
                else:
                    #Amplitude fixed
                    err=list()
                    for z00 in zswp:
                        m=linear_ex(z0,rhoz.T,z00)/Rstd
                        err.append(((DelR.T-m.T*A[k])**2).sum(0))
                    err=np.array(err)
                    i=err.argmin(0)
                    z[k]=zswp[i]
                    
            else:
                if fixA[k] is None:
                    #Correlation time fixed
                    #Here, we build the pseudo-inverse at the input correlation times
                    m=linear_ex(z0,rhoz.T,z[k])/Rstd
                    pinv=((m**2).sum(1)**(-1))*m.T
                    A[k]=(pinv.T*DelR).sum(1)
                    A[k][A[k]<0]=0
                    A[k][A[k]>1]=1
                else:
                    #All parameters fixed (no operations)
                    pass
        ProgressBar(q+1, Niter,prefix='Iterations',suffix=f' of {Niter} steps',length=30,decimals=0)
    #Calculate the fit

    Rc=np.zeros(data.R.shape)
    for m in range(nz):
        Rc+=(linear_ex(data.sens.z,data.sens.rhoz.T,z[m]).T*A[m]).T
    z,A=np.array(z),np.array(A)
    if op_in_rho:
        A[:,A.sum(0)>1]/=A[:,A.sum(0)>1].sum(0)
        Rc+=np.atleast_2d((1-A.sum(0))).T@np.atleast_2d(data.sens.rhoz[:,-1])

    out=copy(data)
    out.R=Rc
    err=((R-Rc[:,include])**2/Rstd**2).sum(1)
    
    out.source.details.append('Back calculation of detector responses for model free fit')
    out.source.details.append('Fitted to {0} correlation times'.format(nz))
    out.source.Type='ModelFreeFit'
    out.source.src_data=None
    
    if data.source.project is not None:data.source.project.append_data(out)
    
    

    i=z.argsort(axis=0)
    z=np.array([o[i0] for o,i0 in zip(z.T,i.T)]).T
    A=np.array([o[i0] for o,i0 in zip(A.T,i.T)]).T
    
    
    return z,A,err,out
    

