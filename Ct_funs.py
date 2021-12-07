#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:12:16 2021

@author: albertsmith
"""

import numpy as np
import types
from numba import njit,prange

class Ct_calc():
    def __init__(self,A,B=None,weight=None,offset=0,index=None,sparse=1):
        """
        Manager for performing calculations of correlation functions. Initialize
        with values x,y that should be correlated. x and y may be lists of the
        same length, where we will calculate all correlation functions for x,y
        pairs, and sum together the results. One may optionally provide weighting
        for the x,y pairs (e.g. for P2, we want 1.5*xx+1.5*yy+1.5*zz+3*xy+3*xz+3*yz)
        
        If sparse sampling of the first type is used, index should also be provided
        """
        if B is None:B=A  #Take as autocorrelation if b not provided
        if not(isinstance(A,list) or isinstance(A,types.GeneratorType)):A=[A]
        if not(isinstance(B,list) or isinstance(A,types.GeneratorType)):B=[B]
        assert len(A)==len(B),'List of values to be correlated (x,y) must have the same length'
        if weight is None:weight=np.ones(len(A))
        assert len(A)==len(weight),'If weight is provided, it must have the same length as x and y'
        
        
        self.index=index
        self.sparse=sparse
        self.A=A
        self.B=B
        self.weight=weight
        self.offset=offset
        
        self.ct=[None for _ in range(len(A))] #Storage for the correlation functions
        
        self.ct_fun=None
        self.cleanup_fun=None
        
        self._mode='CtFT'
    
    @property
    def mode(self):
        """
        Determine what mode we will use to calculate the correlation function. Later,
        we can change this function to select the most efficient method for calculating
        the correlation function (first, we need to figure out what mode is most
        efficient). Currently, just returns the string found in self._mode 
        
        Note, mode should match the name of a function found in this module. 
        Each of these functions should either return two new functions (one that
        executes in run and one that executes at the end of cleanup) or just
        a simple correlation function
        """
        return self._mode
    
    def run(self):
        assert self.mode in globals(),"Function '{}' not found in module".format(self.mode)
        f=globals()[self.mode]
        self.ct_fun,self.cleanup_fun=f() if f.__code__.co_argcount==0 else (f,lambda x,index:x)
        
        for k,(a,b) in enumerate(zip(self.A,self.B)):
            code=self.ct_fun.__code__
            if 'index' in code.co_varnames[:code.co_argcount]:
                self.ct[k]=self.ct_fun(a,b,index=self.index)
            elif 'sparse' in code.co_varnames[:code.co_argcount]:
                self.ct[k]=self.ct_fun(a,b,sparse=self.sparse)
            else:
                self.ct[k]=self.ct_fun(a,b)
    
    def cleanup(self):
        assert np.any(a is None for a in self.ct),"Not all correlation functions have been calculated yet"
        out=self.ct[0]*self.weight[0]
        for ct,w in zip(self.ct[1:],self.weight[1:]):
            out+=ct*w
            
        code=self.cleanup_fun.__code__
        if 'index' in code.co_varnames[:code.co_argcount]:
            return self.cleanup_fun(out,self.index)+self.offset
        elif 'sparse' in code.co_varnames[:code.co_argcount]:
            return self.cleanup_fun(out,self.sparse)+self.offset
        else:
            return self.cleanup_fun(out)+self.offset
    
def CtFT():
    def ct(a,b,index=None):
        if index:
            a0=np.zeros([a.shape[0],index[-1]+1])
            a0[index]=a
            b0=a0
            if a is not b:b0[index]=b
            a,b=a0,b0
        A=np.fft.fft(a,a.shape[-1]<<1,axis=-1)
        B=A.conj() if a is b else np.fft.fft(b,b.shape[-1]<<1,axis=-1)
        return A*B
    def cleanup(AB,index):
        ct=np.fft.ifft(AB)[:AB.shape[0]>>1].real  #Kai's magical divide by two and yield an integer
        ct/=get_count(index) if index else np.arange(ct.shape[-1],0,-1)
        return ct
    return ct,cleanup
        
def Ct():
    def ct(a,b,index=None):
        index=index if index else np.ones(a.shape[-1])
        ct0=np.zeros([index[-1]+1,a.shape[0]])
        a,b=a.T,b.T
        for k in range(len(index)):
            ct[index[k]-index[k:]]+=a[k]*b[k:]
        return ct0
    def cleanup(ct0,index=None):
        ct=ct0.T
        ct/=get_count(index) if index else np.arange(ct.shape[-1],0,-1)
        return ct
    return ct,cleanup
    
@njit(parallel=True)
def CtJit(a,b):
    ct=np.ones(a.shape)
    for k in prange(1,a.shape[-1]):
        ct[k]=np.mean(np.array([a[n]*b[n+k] for n in prange(0,a.shape[-1]-k)]))
    return ct
    
    
        

        
        
def trunc_t_axis(nt,n=100,nr=10,**kwargs):
    """
    Calculates a log-spaced sampling schedule for an MD time axis. Parameters are
    nt, the number of time points, n, which is the number of time points to 
    load in before the first time point is skipped, and finally nr is how many
    times to repeat that schedule in the trajectory (so for nr=10, 1/10 of the
    way from the beginning of the trajectory, the schedule will start to repeat, 
    and this will be repeated 10 times)
    
    """
    
    n=np.array(n).astype('int')
    nr=np.array(nr).astype('int')
    
    if n==-1:
        index=np.arange(nt)
        return index
    
    "Step size: this log-spacing will lead to the first skip after n time points"
    logdt0=np.log10(1.50000001)/n
    
    index=list()
    index.append(0)
    dt=0
    while index[-1]<nt:
        index.append(index[-1]+np.round(10**dt))
        dt+=logdt0
        
    index=np.array(index).astype(int)

    "Repeat this indexing nr times throughout the trajectory"
    index=np.repeat(index,nr,axis=0)+np.repeat([np.arange(0,nt,nt/nr)],index.size,axis=0).reshape([index.size*nr])
    
    "Eliminate indices >= nt, eliminate repeats, and sort the index"
    "(repeats in above line lead to unsorted axis, unique gets rid of repeats and sorts)"
    index=index[index<nt]
    index=np.unique(index).astype('int')
    
    return index


def get_count(index):
    """
    Returns the number of averages for each time point in the sparsely sampled 
    correlation function
    """
    N=np.zeros(index[-1]+1)
    n=np.size(index)
   
    for k in range(n):
        N[index[k:]-index[k]]+=1
        
    return N