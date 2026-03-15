#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:47:25 2026

@author: albertsmith
"""

import numpy as np
from .. import clsDict
from copy import copy


def loadFRET(t,gDD=None,gAA=None,gAD=None,Ct=None,Ct_std=None,tD:float=None,S:float=1,label=None,sens=None):
    """
    Loads FRET time-correlation functions into a data object, including 
    estimation of the error, via a moving variance. tD and S (diffusion 
    parameters) may be provide here, or later via the sensitivity object.
    
    Currently, pyDR does not support loading FRET from a file.

    Parameters
    ----------
    t : np.array
        Time axis of the correlation function, given in ns.
    Ct : np.array
        Correlation functions. Set to np.nan where the correlation function is
        not defined for the given time axis. Should by mxn where m is the 
        number of correlation functions, and n is the length of the correlation
        functions.
    Ct_std : np.array
        User-defined standard deviation. Will be calculated from a moving 
        variance if not provided.
    tD : float, optional
        Correlation time. The default is None.
    S : float, optional
        Ratio of the x/y width vs. z depth of the confocal volume.
        The default is 1.
    label : list-like, optional
        Labels for the correlation functions. The default is None.

    Returns
    -------
    data

    """
    
    assert Ct is not None or (gDD is not None and gAA is not None and gAD is not None),"Provide Ct or gDD,gAA,and gAD"
    
    
    if Ct is None:
        if label is None:label=np.arange(gDD.shape[0])
        Ct=np.zeros([0,gDD.shape[1]])
        lbl=[]
        for gDD0,gAA0,gAD0,lbl0 in zip(gDD,gAA,gAD,label):
            Ct=np.concatenate((Ct,np.atleast_2d(gDD0),np.atleast_2d(gAA0),np.atleast_2d(gAD0)),axis=0)
            lbl.extend([f'{lbl0}_{x}' for x in ['DD','AA','AD']])
        label=lbl
    else:
        label=np.arange(Ct.shape[0])
    
    if Ct_std is None:
        var=np.ones([11,Ct.shape[0],Ct.shape[1]+10])*np.nan
        for k in range(10):
            var[k,:,k:-10+k]=Ct
        var[10,:,10:]=Ct
        var=var[:,:,5:-5]
        n=np.logical_not(np.isnan(var)).sum(0)
        var[np.isnan(var)]=0
        var-=var.sum(0)/n
        Ct_std=np.sqrt((var**2).sum(0)/(n-1))
        
    Ct_std[np.isnan(Ct)]=10000 #Knock out np.nan data
    Ct[np.isnan(Ct)]=0
    
    
    

    if sens is None:
        sens=clsDict['FRET'](t=t,stdev=Ct_std.mean(0),tD=tD,S=S)
    stdev=copy(sens.info['stdev'])
    data=DataFRET(sens=sens,R=Ct,Rstd=np.repeat(np.atleast_2d(stdev),Ct.shape[0],axis=0),label=label)
    
    data.info['stdev']=stdev
    data.info.new_parameter(med_val=np.median(np.abs(Ct),axis=0))
    
    return data


class DataFRET(clsDict['Data']):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._E=None
        self._sigmaE=None
        self._AD0=None
        
    @property
    def E(self):
        if self.Rc is None and self.sens.t[0]!=0:return
        if self._E is None:
            if self.Rc is not None:
                x=self.Rc[::3][:,0]
                y=self.Rc[1::3][:,0]
            else:
                x=self.R[::3][:,0]
                y=self.R[1::3][:,0]
            q=np.sqrt(x/y)
            self._E=q/(1+q)
            self._sigmaE=np.sqrt(y)*self._E
            self._AD0=self._sigmaE**2/(self._E*(1-self._E))
        return self._E

    @property
    def sigmaE(self):
        if self.Rc is None and self.sens.t[0]!=0:return
        self.E
        return self._sigmaE
    
    @property
    def AD0(self):
        if self.Rc is None and self.sens.t[0]!=0:return
        self.E
        return self._AD0
    
    @property
    def AD(self):
        if self.Rc is None and self.sens.t[0]!=0:return
        if self.Rc is not None:
            return self.Rc[2::3][:,0]
        else:
            return self.R[2::3][:,0]
        
        