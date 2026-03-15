#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:33:48 2026

@author: albertsmith
"""

import numpy as np
from scipy.optimize import leastsq
from ..Sens.FRET import Gdiff
import matplotlib.pyplot as plt

def DiffOpt(t,Ct,t0=None,plot=True):
    Ct=np.atleast_2d(Ct)
    
    t00=t
    Ct00=Ct
    if t[0]==0:
        t00=t00[1:]
        Ct00=Ct00[:,1:]
    

    if t0 is not None:
        b=np.argmin(np.abs(t-t0))
        t=t[b:]
        Ct=Ct[:,b:]

    tD0=t[len(t)//2]
    S0=0.05
    
    zt=np.log10(t)
    
    A0=Ct[:,0]
    
    
    def fun(X):
        gd=Gdiff(zt,X[0],X[1])
        return np.concatenate([gd*A-g for A,g in zip(X[2:],Ct)])
    
    out=leastsq(fun, [*A0,tD0,S0])[0]
    
    if plot:
        fig,ax=plt.subplots(1,len(A0))
        ax=np.atleast_1d(ax)
        for k,(a,g) in enumerate(zip(ax,Ct00)):
            a.semilogx(t00,g,color='red')
            a.semilogx(t00,out[k+2]*Gdiff(np.log10(t00),*out[:2]),color='black',linestyle=':')
            a.set_xlabel(r'$\log_{10}(t/s)$')
            a.set_ylabel(r'$g(t)$')
            a.set_ylim([-.1,np.max(out[2:])*1.5])
        ax[0].legend(('Exp.','Fit'))
        ax[-1].text(t00[len(t00)//3],0,fr'$\tau_D$={out[0]:.3f} s'+'\n'+\
                    fr'$S={np.abs(out[1]):.3f}$')
        fig.set_size_inches([3*len(A0),3])
        fig.tight_layout()
    return out
    
    
    