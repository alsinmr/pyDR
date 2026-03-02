#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 09:51:29 2026

@author: albertsmith
"""

from pyDR.Sens import MD
import numpy as np
from pyDR.misc.tools import linear_ex
from scipy.optimize import lsq_linear as lsq
import matplotlib.pyplot as plt

class FRET(MD):
    
    def __init__(self,t,tc=None,z=None,info=None,tD=None,S=1,**kwargs):
        """
        Initial a FRET sensitivity object. Parameters are:
            t: Time points in the correlation function, given in nanoseconds
            stdev: Standard deviation at each time point (optional)
            N: Number of pairs of time points averaged at each time point (optional)
        
        One may also define the correlation time axis by setting tc or z (z is
        the log-10 correlation time)
        
        tc or z should have 2,3 or N elements
            2: Start and end
            3: Start, end, and number of elements
            N: Provide the full axis
        
        
        Note that FRET sensitivity object is not set up to be edited except
        for diffusion parameters (tD, S)
        
        """
        
        if tc is None and z is None:
            z=[np.log10(np.mean(t[:2]))-10,np.log10(t[-1])-8]
        super().__init__(t=t,tc=tc,z=z)
        
        self.tD=tD
        self.S=S
        self.zt=np.log10(t)-9 # in seconds
        self._rhoz=None
    
    def clear(self):
        self._Gdiff=None
        self._Gdiff_dist=None
        self._Reff=None
        self._rhoz=None
        return self
    
    @property
    def tD(self):
        return self._tD
    
    @tD.setter
    def tD(self,tD):
        self._tD=tD
        self.clear()
        
        
    @property
    def S(self):
        return self._S
    @S.setter
    def S(self,S):
        self._S=S
        self.clear()
        
    @property
    def zt(self):
        return self._zt
    @zt.setter
    def zt(self,zt):
        self._zt=zt
        self.clear()
    
    @property
    def Gdiff(self):
        if self.tD is None:return np.ones(self.zt.shape)
        return Gdiff(zt=self.zt,tD=self.tD,S=self.S)
    
    @property
    def Gdiff_dist(self):
        if self.tD is None:return None
        if self._Gdiff_dist is None:
            out=IVL(self.z,self.zt,self.Gdiff)
            self._Gdiff_dist=out['x']
        return self._Gdiff_dist
    
    def plot_Gdiff_dist(self,ax=None):
        if ax is None:ax=plt.subplots()[1]
        ax.plot(self.z,self.Gdiff_dist)
        ax.set_xlabel(r'$\log_{10}(\tau_c$ / s)')
        return ax
    
    def plot_Gdiff_fit(self,ax=None):
        if ax is None:ax=plt.subplots()[1]
        ax.plot(self.zt,self.Gdiff)
        M=np.exp(-np.atleast_2d(10**self.zt).T@np.atleast_2d(10**-self.z))
        ax.plot(self.zt,M@self.Gdiff_dist,color='black',linestyle=':')
        ax.set_xlabel(r'$\log_{10}(t$ / s)')
        return ax
    
    def zeff(self,zM:float):
        """
        Returns the log-effective correlation time for the internal z vector,
        given the log-rotational correlation time, zM

        Parameters
        ----------
        zM : float
            log-rotational correlation time.

        Returns
        -------
        zeff, an array of the effective correlation time

        """
        return self.z+zM-np.log10(10**self.z+10**zM)
    
    @property
    def rhoz(self):
        rhoz=super().rhoz
        if self.tD is None:return rhoz
        
        if self._rhoz is None:
            Reff=np.zeros(rhoz.shape)
            
            for z0,A0 in zip(self.z,self.Gdiff_dist):
                Reff+=A0*linear_ex(self.z,rhoz,self.zeff(z0))
            
            self._rhoz=Reff
            
        return self._rhoz
    
    def plot_rhoz(self,index=None,ax=None,norm=False,tD=None,S=None,**kwargs):
        if tD is not None:
            self.tD=tD
        if S is not None:
            self.S=S
        out=super().plot_rhoz(index=index,ax=ax,norm=norm,**kwargs)
        return out
        


def Gdiff(zt,tD,S=1):
    return 1/(1+10**zt/tD)/np.sqrt(1+10**zt/(S**2*tD))


def IVL(z,zt,f):
    M=np.exp(-np.atleast_2d(10**zt).T@np.atleast_2d(10**-z))
    out=lsq(M,f,bounds=(0,1))
    return out