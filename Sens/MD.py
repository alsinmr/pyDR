#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:11:41 2021

@author: albertsmith
"""
import numpy as np
from pyDR.Sens import Sens

class MD(Sens):
    def __init__(self,tc=None,z=None,info=None,t=None,stdev=None,N=None):
        """
        Initial an MD sensitivity object. Parameters are:
            t: Time points in the correlation function, given in nanoseconds
            stdev: Standard deviation at each time point (optional)
            N: Number of pairs of time points averaged at each time point (optional)
        
        One may also define the correlation time axis by setting tc or z (z is
        the log-10 correlation time)
        
        tc or z should have 2,3 or N elements
            2: Start and end
            3: Start, end, and number of elements
            N: Provide the full axis
        
        
        Note that MD sensitivity object is not set up to be edited (although in
        principle you can edit the info object as usual).
        
        By default, we set the standard deviation (if it is not provided) to be
        proportional np.sqrt(t[-1]/t)
        """
        
        super().__init__(tc=tc,z=z)
        
        if info is not None:
            self.info=info
        else:
            assert t is not None,"t must be provided"
            
            self.info.new_parameter(t=t)
            if stdev is None:
                stdev=np.sqrt(t[1:]/t[-1])
                stdev=np.concatenate(([stdev[0]/1e3],stdev))
            self.info.new_parameter(stdev=stdev)
            if N is not None:self.info.new_parameter(N)
            
    @property
    def t(self):
        """
        Return the time axis
        """
        return self.info['t'].astype(float)
    
    def _rho(self):
        """
        Calculates and returns the sensitivities of all time points in the correlation function
        """
        return np.exp(-np.atleast_2d(self.t*1e-9).T@(1/np.atleast_2d(self.tc)))
    
    def plot_rhoz(self,index=None,ax=None,norm=True,**kwargs):
        """
        Plots the sensitivities of the correlation function object.
        
        By default (index=None), not all sensitivities will be plotted. Instead,
        we will plot up to 41 sensitivities, log-spaced along the time-point axis.
        This can be overwridden by setting the index to include the desired 
        time points
        """
        
        if index is None:
            tlog=np.logspace(np.log10(self.t[1]),np.log10(self.t[-1]),40)
            tlog=np.concatenate(([0],tlog))
            index=np.unique([np.argmin(np.abs(tl-self.t)) for tl in tlog])
        hdl=super().plot_rhoz(index=index,ax=ax,norm=norm,**kwargs)
        ax=hdl[0].axes
        ax.set_ylabel(r'$\rho_{C(t)}(z)$ [a.u.]')
        
        return hdl
    