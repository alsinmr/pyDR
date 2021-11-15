#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:30:05 2021

@author: albertsmith
"""

import numpy as np
from pyDR.Sens.Info import Info
from pyDR.misc.disp_tools import set_plot_attr,NiceStr
import matplotlib.pyplot as plt
from matplotlib import ticker

class Sens():
    def __init__(self,tc=None,z=None):
        """
        Parent class for all sensitivity objects. Input tc or z (linear correlation
        time or log-10 correlation time.
        
        tc or z should have 2,3 or N elements
            2: Start and end
            3: Start, end, and number of elements
            N: Provide the full axis
        
        """

        if tc is not None:
            if len(tc)==2:
                self.__z=np.linspace(np.log10(tc[0]),np.log10(tc[1]),200)
            elif len(tc)==3:
                self.__z=np.linspace(np.log10(tc[0]),np.log10(tc[1]),tc[2])
            else:
                self.__z=np.array(np.log10(tc))
        elif z is not None:
            if len(z)==2:
                self.__z=np.linspace(z[0],z[1],200)
            elif len(z)==3:
                self.__z=np.linspace(*z)
            else:
                self.__z=np.array(z)
        else:
            self.__z=np.linspace(-14,-3,200)
            
        self.info=Info()
        self.__rho=np.zeros([0,self.__z.size])
        self.__rho_eff=list()
        self._bonds=list() #Store different sensitivities for different bonds
        self._parent=None  #If this is a child, keep track of the parent sensitivity
            
    @property
    def tc(self):
        return 10**self.__z
    
    @property
    def z(self):
        return self.__z.copy()
        
    @property
    def rhoz(self):
        if self.info.edited or self.__rho.shape[0]==0:
            self.__rho=self._rhoz()
            self.info.updated()
        return self.__rho.copy()
    
    @property
    def _rho_eff(self):
        return self.rhoz,np.zeros(self.rhoz.shape[0])
        
    
    def __getitem__(self,index):
        """
        Here, we reserve the possibility to store multiple sensitivity objects
        in one main sensitivity object, for example, for relaxation occuring
        under anisotropic tumbling. List of sensitivity objects stored in bonds
        """
        
        if len(self.__bonds)==0:
            return self
        else:
            assert index<len(self.__bonds),"index must be less than the number of stored sensitivity objects ({})".format(len(self.__bonds))
            return self.__bonds[index]


    def plot_rhoz(self,index=None,ax=None,norm=False,**kwargs):
        """
        Plots the sensitivities of the data object.
        """
        
        if index is None:index=np.ones(self.rhoz.shape[0],dtype=bool)
        index=np.atleast_1d(index)
            
        assert np.issubdtype(index.dtype,int) or np.issubdtype(index.dtype,bool),"index must be integer or boolean"
    
        a=self.rhoz[index].T #Get sensitivities
        
        if norm:
            norm_vec=np.max(np.abs(a),axis=0)
            a=a/np.tile(norm_vec,[np.size(self.z),1])      
        
        if ax is None:
            fig=plt.figure()
            ax=fig.add_subplot(111)

        hdl=ax.plot(self.z,a)

        set_plot_attr(hdl,**kwargs)
        
        
        ax.set_xlim(self.z[[0,-1]])
        ticks=ax.get_xticks()
        nlbls=4
        step=int(len(ticks)/(nlbls-1))
        start=0 if step*nlbls==len(ticks) else 1
        lbl_str=NiceStr('{:q1}',unit='s')
        ticklabels=['' for _ in range(len(ticks))]
        for k in range(start,len(ticks),step):ticklabels[k]=lbl_str.format(10**ticks[k])
        
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
        
#        ax.set_xticklabels(ticklabels)
        ax.set_xlabel(r'$\tau_\mathrm{c}$')
        
        ax.set_ylabel(r'$\rho_n(z) [a.u.]$')
        
        
        return hdl   