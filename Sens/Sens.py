#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:30:05 2021

@author: albertsmith
"""

import numpy as np
from Info import Info
import matplotlib.pyplot as plt

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
        
        self.plot_pars={'ylabel':r'$\rho_n(z)$','plt_index':None}
            
    @property
    def tc(self):
        return 10**self.__z
    
    @property
    def z(self):
        return self.__z.copy()
        
    @property
    def rhoz(self):
        if self.info.edited or self.__rho.shape[0]==0:
            self.__rho=self.__rhoz(self.info)
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


    def plot_rhoz(self,index=None,ax=None,bond=None,norm=False,mdl_num=None,**kwargs):
        """
        Plots the sensitivities of the data object.
        """
        
        if index is None:
            index=self.plot_pars['plt_index'] if self.plot_pars['plt_index']\
                is not None else np.ones(self.rhoz.shape[0],dtype=bool)
        index=np.array(index)
            
        assert np.issubdtype(index.dtype,int) or np.issubdtype(index,bool),"index must be integer or boolean"
    
        a=self.rhoz[index] #Get sensitivities
        
        if norm:
            norm_vec=np.max(np.abs(a),axis=0)
            a=a/np.tile(norm_vec,[np.size(self.z),1])      
        
        if ax is None:
            fig=plt.figure()
            ax=fig.add_subplot(111)

        hdl=ax.plot(self.z(),a)

        
        _set_plot_attr(hdl,**kwargs)
        
            
        ax.set_xlabel(r'$\log_{10}(\tau$ / s)')
        if norm:
            ax.set_ylabel(r'$R$ (normalized)')
        else:
            ax.set_ylabel(r'$R$ / s$^{-1}$')
        ax.set_xlim(sens.z()[[0,-1]])
        ax.set_title('Sensitivity (no model)')
        
    #    fig.show()
        return hdl   