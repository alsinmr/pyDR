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
        self.__rho=np.zeros([0,self.__z.size])      #Store sensitivity calculations
        self.__rhoCSA=np.zeros([0,self.__z.size])   #Store sensitivity calculations
        self._bonds=list() #Store different sensitivities for different bonds
        self._parent=None  #If this is a child, keep track of the parent sensitivity
        self.__index=-1     #Index for iterating
        self.__norm=None
        self.__edited=False
        
    @property
    def norm(self):
        """
        Returns the normalization of the sensitivities for this sensitivity object
        
        If 'stdev' and 'med_val' are found in sens.info, then the normalization is
        defined as
            med_val/stdev/max(rhoz)
        This approach ensures that weighting is determined by the ratio of the
        median value and the standard deviation of the given experiment.
        
        If only 'stdev' is found in sens.info, then the normalization is defined
        as  
            1/stdev
        In this case, we assume the amplitude of the parameter is closely related
        to its sensitivity (i.e. med_val/max(rhoz) is relatively uniform)
        
        If neither are found in sens.info, we normalize by
            1/max(rhoz)
        """
        if self.__norm is None or self.info.edited:
            self._norm()
            
        return self.__norm
    
        
    
    def _norm(self):
        """
        Calculate and store the normalization.
        """
        if 'stdev' in self.info.keys and np.all(self.info['stdev']): 
            if 'med_val' in self.info.keys:
                self.__norm=(self.info['med_val'].astype(float)/self.info['stdev'].astype(float)/self._rho_eff[0].max(axis=1))
            else:
                self.__norm=1/(self.info['stdev']).astype(float)
        else:
            self.__norm=1/self._rho_eff[0].max(axis=1)
            
    @property
    def tc(self):
        return 10**self.__z
    
    @property
    def z(self):
        return self.__z.copy()
        

#%% Functions dealing with sensitivities    
    def _update_rho(self):
        """
        Updates the values of all sensitivities to current experimental parameters
        (only run in case self.info indicates that it has been edited)
        """
        if self.info.edited or self.__rho.shape[0]==0:
            self.__rho=self._rho()
            self.__rhoCSA=self._rhoCSA() if hasattr(self,'_rhoCSA') else np.zeros([self.info.N,self.z.size])
            self.info.updated()
            self._norm()

            self.__edited=True
    
    @property
    def edited(self):
        """
        Determines if the sensitivities of this object have changed
        """
        return self.__edited or self.info.edited
    
    def updated(self,edited=False):
        """
        Call self.updated if the sensitivities have been updated, thus setting
        self.edited to False
        """
        self.__edited=edited

    @property
    def rhoz(self):
        """
        Return the sensitivities stored in this sensitivity object
        """
        self._update_rho()
        return self.__rho.copy()
         
    @property
    def _rhozCSA(self):
        """
        Return the sensitivities due to CSA relaxation in this sensitivity object
        """
        self._update_rho()
        return self.__rhoCSA.copy()
    
    @property
    def _rho_eff(self):
        """
        This will be used generally to obtain the sensitivity of the object, plus
        offsets of the parameter if required, whereas rhoz, rhoz_eff, etc. may
        change depending on the subclass.
        """
        return self.rhoz,np.zeros(self.rhoz.shape[0])
    
    @property
    def _rho_effCSA(self):
        """
        This will be used generally to obtain the sensitivity of the object due
        to CSA
        """
        return self._rhozCSA,np.zeros(self._rhozCSA.shape[0])
    
#%% Properties relating to iteration over bond-specific sensitivities        
    def __len__(self):
        """
        1 or number of items in self.__bonds
        """
        return 1 if len(self._bonds)==0 else len(self._bonds)
    
    def __getitem__(self,index):
        """
        Here, we reserve the possibility to store multiple sensitivity objects
        in one main sensitivity object, for example, for relaxation occuring
        under anisotropic tumbling. List of sensitivity objects stored in bonds
        """
        
        if len(self._bonds)==0:
            return self
        else:
            assert index<len(self.__bonds),"index must be less than the number of stored sensitivity objects ({})".format(len(self.__bonds))
            return self.__bonds[index]

    def __setitem__(self,index,value):
        """
        Set bond-specific sensitivities for a given index
        """
        assert index<self._bonds,"index must be less than the number of stored sensitivity objects ({})".format(len(self.__bonds))
        assert isinstance(value,self.__class__),"Bond-specific sensitivities must have the same class as their parent sensitivity"
        self._bonds[index]=value
        self._bonds[index]._parent=self
        
    def append(self,value):
        """
        Add a bond-specific sensitivity
        """
        assert isinstance(value,self.__class__),"Bond-specific sensitivities must have the same class as their parent sensitivity"
        self._bonds.append(value)
        self._bonds[-1]._parent=self

    def __next__(self):
        """
        __next__ method for iteration
        """
        self.__index+=1
        if self.__index<self.N:
            return self.__getitem__(self.__index)
        self.__index=-1
        raise StopIteration
        
    def __iter__(self):
        """
        Iterate over the experiments
        """
        self.__index=-1
        return self

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
        
        ax.set_ylabel(r'$\rho_n(z)$')
        
        
        return hdl   