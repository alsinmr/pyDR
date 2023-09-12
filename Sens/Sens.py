#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:30:05 2021

@author: albertsmith

Some notes:
    __rho
    __rhoCSA are intended for storage of calculated sensitivities
    
    rhoz
    _rhozCSA return the sensitivities, after checking for updates
    
    Child classes should have functions
    _rho
    _rhoCSA which are called by the Sens when storing calculated sensitivities
"""

import numpy as np
from pyDR.Sens.Info import Info
from pyDR.misc.disp_tools import set_plot_attr,NiceStr
import matplotlib.pyplot as plt
# from matplotlib import ticker
from copy import deepcopy,copy
from pyDR import Defaults,clsDict

# from pyDR._Data._Data import write_file

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
        zrange=Defaults['zrange'] #Program defaults for correlation times
        if tc is not None:
            if len(tc)==2:
                self.__z=np.linspace(np.log10(tc[0]),np.log10(tc[1]),zrange[2])
            elif len(tc)==3:
                self.__z=np.linspace(np.log10(tc[0]),np.log10(tc[1]),tc[2])
            else:
                self.__z=np.array(np.log10(tc))
        elif z is not None:
            if len(z)==2:
                self.__z=np.linspace(z[0],z[1],zrange[2])
            elif len(z)==3:
                self.__z=np.linspace(*z)
            else:
                self.__z=np.array(z)
        else:
            self.__z=np.linspace(*zrange)
            
        self.info=Info()
        self.__rho=np.zeros([0,self.__z.size])      #Store sensitivity calculations
        self.__rhoCSA=np.zeros([0,self.__z.size])   #Store sensitivity calculations
        self._bonds=list() #Store different sensitivities for different bonds
        self._parent=None  #If this is a child, keep track of the parent sensitivity
        self.__index=-1     #Index for iterating
        self.__norm=None

    

    def del_exp(self,index:int):
        """
        Deletes an experiment or experiment (provide a list)

        Parameters
        ----------
        index : int or list of indices
            Experiment index, or list of indices.

        Returns
        -------
        None.

        """
        self.info.del_exp(index)
        return self
        
    
    def copy(self):
        """
        Returns a deep copy of the sensitivity object. 
        """
        return deepcopy(self)
    
    def __copy__(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.info=copy(self.info)
        return out
    
    # def save(self,filename,overwrite=False):
    #     write_file(filename,self,overwrite)
    
    def Detector(self):
        """
        Returns a detector created from the sensitivity object

        Returns
        -------
        pyDR.Sens.Detector

        """
        return clsDict['Detector'](self)
    
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
            if 'med_val' in self.info.keys and np.all(self.info['med_val']):
                self.__norm=(self.info['med_val'].astype(float)/self.info['stdev'].astype(float)/np.abs(self.rhoz).max(axis=1))
            else:
                self.__norm=1/(self.info['stdev']).astype(float)
        else:
            self.__norm=1/self.rhoz.max(axis=1)
            
    @property
    def tc(self):
        return 10**self.__z
    
    @property
    def z(self):
        return self.__z.copy()
    @property
    def dz(self):
        return self.z[1]-self.z[0]
        
    @property
    def _hash(self):
        #todo get rid of that later
        return hash(self)

    def __hash__(self):
        # if hasattr(self,'opt_pars') and 'n' not in self.opt_pars:   #Unoptimized set of detectors (hash not defined)
        #     return hash(self.sens)
        if self.rhoz.size>100000:
            return hash(self.rhoz[:,::20].tobytes())
        return hash(self.rhoz.tobytes())
        

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
    
    @property
    def R0(self):
        return self._rho_eff[1]
    
    def overlap_index(self,sens,threshold=0.95,check_amp=True):
        if self==sens:return [np.arange(self.rhoz.shape[0]) for _ in range(2)]
        x1=self.rhoz.T/np.sqrt((self.rhoz**2).sum(1))
        x2=sens.rhoz.T/np.sqrt((sens.rhoz**2).sum(1))
        mat=(x1.T@x2)
        mat[mat<threshold]=0
        for k,m in enumerate(mat):
            v,i=m.max(),np.argmax(m)
            mat[k] = 0
            mat[k, i] = v
        mat = mat.T
        for k,m in enumerate(mat):
            v, i = m.max(), np.argmax(m)
            mat[k]=0
            mat[k, i] = v
        mat=mat.T
        if check_amp:
            for k,j in np.argwhere(mat):
                rat=self.rhoz[k].max()/sens.rhoz[j].max()
                if rat<threshold or rat>1/threshold:
                    mat[k,j]=False
                    
        out=np.argwhere(mat).T
        return out[0], out[1]
        
#%% Properties relating to iteration over bond-specific sensitivities
    def __len__(self):
        """
        1 or number of items in self._bonds
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
            assert index<len(self._bonds),"index must be less than the number of stored sensitivity objects ({})".format(len(self._bonds))
            return self._bonds[index]

    def __setitem__(self,index,value):
        """
        Set bond-specific sensitivities for a given index
        """
        assert index<self._bonds,"index must be less than the number of stored sensitivity objects ({})".format(len(self._bonds))
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
        if self.__index<self.__len__():
            return self.__getitem__(self.__index)
        self.__index=-1
        raise StopIteration
        
    def __iter__(self):
        """
        Iterate over the experiments
        """
        self.__index=-1
        return self
    
    def __eq__(self, ob):
        """
        We'll define equals as having the same sensitivity for this object and
        all objects available via iteration. We will allow for minor deviations
        in the sensitivity, to allow for small errors when data is coming possible
        from different sources.
        """
        if self is ob:return True        #If same object, then equal
        if len(self)!=len(ob):return False  #If different lengths, then not equal
        
        for s,o in zip(self,ob):
            if s.rhoz.shape!=o.rhoz.shape:return False #Different sizes, then not equal
            if np.max(np.abs(s.rhoz-o.rhoz))>1e-6:return False #Different sensitivities
        return True

    #%% Plot rhoz
    def plot_rhoz(self,index=None,ax=None,norm=False,**kwargs):
        """
        Plots the sensitivities of the data object.
        """
        
        if index is None:index=np.ones(self.rhoz.shape[0],dtype=bool)
        index=np.atleast_1d(index)
            
        assert np.issubdtype(index.dtype,int) or np.issubdtype(index.dtype,bool),"index must be integer or boolean"
    
        a=self.rhoz[index].T #Get sensitivities
        a/=np.abs(a).max(0) if norm else 1 
                
        if ax is None:
            fig=plt.figure()
            ax=fig.add_subplot(111)

        hdl=ax.plot(self.z,a)
        set_plot_attr(hdl,**kwargs)
        
        # ax.set_xlim(self.z[[0,-1]])
        # ticks=ax.get_xticks()
        # nlbls=4
        # step=int(len(ticks)/(nlbls-1))
        # start=0 if step*nlbls==len(ticks) else 1
        # lbl_str=NiceStr('{:q1}',unit='s')
        # ticklabels=['' for _ in range(len(ticks))]
        # for k in range(start, len(ticks),step):ticklabels[k]=lbl_str.format(10**ticks[k])
        
        # ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        # ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
        
        
        def format_func(value,tick_number):
            prec='{:q1}' if int(value)==value else '{:q3}'
            lbl_str=NiceStr(prec,unit='s')
            return lbl_str.format(10**value)
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
#        ax.set_xticklabels(ticklabels)
        ax.set_xlabel(r'$\tau_\mathrm{c}$')
        ax.set_ylabel(r'$\rho_n(z)$')
                
        return hdl

        
        
    
    
