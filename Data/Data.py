#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:58:40 2022

@author: albertsmith
"""
import numpy as np
from .. import Defaults,clsDict
from ..IO import write_file
from .Plotting import plot_fit,DataPlots
from ..Fitting import fit,opt2dist
from matplotlib.figure import Figure

dtype=Defaults['dtype']


#%% Data object

class Data():
    def __init__(self, R=None, Rstd=None, label=None, sens=None, select=None, src_data=None, Type=None,
                 S2=None, S2std=None, Rc=None):
        """
        Initialize a data object. Optional inputs are R, the data, R_std, the 
        standard deviation of the data, sens, the sensitivity object which
        describes the data, and src_data, which provides the source of this
        data object.
        """

        # todo I was wondering, if it might be useful to replace every argument with **kwargs, since all of them are
        #  optional and just be like R = kwargs.get("R")
        #  could be even more useful like
        #   for val in ["R","R_std",...]:
        #       if kwargs.get(val):
        #           attr = np.array(val)
        #       setattr(self,val, attr)
        #  smth like this?

        if R is not None:R=np.array(R,dtype=dtype)
        if Rstd is not None:Rstd=np.array(Rstd,dtype=dtype)
        
        "We start with some checks on the data sizes, etc"
        if R is not None and Rstd is not None:
            assert R.shape==Rstd.shape,"Shapes of R and R_std must match when initializing a data object"
        if R is not None and sens is not None and len(sens.info):
            assert R.shape[1]==sens.rhoz.shape[0],"Shape of sensitivity object is not consistent with shape of R"
        
        self.R=np.array(R) if R is not None else np.zeros([0,sens.rhoz.shape[0] if sens else 0],dtype=dtype)
        self.Rstd=np.array(Rstd) if Rstd is not None else np.zeros(self.R.shape,dtype=dtype)
        self.S2=np.array(S2) if S2 is not None else None
        self.S2std=np.array(S2std) if S2std is not None else None
        self._Rc=np.array(Rc) if Rc is not None else None
        self._S2c=None
        self.label=np.array(label) if label is not None else None
        if self.label is None:
            if select is not None and select.label is not None and len(select.label)==self.R.shape[0]:
                self.label=select.label
            else:
                self.label=np.arange(self.R.shape[0],dtype=object)
        self.sens=sens
        self.detect=clsDict['Detector'](sens) if sens is not None else None
        self.source=clsDict['Source'](src_data=src_data,select=select,Type=Type)
#        self.select=select #Stores the molecule selection for this data object
        self.vars=dict() #Storage for miscellaneous variable
        
    @property
    def Rc(self):
        if self._Rc is None and hasattr(self.sens,'opt_pars') and 'n' in self.sens.opt_pars:
            print("Don't forget to check that this returns the right results")
            #TODO check that this calculation is correct
            Rc=(self.sens.r@self.R.T).T
            if 'inclS2' in self.sens.opt_pars['options']:
                self._S2c,self._Rc=1-Rc[:,-1],Rc[:,:-1]
            else:
                self._Rc=Rc
        return self._Rc
    
    @property
    def S2c(self):
        if self._S2c is None and hasattr(self.sens,'opt_pars') and\
            'n' in self.sens.opt_pars and 'inclS2' in self.sens.opt_pars['options']:
                self.Rc
        return self._S2c
    
    def _ipython_display_(self):
        print(self.title+' with {0} data points'.format(self.__len__()) +'\n'+self.__repr__())
    
    def __setattr__(self, name, value):
        """Special controls for setting particular attributes.
        """

        if name == 'sens' and hasattr(self,'detect'):
            if hasattr(value,'opt_pars'):
                value.lock() #Lock detectors that are assigned as sensitivities
            if self.detect is None:
                super().__setattr__('detect',clsDict['Detector'](value))
            elif self.detect.sens != value:
                print('Warning: Assigned sensitivity object did not equal detector sensitivities. Re-defining detector object')
                super().__setattr__('detect',clsDict['Detector'](value))
        elif name == 'detect' and value is not None:
            assert self.sens == value.sens,"detect not assigned: Detector object's sensitivity does not match data object's sensitivity"
        elif name == 'src_data':
            self.source._src_data = value
            return
        elif name in ['select', 'project']:
            setattr(self.source, name, value)
            return
        elif name in ['Rc','S2c']:
            setattr(self,'_'+name,value)
            return
        super().__setattr__(name, value)
        

    @property
    def title(self):
        return self.source.title

    @property
    def select(self):
        return self.source.select
    
    @property
    def src_data(self):
        return self.source.src_data
    
    @property
    def n_data_pts(self):
        return self.R.shape[0]
    
    @property
    def ne(self):
        return self.R.shape[1]
    
    @property
    def info(self):
        return self.sens.info if self.sens is not None else None
    
    @property
    def _hash(self) -> int:
        flds = ['R', 'Rstd', 'S2', 'S2std', 'sens']
        out = 0
        for f in flds:
            if hasattr(self, f) and getattr(self, f) is not None:
                x = getattr(self, f)
                out += x._hash if hasattr(x, '_hash') else hash(x.data.tobytes())
        return out
    
    def __len__(self):
        return self.R.shape[0]
    
    def __eq__(self, data) -> bool:
        "Using string means this doesn't break when we do updates. Maybe delete when finalized"
        assert str(self.__class__) == str(data.__class__), "Object is not the same type. == not defined"
        return self._hash == data._hash
    
    def fit(self, bounds: bool = True, parallel: bool = False):
        # todo I was a little confused by that, might be useful to rename the return function? -K
        return fit(self, bounds=bounds, parallel=parallel)
    
    def opt2dist(self,rhoz_cleanup=False,parallel=False):
        return opt2dist(self,rhoz_cleanup=rhoz_cleanup,parallel=parallel)
    
    def save(self, filename, overwrite: bool = False, save_src: bool = True, src_fname=None):
        # todo might be useful to check for src_fname? -K
        if not(save_src):
            src=self.src_data
            self.src_data=src_fname  #For this to work, we need to update bin_io to save and load by filename
        write_file(filename=filename,ob=self,overwrite=overwrite)
        if not(save_src):self.src_data=src
    
    def del_data_pt(self,index:int) -> None:
        if hasattr(index,'__len__'):
            index=[i%len(self) for i in index]
            for i in np.sort(index)[::-1]:
                self.del_data_pt(i)
        else:
            index%=len(self)
            flds = ['R', 'Rstd', 'S2', 'S2std', 'label','Rc']
            for f in flds:
                if hasattr(self,f) and getattr(self,f) is not None:
                    setattr(self,f,np.delete(getattr(self,f),index,axis=0))
    
    def plot(self, errorbars=False, style='plot', fig=None, index=None, rho_index=None, plot_sens=True, split=True,**kwargs):
        # todo maybe worth to remove the args and put them all into kwargs? -K
        """
        Plots the detector responses for a given data object. Options are:
        
        errorbars:  Show the errorbars of the plot (False/True or int)
                    (default 1 standard deviation, or insert a constant to multiply the stdev.)
        style:      Plot style ('plot','scatter','bar')
        fig:        Provide the desired figure object (matplotlib.pyplot.figure) or provide
                    an integer specifying which of the project's figures to append
                    the plot to (if data attached to a project)
        index:      Index to specify which residues to plot (None or logical/integer indx)
        rho_index:  Index to specify which detectors to plot (None or logical/integer index)
        plot_sens:  Plot the sensitivity as the first plot (True/False)
        split:      Break the plots where discontinuities in data.label exist (True/False)  
        
        By default, if the data has an associated project, this will create a plot within the
        project, unless a figure
        """
        if self.source.project is None or fig.__class__ is Figure:
            "Don't append the plot to the project"
            return DataPlots(data=self, style=style, errorbars=errorbars, index=index,
                             rho_index=rho_index, plot_sens=plot_sens, split=split,fig=fig,**kwargs)
        else:
            self.source.project.plot(data=self,style=style,errorbars=errorbars,index=index,
                             rho_index=rho_index, plot_sens=plot_sens, split=split,
                             fig=fig, **kwargs)
            return self.source.project.plots[self.source.project.current_plot-1]
            
            
    
    def plot_fit(self,index=None, exp_index=None, fig=None):
        # todo maybe worth to remove the args and put them all into kwargs? -K
        assert self.src_data is not None and hasattr(self, 'Rc') and self.Rc is not None,\
            "Plotting a fit requires the source data(src_data) and Rc"
        info = self.src_data.info.copy()
        lbl, Rin, Rin_std=[getattr(self.src_data,k) for k in ['label','R','Rstd']]
        Rc=self.Rc
        if 'inclS2' in self.sens.opt_pars['options']:
            info.new_exper(Type='S2')
            Rin=np.concatenate((Rin,np.atleast_2d(1-self.src_data.S2).T),axis=1)
            Rc=np.concatenate((Rc,np.atleast_2d(1-self.S2c).T),axis=1)
            Rin_std=np.concatenate((Rin_std,np.atleast_2d(self.src_data.S2std).T),axis=1)
        
        return plot_fit(lbl=lbl,Rin=Rin,Rc=Rc,Rin_std=Rin_std,\
                    info=info,index=index,exp_index=exp_index,fig=fig)
        
            
            
    

 

    


