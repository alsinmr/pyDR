#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:58:40 2022

@author: albertsmith
"""
import numpy as np
import matplotlib.pyplot as plt
from .. import Defaults,clsDict
from ..IO import write_file
from .data_plots import plot_rho,plot_fit
from ..misc.disp_tools import set_plot_attr
from ..Fitting import fit

dtype=Defaults['dtype']


#%% Data object

class Data():
    def __init__(self,R=None,Rstd=None,label=None,sens=None,select=None,src_data=None,Type=None,S2=None,S2std=None,Rc=None):
        """
        Initialize a data object. Optional inputs are R, the data, R_std, the 
        standard deviation of the data, sens, the sensitivity object which
        describes the data, and src_data, which provides the source of this
        data object.
        """
        
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
        self.Rc=np.array(Rc) if Rc is not None else None
        self.label=label
        if self.label is None:
            if select is not None and select.label is not None and len(select.label)==self.R.shape[0]:
                self.label=select.label
            else:
                self.label=np.arange(self.R.shape[0],dtype=object)
        self.sens=sens
        if self.Rstd.shape[0]>0 and Rstd is not None:
            self.sens.info.new_parameter(stdev=np.median(self.Rstd,0))
        self.detect=clsDict['Detector'](sens) if sens is not None else None
        self.source=clsDict['Source'](src_data=src_data,select=select,Type=Type)
#        self.select=select #Stores the molecule selection for this data object
        self.vars=dict() #Storage for miscellaneous variable
        
        
    
    def __setattr__(self, name, value):
        """Special controls for setting particular attributes.
        """

        if name=='sens' and hasattr(self,'detect'):
            if hasattr(value,'opt_pars'):value.lock() #Lock detectors that are assigned as sensitivities
            if self.detect is None:
                super().__setattr__('detect',clsDict['Detector'](value))
            elif self.detect.sens!=value:
                print('Warning: Assigned sensitivity object did not equal detector sensitivities. Re-defining detector object')
                super().__setattr__('detect',clsDict['Detector'](value))
                
        if name=='detect' and value is not None:
            assert self.sens==value.sens,"detect not assigned: Detector object's sensitivity does not match data object's sensitivity"

        if name=='src_data':
            self.source._src_data=value
            return
        if name in ['select','project']:
            setattr(self.source,name,value)
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
    def _hash(self):
        flds=['R','Rstd','S2','S2std','sens']
        out=0
        for f in flds:
            if hasattr(self,f) and getattr(self,f) is not None:
                x=getattr(self,f)
                out+=x._hash if hasattr(x,'_hash') else hash(x.data.tobytes())
        return out
    
    
    def __eq__(self,data):
        "Using string means this doesn't break when we do updates. Maybe delete when finalized"
        assert str(self.__class__)==str(data.__class__),"Object is not the same type. == not defined"
        return self._hash==data._hash
    
    def fit(self,bounds=True,parallel=False):
        return fit(self,bounds=bounds,parallel=parallel)
    
    def save(self,filename,overwrite=False,save_src=True,src_fname=None):
        if not(save_src):
            src=self.src_data
            self.src_data=src_fname  #For this to work, we need to update bin_io to save and load by filename
        write_file(filename=filename,ob=self,overwrite=overwrite)
        if not(save_src):self.src_data=src
        
    def plot(self,errorbars=False,style='plot',fig=None,index=None,rho_index=None,plot_sens=True,split=True,**kwargs):
        """
        Plots the detector responses for a given data object. Options are:
        
        errorbars:  Show the errorbars of the plot (False/True or int)
                    (default 1 standard deviation, or insert a constant to multiply the stdev.)
        style:      Plot style ('plot','scatter','bar')
        fig:        Provide the desired figure object (matplotlib.pyplot.figure)
        index:      Index to specify which residues to plot (None or logical/integer indx)
        rho_index:  Index to specify which detectors to plot (None or logical/integer index)
        plot_sens:  Plot the sensitivity as the first plot (True/False)
        split:      Break the plots where discontinuities in data.label exist (True/False)  
        """
        if rho_index is None:rho_index=np.arange(self.R.shape[1])
        if index is None:index=np.arange(self.R.shape[0])
        index,rho_index=np.array(index),rho_index
        if rho_index.dtype=='bool':rho_index=np.argwhere(rho_index)[:,0]
        assert rho_index.__len__()<=50,"Too many data points to plot (>50)"
        
        if fig is None:fig=plt.figure()
        ax=[fig.add_subplot(2*plot_sens+rho_index.size,1,k+2*plot_sens+1) for k in range(rho_index.size)]
        if plot_sens:
            ax.append(fig.add_subplot(2*plot_sens+rho_index.size,1,1))
            bbox=ax[-1].get_position()
            bbox.y0-=0.5*(bbox.y1-bbox.y0)
            ax[-1].set_position(bbox)
        # ax=[fig.add_subplot(rho_index.size+plot_sens,1,k+1) for k in range(rho_index.size+plot_sens)]
        
        if plot_sens:
            # ax=[*ax[1:],ax[0]] #Put the sensitivities last in the plotting
            hdl=self.sens.plot_rhoz(index=rho_index,ax=ax[-1])
            colors=[h.get_color() for h in hdl]
            for a in ax[1:-1]:a.sharex(ax[0])
        else:
            for a in ax[1:]:a.sharex(ax[0])
            colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        not_rho0=self.sens.rhoz[0,0]/self.sens.rhoz[0].max()<.98
        for k,a,color in zip(rho_index,ax,colors):
            hdl=plot_rho(self.label[index],self.R[index,k],self.Rstd[index,k]*errorbars if errorbars else None,\
                     style=style,color=color,ax=a,split=split)
            set_plot_attr(hdl,**kwargs)
            if not(a.is_last_row()):plt.setp(a.get_xticklabels(), visible=False)
            a.set_ylabel(r'$\rho_'+'{}'.format(k+not_rho0)+r'^{(\theta,S)}$')
        # fig.tight_layout()    
        return ax
    
    def plot_fit(self,index=None,exp_index=None,fig=None):
        assert self.src_data is not None and hasattr(self,'Rc') and self.Rc is not None,"Plotting a fit requires the source data(src_data) and Rc"
        info=self.src_data.info.copy()
        lbl,Rin,Rin_std=[getattr(self.src_data,k) for k in ['label','R','Rstd']]
        Rc=self.Rc
        if 'inclS2' in self.sens.opt_pars['options']:
            info.new_exper(Type='S2')
            Rin=np.concatenate((Rin,np.atleast_2d(1-self.src_data.S2).T),axis=1)
            Rc=np.concatenate((Rc,np.atleast_2d(1-self.S2c).T),axis=1)
            Rin_std=np.concatenate((Rin_std,np.atleast_2d(self.src_data.S2std).T),axis=1)
        
        return plot_fit(lbl=lbl,Rin=Rin,Rc=Rc,Rin_std=Rin_std,\
                    info=info,index=index,exp_index=exp_index,fig=fig)
        
            
            
    

 

    


