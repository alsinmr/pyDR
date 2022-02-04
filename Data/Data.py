#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:58:40 2022

@author: albertsmith
"""
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear as lsq
from pyDR.Defaults import Defaults
from pyDR.Sens import Detector
from pyDR.IO import write_file
from pyDR.Data.data_plots import plot_rho,plot_fit
from pyDR.misc.disp_tools import set_plot_attr

dtype=Defaults['dtype']

class Data():
    def __init__(self,R=None,Rstd=None,label=None,sens=None,select=None,src_data=None):
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
        if R is not None and sens is not None:
            assert R.shape[1]==sens.rhoz.shape[0],"Shape of sensitivity object is not consistent with shape of R"
        
        self.R=np.array(R) if R is not None else np.zeros([0,sens.rhoz.shape[0] if sens else 0],dtype=dtype)
        self.Rstd=np.array(Rstd) if Rstd is not None else np.zeros(self.R.shape,dtype=dtype)
        self.label=np.array(label) if label is not None else np.arange(self.R.shape[0],dtype=object)
        self.sens=sens
        self.detect=Detector(sens) if sens is not None else None
        self.src_data=src_data
        self.select=select #Stores the molecule selection for this data object
        
        
    
    def __setattr__(self, name, value):
        """Special controls for setting particular attributes.
        """
        if name=='sens' and value is not None and hasattr(self,'detect') and self.detect is not None:
            assert self.detect.sens==value,"Detector input sensitivities and data sensitivities should match"
            if self.detect.sens is not value:
                print("Warning: Detector object's input sensitivity is not the same object as the data sensitivity.")
                print("Changes to the data sensitivity object will not be reflected in the detector behavior")
        if name=='detect' and value is not None and hasattr(self,'sens') and self.sens is not None:
            assert self.sens==value.sens,"Detector input sensitivities and data sensitivities should match"
            if self.sens is not value.sens:
                print("Warning: Detector object's input sensitivity does is not the same object as the data sensitivity.")
                print("Changes to the data sensitivity object will not be reflected in the detector behavior")
        super().__setattr__(name, value)

        
    
    @property
    def n_data_pts(self):
        return self.R.shape[0]
    
    @property
    def ne(self):
        return self.R.shape[1]
    
    @property
    def info(self):
        return self.sens.info if self.sens is not None else None
    
    
    def fit(self,bounds=True,parallel=True):
        return fit(self,bounds=bounds,parallel=parallel)
    
    def save(self,filename,overwrite=False,save_src=True):
        if not(save_src):
            src=self.src_data
            self.src_data=None
        write_file(filename,self,overwrite)
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
        
            
            
        
#%% Functions for fitting the data
def fit(data,bounds=True,parallel=True):
    """
    Performs a detector analysis on the provided data object. Options are to
    include bounds on the detectors and whether to utilize parallelization to
    solve the detector analysis. 
    
    Note that instead of setting parallel=True, one can set it equal to an 
    integer to specify the number of cores to use for parallel processing
    (otherwise, defaults to the number of cores available on the computer)
    """
    detect=data.detect.copy()
    out=Data(sens=detect) #Create output data with sensitivity as input detectors
    out.label=data.label
    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.src_data=data
    out.select=data.select
    
    "Prep data for fitting"
    X=list()
    for k,(R,Rstd) in enumerate(zip(data.R,data.Rstd)):
        r0=detect[k] #Get the detector object for this bond
        UB=r0.rhoz.max(1)#Upper and lower bounds for fitting
        LB=r0.rhoz.min(1)
        R-=data.sens[k].R0 #Offsets if applying an effective sensitivity
        if 'inclS2' in r0.opt_pars['options']: #Append S2 if used for detectors
            R=np.concatenate((R,[data.S2[k]]))
            Rstd=np.concatenate((Rstd,[data.S2std[k]]))
        R/=Rstd     #Normalize data by its standard deviation
        r=(r0.r.T/Rstd).T   #Also normalize R matrix
        
        X.append((r,R,(LB,UB) if bounds else None,Rstd))
    
    "Perform fitting"
    if parallel:
        nc=parallel if isinstance(parallel,int) else mp.cpu_count()
        with mp.Pool(processes=nc) as pool:
            Y=pool.map(fit0,X)
    else:
        Y=[fit0(x) for x in X]
    
    "Extract data into output"
    out.R=np.zeros([len(Y),detect.r.shape[1]],dtype=dtype)
    out.R_std=np.zeros(out.R.shape,dtype=dtype)
    out.Rc=np.zeros([out.R.shape[0],detect.r.shape[0]],dtype=dtype)
    for k,y in enumerate(Y):
        out.R[k],out.R_std[k],Rc0=y
        out.R[k]+=detect[k].R0
        out.Rc[k]=Rc0*X[k][3]
        
    if 'inclS2' in detect.opt_pars['options']:
        out.S2c,out.Rc=out.Rc[:,-1],out.Rc[:,:-1]
    if 'R2ex' in detect.opt_pars['options']:
        out.R2,out.R=out.R[:,-1],out.R[:,:-1]
        out.R2std,out.Rstd=out.Rstd[:,-1],out.Rstd[:,:-1]
        
    return out
    
    

def fit0(X):
    """
    Used for parallel fitting of data. Single argument in the input should
    include data, the r matrix, and the upper and lower bounds
    """
    if X[2] is None:
        pinv=np.linalg.pinv(X[0])   #Simple pinv fit if no bounds required
        rho=pinv@X[1]
        Rc=X[0]@rho
        stdev=np.sqrt((pinv**2).sum())
    else:
        Y=lsq(X[0],X[1],bounds=X[2])
        rho=Y['x']
        Rc=Y['fun']+X[1]
        stdev=np.sqrt((np.linalg.pinv(X[0])**2).sum(1))
    return rho,stdev,Rc        