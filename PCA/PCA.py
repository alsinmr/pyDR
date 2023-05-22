#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:14:41 2022

@author: albertsmith
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh,svd
import os
import matplotlib.pyplot as plt
from .. import Project
from ..MDtools import Ctcalc
from .. import Defaults
from pyDR import clsDict
from copy import copy
from pyDR import Data #Note, this means PCA needs to get imported to pyDR after Data
from pyDR.Fitting.fit import model_free

dtype=Defaults['dtype']

class PCA():
    def __init__(self,select):
        self.select=select
        if select.project is None:select.project=Project()
        self.select._mdmode=True
        self.sel0=None
        self.clear()
        self._source=clsDict['Source']('PCAmode')
        assert self.select._mdmode==True,'select._mdmode could not be set to True. Multi-atom selections are not allowed in PCA'
    
    @property
    def source(self):
        return self._source
    
    def clear(self):
        """
        Clears out various stored parameters (use if the selection is changed)

        Returns
        -------
        None.

        """
        keys=['_pos','_covar','_sel1index','_sel2index','_lambda','_PC','_pcamp','_Ct','_data','_mean']
        for k in keys:setattr(self,k,None)
    
    def select_atoms(self,select_string:str):
        """
        Selects a group of atoms based on a selection string, using the 
        MDAnalysis select_atoms function.
        
        The resulting selection will replace the selection in PCA.select.sel1 and
        set PCA.select.sel2 to None. Not intended for use when back-calculating
        the correlation functions in NMR.

        Parameters
        ----------
        select_string : str
            String to filter the MDanalysis universe.

        Returns
        -------
        self
        
        """
        
        a0=self.atoms
        atoms=self.select.molsys.select_atoms(select_string)
        self.sel0=atoms
        if a0 is None or a0!=self.atoms:
            self.clear()
        return self          

    @property
    def project(self):
        return self.select.project
    
    @property
    def atoms(self):
        """
        Returns an atom group consisting of all unique atoms in self.sel1 and
        self.sel2

        Returns
        -------
        AtomGroup
            MDAnalysis atom group.

        """
        if self.select.sel1 is None and self.select.sel2 is None and self.sel0 is None:return None
        
        sel=self.select.uni.atoms[:0]
        if self.sel0 is not None:sel+=self.sel0
        if self.select.sel1 is not None:sel+=self.select.sel1
        
        if self.sel0 is not None:
            sel=self.sel0
            
        
        sel=self.sel0 if self.sel0 is not None else (self.select.sel1 if self.select.sel1 is not None else self.select.sel2)
        if self.select.sel1 is not None:sel+=self.select.sel1
        if self.select.sel2 is not None:sel+=self.select.sel2
        sel,i=np.unique(sel,return_inverse=True)
        if self.sel0 is None:
            if self.select.sel1 is None:
                self._sel1index=None
                self._sel2index=i
            else:
                self._sel1index=i[:len(self.select.sel1)]
                self._sel2index=None if self.select.sel2 is None else i[len(self.select.sel1):]
        else:
            if self.select.sel1 is None:
                self._sel1index=None
                self._sel2index=None if self.select.sel2 is None else i[len(self.sel0):]
            else:
                self._sel1index=i[len(self.sel0):len(self.sel0)+len(self.select.sel1)]
                self._sel2index=None if self.select.sel2 is None else i[len(self.sel0)+len(self.select.sel1):]
        return sum(sel)
    
    @property
    def sel1index(self):
        """
        Returns an index connecting the atoms in PCA.atoms to the atoms in 
        PCA.select.sel1 such that PCA.atoms[PCA._sel1index]=PCA.select.sel1
        

        Returns
        -------
        np.ndarray (dtype=int)
            1D array of integers

        """
        if self._sel1index is None:self.atoms
        return self._sel1index
    
    @property
    def sel2index(self):
        """
        Returns an index connecting the atoms in PCA.atoms to the atoms in 
        PCA.select.sel2 such that PCA.atoms[PCA._sel2index]=PCA.select.sel2
        

        Returns
        -------
        np.ndarray (dtype=int)
            1D array of integers

        """
        if self._sel2index is None:self.atoms
        return self._sel2index
    
        
    @property
    def traj(self):
        """
        Returns the trajecory object stored in self.select

        Returns
        -------
        Trajectory object.

        """
        return self.select.traj
    
    @property
    def pos(self):
        if self._pos is None:
            self.load()
        return self._pos
    
    def load(self,t0:int=None,tf:int=None,step=None):
        """
        Loads coordinates from the MD trajectory for all atoms in PCA.sel1 and
        PCA.sel2. By default uses t0,tf, and step that are stored in 
        PCA.traj, although these may also be set here.

        Returns
        -------
        None.

        """
        t={'t0':t0,'tf':tf,'step':step}
        if self._pos is not None:  #Determine if we need to run load
            run=False
            for k,v in t.items():
                if v is not None and getattr(self.traj,k)!=v:
                    run=True
                    break
            if not(run):return
            
        for k,v in t.items(): #Set time axis settings
            if v is not None:setattr(self.traj,k,v)
            
        atoms=self.atoms
        self.traj.ProgressBar=True
        pos=np.array([atoms.positions for _ in self.traj])

        self._source=clsDict['Source'](Type='PCAmode',select=copy(self.select),filename=self.traj.files,
                      status='raw')
        self.source.details.append('PCA analysis')
        self.source.details.append(self.select.details)
        self.source.project=self.project
        if self.sel0 is not None:
            self.source.details.append('PCA sel0 selection with {0} elements'.format(len(self.sel0)))
        
        self._pos=pos
        self.align()
        self._covar=None #Clears any existing covariance matrix
        
    def align(self):
        pos=self.pos.swapaxes(0,1)
        pos-=pos.mean(0)
        pos=pos.swapaxes(0,1)
        
        # ref=pos.mean(0)
        ref=pos[0]
        
        for k,pos0 in enumerate(pos):
            H=pos0.T@ref
            
            U,S,Vt=svd(H)
            V=Vt.T
            Ut=U.T
            
            # d=np.linalg.det(np.dot(V,Ut))
            R=V@Ut
            pos[k]=(R@pos0.T).T
        
        self._pos=pos
    
    @property
    def mean(self):
        if self._mean is None:
            self._mean=self.pos.mean(0)
        return copy(self._mean)
    
    @property    
    def CoVar(self):
        """
        Returns the covariance matrix for the PCA

        Returns
        -------
        np.array
            Covariance matrix, with dimensions (3N,3N) for N atoms

        """
        # self._covar=None
        if self._covar is None:
            # step=1
            # mat0=(self.pos[::step]-self.pos[::step].mean(0)).reshape([self.pos[::step].shape[0],self.pos.shape[1]*self.pos.shape[2]])
            mat0=(self.pos-self.mean).reshape([self.pos.shape[0],self.pos.shape[1]*self.pos.shape[2]])
            self._covar=mat0.T@mat0       
            # self._covar/=(self.pos[::step].shape[0]-1)
            self._covar/=(self.pos.shape[0]-1)
        return self._covar
    
    def runPCA(self,n:int=10):
        """
        Runs the PCA with n principal components. Set n to all to obtain all
        principal components

        Parameters
        ----------
        n : int (or 'all')
            Number of principal components to calculate.

        Returns
        -------
        self

        """
        if isinstance(n,str) and n.lower()=='all':
            n=self.CoVar.shape[0]
        
        if n>self.CoVar.shape[0]/2:
            w,v=eigh(self.CoVar)
        else:
            w,v=eigsh(self.CoVar,k=n,which='LM')
            
        i=np.argsort(w)[::-1]
        self._lambda,self._PC=w[i],v[:,i]
        self._Ct=None
        self._pcamp=None
        self._data=None
        return self
    
    @property    
    def Lambda(self):
        """
        Returns the eigenvalues (variance) corresponding to each of the 
        principal components

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # self._lambda=None
        if self._lambda is None:
            self.runPCA()
        return self._lambda
    
    @property
    def PC(self):
        if self._PC is None:
            self.runPCA()
        return self._PC
    
    def PC2pos(self,n:int=0,A:float=None,sel:int=None):
        """
        Calculates the motion on atoms resulting from the nth principal component
        deviating from the mean position by sigma standard deviations. One
        may which principal component to apply (n), how many standard deviations
        (sigma, defaults to np.sqrt(pca.Lambda(n))), and which atoms (either sel=
        None, which is all atoms in pca.atoms, or 1 (sel1) or 2 (sel2))

        Parameters
        ----------
        n : int, optional
            Which principal component to use. The default is 0.
        A : float, optional
            Amplitude of the principal component to calculate. If set to None,
            then we use one standard deviation (np.sqrt(self.Lambda[n]))
        sel : int, optional
            Which group of atoms to use. 1 selects atoms in PCA.select.sel1,
            2 selects atoms in PCA.select.sel2, None takes atoms in PCA.atoms.
            The default is None.

        Returns
        -------
        None.

        """
        if A is None:A=np.sqrt(self.Lambda[n])
        i=np.arange(len(self.atoms)) if sel is None else (self.sel1index if sel==1 else self.sel2index)
        pos0=self.mean[i]
        
        pos0+=A*self.PC[:,n].reshape([self.PC.shape[0]//3,3])
        return pos0
    
    def write_pdb(self,n:int=0,A:float=None,PCamp:list=None,filename:str=None):
        """
        

        Parameters
        ----------
        n : int, optional
            Which principal component to use. The default is 0.
        A : float, optional
            Amplitude of the principal component to plot 
        filename : str, optional
            Location of the pdb. Defaults to pca.pdb in the project folder if
            it exists or in the same folder as the original topology.
        PCamp : list-like, optional
            List of amplitudes for the first len(PCamp) principal components. 
            If this is provided, then 'n' and 'std' are ignored.
            The default is None.

        Returns
        -------
        None.

        """
        if filename is None:
            folder=self.project.directory if self.project is not None and self.project.directory is not None \
                else os.path.split(self.select.molsys.topo)[0]
            filename=os.path.join(folder,'pca.pdb')
        if PCamp is not None:
            pos=self.mean
            for k,A in enumerate(PCamp):
                pos+=A*self.PC[:,k].reshape([self.PC.shape[0]//3,3])
            self.atoms.positions=pos
        else:
            self.atoms.positions=self.PC2pos(n=n,A=A)
        self.atoms.write(filename)
        return filename
        
    def chimera(self,n:int=0,std:float=1,PCamp:list=None):
        """
        Plots the change in a structure for a given principal component. That is,
        if n=0, then we plot the mean structure with +/-sigma for the n=0 
        principal component to see how that component changes.
        
        Alternatively, one may provide a list of amplitudes in PCamp, where
        these correspond to the first len(PCamp) principal components to 
        determine the appearance of the molecule for a given position in the
        PCA histograms. If PCamp is provided, then 'n' and 'std' are ignored.
        

        Parameters
        ----------
        n : int, optional
            principal component to plot. The default is 0.
        std : float, optional
            Number of standard deviations away from the mean to plot. 
            The default is 1.
        PCamp : list, optional
            List of amplitudes for the first len(PCamp) principal components. 
            If this is provided, then 'n' and 'std' are ignored.
            The default is None.

        Returns
        -------
        None.

        """            
        if PCamp is not None:
            filename=self.write_pdb(n=n,PCamp=PCamp)
            if self.project.chimera.current is None:self.project.chimera.current=0
            self.project.chimera.command_line('open "{0}"'.format(filename))
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=[int(c*100) for c in plt.get_cmap('tab10')(mdls-1)[:-1]]
            self.project.chimera.command_line(['~ribbon','show','color #{0} {1},{2},{3}'.format(mdls,clr[0],clr[1],clr[2])])
            self.project.chimera.command_line(self.project.chimera.saved_commands)
            return
        
            
        if not(hasattr(std,'__len__')):
            A=np.array([-std,std])*np.sqrt(self.Lambda[n])
        elif hasattr(std,'__len__') and len(std)==1:
            A=np.array([-std[0],std[0]])*np.sqrt(self.Lambda[n])
        else:
            A=np.array(std)*np.sqrt(self.Lambda[n])
        if self.project.chimera.current is None:
            self.project.chimera.current=0
        for A0 in A:
            filename=self.write_pdb(n=n,A=A0,PCamp=PCamp)
            if self.project.chimera.current is None:self.project.chimera.current=0
            self.project.chimera.command_line('open "{0}"'.format(filename))       
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=[int(c*100) for c in plt.get_cmap('tab10')(mdls-1)[:-1]]
            self.project.chimera.command_line(['~ribbon','show','color #{0} {1},{2},{3}'.format(mdls,clr[0],clr[1],clr[2])])
        self.project.chimera.command_line(self.project.chimera.saved_commands)
    @property
    def PCamp(self):
        """
        Projects the position onto the principal components

        Returns
        -------
        array.
            Time dependent amplitude of the principal components

        """
        if self._pcamp is None:
            mat0=(self.pos-self.mean).reshape([self.pos.shape[0],self.pos.shape[1]*self.pos.shape[2]])
            self._pcamp=(mat0@self.PC).T
        return self._pcamp
    
    def plot(self,n0:int=0,n1:int=1,ax=None,maxbin:float=None,nbins:int=None,**kwargs):
        """
        Creates a 2D histogram of two principal components. Specify the desired
        components (n0,n1=0,1 by default)

        Parameters
        ----------
        n0 : int, optional
            1st PC to be plotted. The default is 0.
        n1 : int, optional
            2nd PC to be plotted. The default is 1.
        ax : TYPE, optional
            Axis object to plot into. The default is None.
        maxbin : float, optional
            Largest bin. The default is None.
        nbins : int, optional
            Number of bins. The default is None.
        **kwargs : TYPE
            Plotting arguments to be passed to hist2d.

        Returns
        -------
        TYPE
            Handle for the hist2d object

        """
        if ax is None:ax=plt.figure().add_subplot(111)
        if maxbin is None:
            maxbin=np.max(np.abs(self.PCamp[min([n0,n1])])) 
        if nbins is None:
            nbins=min([100,self.PCamp.shape[1]//4])
        
        out=ax.hist2d(self.PCamp[n0],self.PCamp[n1],bins=np.linspace(-maxbin,maxbin,nbins),**kwargs)
        ax.set_xlabel(f'PC {n0}')
        ax.set_ylabel(f'PC {n1}')
        return out
        
    
    def hist2struct(self,nmax:int=4,ref_struct:bool=True,**kwargs):
        """
        Interactively view structures corresponding to positions on the 
        histogram plots. Specify the maximum principle component to display. Then,
        click on the plots until all principle components are specified. The
        corresponding structure will then be displayed in chimeraX. Subsequent
        clicks will add new structures to chimeraX.

        Parameters
        ----------
        nmax : int, optional
            Maximum principal component to show. The default is 6.
        ref_struct : bool, optional
            Show a reference structure in chimera (mean structure)
        **kwargs : TYPE
            Keyword arguments to be passed to the PCA.plot function.

        Returns
        -------
        None.

        """
        x,y=int(np.ceil(np.sqrt(nmax))),int(np.ceil(np.sqrt(nmax)))
        if (x-1)*y>=nmax:x-=1
        fig=plt.figure()
        ax=[fig.add_subplot(x,y,k+1) for k in range(nmax)]
        hdls=list()
        for k,a in enumerate(ax):
            self.plot(n0=k,n1=k+1,ax=a,**kwargs)
            hdls.append([a.plot([0,0],[0,0],color='black',linestyle=':',visible=False)[0] for _ in range(2)])
        
        fig.tight_layout()    
        
        if ref_struct:
            self.chimera(PCamp=[0])
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=plt.get_cmap('tab10')(mdls-1)
            for k,a in enumerate(ax):
                a.scatter(0,0,marker='x',color=clr)
        
        PCamp=[None for _ in range(nmax+1)]
        markers=['x','o','+','v','>','s','1','*']
        def onclick(event):
            if event.inaxes:
                ax0=event.inaxes
                i=ax.index(ax0)
                PCamp[i]=event.xdata
                PCamp[i+1]=event.ydata
                hdls[i][0].set_xdata([event.xdata,event.xdata])
                hdls[i][0].set_ydata(ax0.get_ylim())
                hdls[i][0].set_visible(True)
                hdls[i][1].set_xdata(ax0.get_xlim())
                hdls[i][1].set_ydata([event.ydata,event.ydata])
                hdls[i][1].set_visible(True)
                if i+1<len(hdls):
                    ax0=ax[i+1]
                    hdls[i+1][0].set_xdata([event.ydata,event.ydata])
                    hdls[i+1][0].set_ydata(ax0.get_ylim())
                    hdls[i+1][0].set_visible(True)
                if i>0:
                    ax0=ax[i-1]
                    hdls[i-1][1].set_xdata(ax0.get_ylim())
                    hdls[i-1][1].set_ydata([event.xdata,event.xdata])
                    hdls[i-1][1].set_visible(True)
                
                if not(None in PCamp):  #All positions defined. Add new molecule in chimera
                    self.chimera(PCamp=PCamp)
                    for k,a in enumerate(ax):
                        mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
                        clr=plt.get_cmap('tab10')((mdls-1)%10)
                        a.scatter(PCamp[k],PCamp[k+1],100,marker=markers[(mdls-1)%len(markers)],linewidth=3,color=clr)
                        
                    #Clear the positions in the plot
                    for k in range(len(PCamp)):PCamp[k]=None
                    for h in hdls:
                        for h0 in h:
                            h0.set_visible(False)
                plt.pause(0.01)
            else: #Clicking outside the axes clears out the positions
                for k in range(len(PCamp)):PCamp[k]=None
                for h in hdls:
                    for h0 in h:
                        h0.set_visible(False)
                plt.pause(0.01)
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        self.h2s_ax=ax
        return ax
    
    def load_points(self,pts):
        """
        After running hist2struct, one may load points in from an array instead
        of interactive selective (e.g. for reproducibility of structures)

        Parameters
        ----------
        pts : array
            Should have dimensions of (nmax+1) X npoints.

        Returns
        -------
        None.

        """
        assert hasattr(self,'h2s_ax'),'One must first run hist2struct before running load_points'
        ax=self.h2s_ax
        nmax=len(ax)
        
        pts=np.array(pts).T
        
        cmap=plt.get_cmap('tab10')
        markers=['x','o','+','v','>','s','1','*']
        for pt in pts:
            self.chimera(PCamp=pt)
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)-1
            for a,n0,n1 in zip(ax,range(nmax),range(1,nmax+1)):
                a.scatter(pt[n0],pt[n1],100,marker=markers[mdls%len(markers)],linewidth=3,color=cmap(mdls%10))
                
    def get_points(self):
        """
        After running hist2struct, return the points set interactively in the 
        various plots.

        Returns
        -------
        np.array

        """
        assert hasattr(self,'h2s_ax'),'One must first run hist2struct before running load_points'
        ax=self.h2s_ax
        nmax=len(ax)
        
        pts=list()
        for n in range(nmax):
            pts.append([h.get_offsets().data[0,0] for h in ax[n].collections[1:]])
        pts.append([h.get_offsets().data[0,1] for h in ax[-1].collections[1:]])
            
        return np.array(pts)
        
    
    @property
    def t(self):
        """
        Time axis returned in ns

        Returns
        -------
        np.ndarray
            Time axis in ns.

        """
        if hasattr(self,'_t') and self._t is not None:return self._t
        return np.arange(self.pos.shape[0])*self.traj.dt/1e3
        
    def Ct(self,t0:int=0,tf:int=None):
        """
        Calculates the linear correlation functions for each principal component.
        Correlation functions are normalized to start from 1, and decay towards
        zero.

        Returns
        -------
        np.ndarray
            nxnt array with each row corresponding to a different principal 
            component.

        """
        assert t0<self.PCamp.shape[1],f"t0 must be less than the trajectory length ({self.PCamp.shape[1]})"
        tf=min(tf,self.PCamp.shape[1])
        t0=t0%self.PCamp.shape[1]
        tf=self.PCamp.shape[1] if tf is None else (((tf-1)%self.PCamp.shape[1])+1 if tf else 0)
        if self._Ct is None or t0!=self._Ct[0] or tf!=self._Ct[1]:
            ctc=Ctcalc()
            ctc.a=self.PCamp[:,t0:tf]
            ctc.add()
            ct=ctc.Return()[0]
            ct=ct.T/self.Lambda
            self._Ct=t0,tf,ct.T
        return self._Ct[-1]
        
    def PCA2data(self,norm:bool=True,t0:int=0,tf:int=None):
        """
        Exports correlation functions for the principal components to a data
        object
        
        Parameters
        ----------
        norm : bool
            Normalize correlation functions to have an initial value of 1.
            Otherwise, correlation functions will have an initial value of
            sqrt(Lambda). Default is True

        Returns
        -------
        None.

        """
        t0=t0%self.PCamp.shape[1]
        tf=self.PCamp.shape[1] if tf is None else (((tf-1)%self.PCamp.shape[1])+1 if tf else 0)
        
        if self._data is None or t0!=self._data[0] or tf!=self._data[1]:
            sens=clsDict['MD'](t=self.t[t0:tf]-self.t[t0])
            out=Data_PCA(R=self.Ct(t0,tf).astype(dtype) if norm else self.Ct(t0,tf).astype(dtype).T*np.sqrt(self.Lambda).T,
                         Rstd=np.repeat(np.array([sens.info['stdev']],dtype=dtype),self.Ct(t0,tf).shape[0],axis=0),
                         sens=sens,
                         select=self.select,Type='PCAmode')
            out.source.filename=self.select.traj.files
            out.source.status='raw'
            out.source.details=self.select.details
            out.source.details.append(f'PCA based on analysis of {self.atoms.__len__()}')
            out.source.details.append(f'PCA exported to data with {len(self.Lambda)} principal components')
        
            out.R=np.array(self.Ct(t0,tf),dtype=dtype)
            out.Rstd=np.repeat(np.array([out.sens.info['stdev']],dtype=dtype),self.Ct(t0,tf).shape[0],axis=0)
            out.label=np.arange(out.R.shape[0],dtype=int)
            self._data=t0,tf,out
            self.project.append_data(out)
            
        return copy(self._data[2])
    
    def tot_z_dist(self,nd=8):
        """
        Takes the PCA and calculates ALL principal components

        Returns
        -------
        None.

        """
        if self._lambda is None or self._lambda.shape[0]<self.pos.shape[1]*3:
            self.runPCA(n='all')
            
        data=copy(self.PCA2data())
        # data.del_exp(0)
        data.detect.r_no_opt(n=nd)
        fit0=data.fit(bounds=False)
        fit0.detect.r_auto(n=nd)
        fit=fit0.fit()
        z,A,_,fit1=model_free(fit,nz=2,fixz=[-14,None])
        
        self._MFfit={'z':z,'A':A,'data':fit,'fit':fit1}    
        
        
    @property
    def S2direct(self):
        """
        Calculates the order parameters for the bond selections using the usual
        approach.

        Returns
        -------
        np.array
            Array of order parameters

        """
        if self.select.sel1 is None or self.select.sel2 is None:return None
        
        v=(self.pos[:,self.sel1index]-self.pos[:,self.sel2index]).T
        v/=np.sqrt((v**2).sum(0))      
        
        S2=np.ones(v.shape[1])*(-1/2)
        for k in range(3):
            for j in range(k,3):
                S2+=(v[k]*v[j]).mean(-1)**2*(3/2 if k==j else 3)
        return S2
    
    def S2pca(self,n='all'):
        """
        Calculates the order parameters for the bond selections via principal
        component analysis

        Returns
        -------
        None.

        """
        if n=='all':n=self.pos.shape[1]*3
        
        if self._PC is None or self._PC.shape[1]<n:self.runPCA(n)
        
        # v0=(self.pos[:,self.sel1index]-self.pos[:,self.sel2index]).mean(0)
        # len2=(v0**2).sum(-1)
        v0=self.pos[:,self.sel1index]-self.pos[:,self.sel2index]
        len2=np.sqrt((v0**2).sum(-1)).mean(0)**2
        
        
        
        Delta=[self.PC[k::3][self.sel1index]-self.PC[k::3][self.sel2index] for k in range(3)]
        
        S2=np.ones(len2.shape)*(-1/2)
        for k in range(3):
            for j in range(k,3):
                S2+=(((Delta[k]*Delta[j])*self.Lambda).T/len2).sum(0)**2*(3/2 if k==j else 3)
                
        return S2
    
    def S2covar(self):
        """
        Calculates S2 from the covariance matrix.

        Returns
        -------
        np.array
            Array of order parameters

        """
        
        v0=self.pos[:,self.sel1index]-self.pos[:,self.sel2index]
        len2=np.sqrt((v0**2).sum(-1)).mean(0)**2
        
        S2=np.ones(len2.shape)*(-1/2)
        M=self.CoVar
        
        for k in range(3):
            for j in range(k,3):
                S2+=M[k::3,j::3][self.sel1index][:,self.sel2index] #Not correct yet
                """
                We need cov(k_1,j_1)+cov(k_2,j_2)-cov(k_1,j_2)-cov(k_2,j_1)+
                            +(<k_1>-<k_2>)*(<j_1>-<j_2>)
                            
                We should also work out the impact of the length normalization,
                which may not be so simple.
                """
        
        
        
            
class Data_PCA(Data):
    @property
    def select(self):
        """
        Returns the selection if iRED in bond mode

        Returns
        -------
        molselect
            Selection object.

        """
        if 'mode' in self.source.Type:
            return None
        return self.source.select
        
        