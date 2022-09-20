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
        self.source=None
        assert self.select._mdmode==True,'select._mdmode could not be set to True. Multi-atom selections are not allowed in PCA'
    
    def clear(self):
        """
        Clears out various stored parameters (use if the selection is changed)

        Returns
        -------
        None.

        """
        keys=['_pos','_covar','_sel1index','_sel2index','_lambda','_PC','_pcamp','_Ct','_data']
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
        AtomGroup
            MDAnalysis atom group
        """
        
        a0=self.atoms
        atoms=self.select.molsys.select_atoms(select_string)
        self.sel0=atoms
        if a0 is None or a0!=self.atoms:
            self.clear()
        return atoms            

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

        self.source=clsDict['Source'](Type='PCAmode',select=copy(self.select),filename=self.traj.files,
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
            mat0=(self.pos-self.pos.mean(0)).reshape([self.pos.shape[0],self.pos.shape[1]*self.pos.shape[2]])
            self._covar=mat0.T@mat0       
            # self._covar/=(self.pos[::step].shape[0]-1)
            self._covar/=(self.pos.shape[0]-1)
        return self._covar
    
    def runPCA(self,n:int=10):
        """
        Runs the PCA with n principle components. Set n to all to obtain all
        principle components

        Parameters
        ----------
        n : int (or 'all')
            Number of principle components to calculate.

        Returns
        -------
        None.

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
    
    @property    
    def Lambda(self):
        """
        Returns the eigenvalues (variance) corresponding to each of the 
        principle components

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
    
    def PC2pos(self,n:int=0,sigma:float=None,sel:int=None):
        """
        Calculates the motion on atoms resulting from the nth principle component
        deviating from the mean position by sigma standard deviations. One
        may which principle component to apply (n), how many standard deviations
        (sigma, defaults to np.sqrt(pca.Lambda(n))), and which atoms (either sel=
        None, which is all atoms in pca.atoms, or 1 (sel1) or 2 (sel2))

        Parameters
        ----------
        n : int, optional
            Which principle component to use. The default is 0.
        sigma : float, optional
            How many standard deviations (can be negative) to calculate. Default
            is equal to the standard deviation for that component (set to None).
        sel : int, optional
            Which group of atoms to use. 1 selects atoms in PCA.select.sel1,
            2 selects atoms in PCA.select.sel2, None takes atoms in PCA.atoms.
            The default is None.

        Returns
        -------
        None.

        """
        if sigma is None:sigma=np.sqrt(self.Lambda[n])
        i=np.arange(len(self.atoms)) if sel is None else (self.sel1index if sel==1 else self.sel2index)
        pos0=self.pos.mean(0)[i]
        
        pos0+=sigma*self.PC[:,n].reshape([self.pos.shape[1],3])
        return pos0
    
    def write_pdb(self,n:int=0,sigma:float=None,filename:str=None):
        """
        

        Parameters
        ----------
        n : int, optional
            Which principle component to use. The default is 0.
        sigma : float, optional
            How many standard deviations (can be negative) to calculate. Default
            is equal to the standard deviation for that component (set to None).
        filename : str, optional
            Location of the pdb. Defaults to pca.pdb in the project folder if
            it exists or in the same folder as the original topology.

        Returns
        -------
        None.

        """
        if filename is None:
            folder=self.project.directory if self.project is not None and self.project.directory is not None \
                else os.path.split(self.select.molsys.topo)[0]
            filename=os.path.join(folder,'pca.pdb')
        self.atoms.positions=self.PC2pos(n=n,sigma=sigma)
        self.atoms.write(filename)
        return filename
        
    def chimera(self,n:int=0,std=2):
        if not(hasattr(std,'__len__')):
            sigma=np.array([-std,std])*np.sqrt(self.Lambda[n])
        elif hasattr(std,'__len__') and len(std)==1:
            sigma=np.array([-std[0],std[0]])*np.sqrt(self.Lambda[n])
        else:
            sigma=np.array(std)*np.sqrt(self.Lambda[n])
        if self.project.chimera.current is None:
            self.project.chimera.current=0
        for sigma0 in sigma:
            filename=self.write_pdb(n=n,sigma=sigma0)
            self.project.chimera.command_line('open "{0}"'.format(filename))       
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=[int(c*100) for c in plt.get_cmap('tab10')(mdls-1)[:-1]]
            self.project.chimera.command_line(['~ribbon','show','color #{0} {1},{2},{3}'.format(mdls,clr[0],clr[1],clr[2])])
    
    @property
    def PCamp(self):
        """
        Projects the position onto the principle components

        Returns
        -------
        array.
            Time dependent amplitude of the principle components

        """
        if self._pcamp is None:
            mat0=(self.pos-self.pos.mean(0)).reshape([self.pos.shape[0],self.pos.shape[1]*self.pos.shape[2]])
            self._pcamp=(mat0@self.PC).T
        return self._pcamp
    
    def plot(self,n0:int=0,n1:int=1,ax=None,maxbin:float=None,nbins:int=None,**kwargs):
        if ax is None:ax=plt.figure().add_subplot(111)
        if maxbin is None:
            maxbin=np.max(np.abs(self.PCamp[min([n0,n1])])) 
        if nbins is None:
            nbins=min([100,self.pos.shape[0]//4])
        
        return ax.hist2d(self.PCamp[n0],self.PCamp[n1],bins=np.linspace(-maxbin,maxbin,nbins),**kwargs)
        
        
    @property
    def t(self):
        """
        Time axis returned in ns

        Returns
        -------
        np.ndarray
            Time axis in ns.

        """
        return np.arange(self.pos.shape[0])*self.traj.dt/1e3
        
    @property
    def Ct(self):
        """
        Calculates the linear correlation functions for each principle component.
        Correlation functions are normalized to start from 1, and decay towards
        zero.

        Returns
        -------
        np.ndarray
            nxnt array with each row corresponding to a different principle 
            component.

        """
        if self._Ct is None:
            ctc=Ctcalc()
            ctc.a=self.PCamp
            ctc.add()
            ct=ctc.Return()[0]
            ct=ct.T/self.Lambda
            self._Ct=ct.T
        return self._Ct
        
    def PCA2data(self):
        """
        Exports correlation functions for the principle components to a data
        object

        Returns
        -------
        None.

        """
        
        if self._data is None:
            out=Data_PCA(sens=clsDict['MD'](t=self.t))
            out.source=copy(self.source)
            out.source.details.append('PCA exported to data with {0} principle components'.format(len(self.Lambda)))
        
            out.R=np.array(self.Ct,dtype=dtype)
            out.Rstd=np.repeat(np.array([out.sens.info['stdev']],dtype=dtype),self.Ct.shape[0],axis=0)
            out.label=np.arange(out.R.shape[0],dtype=object)
            self._data=out
            self.project.append_data(out)
            
        return self._data
    
    def tot_z_dist(self,nd=8):
        """
        Takes the PCA and calculates ALL principle components

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
        Calculates the order parameters for the bond selections via principle
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
        
        