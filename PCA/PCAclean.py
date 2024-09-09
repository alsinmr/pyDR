#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:08:30 2024

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
from .. import clsDict
from copy import copy,deepcopy
from ..misc.tools import linear_ex
from .PCAmovies import PCAmovies
from .PCAsubs import PCA_Ct,PCA_S2,PCAvecs,PCA2Data,Weighting,Impulse,Hist,Cluster



dtype=Defaults['dtype']

class PCA():
    def __init__(self,select,align_ref='name CA',project=None):
        self.select=select
        self.project=self.select.project if project is None else project
        self.select._mdmode=True
        self._sel0=None
        self._atoms=None
        self._sel0index=None
        self._sel1index=None
        self._sel2index=None
        self.align_ref=align_ref
        self.clear()
        self._source=clsDict['Source']('PCAmode')
        self._n=None
        self._ref_pos=None
        
        self._select_bond=self.select.select_bond
        
        self.clear()
        
        assert self.select._mdmode==True,'select._mdmode could not be set to True. Multi-atom selections are not allowed in PCA'
    
    #%% Sub-classes
    @property
    def Movie(self):
        if self._Movie is None:
            self._Movie=PCAmovies(self)
        return self._Movie
    
    @property
    def Ct(self):
        if self._Ct is None:
            self._Ct=PCA_Ct(self)
        return self._Ct
    
    @property
    def S2(self):
        if self._S2 is None:
            self._S2=PCA_S2(self)
        return self._S2
    
    @property
    def Vecs(self):
        if self._Vecs is None:
            self._Vecs=PCAvecs(self)
        return self._Vecs
    
    @property
    def Data(self):
        if self._Data is None:
            self._Data=PCA2Data(self)
        return self._Data
    
    @property
    def Weighting(self):
        if self._Weighting is None:
            self._Weighting=Weighting(self)
        return self._Weighting
    
    @property
    def Impulse(self):
        if self._Impulse is None:
            self._Impulse=Impulse(self)
        return self._Impulse
    
    @property
    def Hist(self):
        if self._Hist is None:
            self._Hist=Hist(self)
        return self._Hist
    
    @property
    def Cluster(self):
        if self._Cluster is None:
            self._Cluster=Cluster(self)
        return self._Cluster
    
    #%% Misc.
    def clear(self):
        """
        Clears out various stored parameters (use if the selection is changed)

        Returns
        -------
        None.

        """
        keys=['_pos','_covar','_sel1index','_sel2index','_lambda','_PC','_pcamp','_mean',
              '_S2','_Ct','_Vecs','_Data','_Movie','_Weighting','_Impulse','_Hist','_Cluster']
        for k in keys:setattr(self,k,None)
        return self
    
    @property
    def source(self):
        return self._source
    
    #%% Selection
    def select_atoms(self,select_string:str):
        """
        Selects a group of atoms based on a selection string, using the 
        MDAnalysis select_atoms function.
        
        The resulting selection will be stored in PCA.sel0
        
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
   
    
    def select_bond(self,Nuc,resids=None,segids=None,filter_str:str=None,label=None):
        """
        Select a bond according to 'Nuc' keywords
            '15N','N','N15': select the H-N in the protein backbone
            'CO','13CO','CO13': select the CO in the protein backbone
            'CA','13CA','CA13': select the CA in the protein backbone
            'ivl','ivla','ch3': Select methyl groups 
                (ivl: ILE,VAL,LEU, ivla: ILE,LEU,VAL,ALA, ch3: all methyl groups)
                (e.g. ivl1: only take one bond per methyl group)
                (e.g. ivlr,ivll: Only take the left or right methyl group)
            'sidechain' : Selects a vector representative of sidechain motion
        
        Note that it is possible to provide a list of keywords for Nuc in order
        to simultaneously evaluate multiple bond types. In this case, sorting
        will be such that bonds from the same residues/segments are grouped 
        together.

        Parameters
        ----------
        Nuc : str or list, optional
            Nucleus keyword or list of keywords. 
        resids : list/array/single element, optional
            Restrict selected residues. The default is None.
        segids : list/array/single element, optional
            Restrict selected segments. The default is None.
        filter_str : str, optional
            Restricts selection to atoms selected by the provided string. String
            is applied to the MDAnalysis select_atoms function. The default is None.
        label : list/array, optional
            Manually provide a label for the selected bonds. The default is None.

        Returns
        -------
        self

        """
        
        self._atoms=None
        self._select_bond(Nuc=Nuc,resids=resids,segids=segids,filter_str=filter_str,label=label)
        return self
    
    @property
    def uni(self):
        return self.select.uni
    
    @property
    def sel0(self):
        if self._sel0 is None:return self.uni.atoms[:0]
        return self._sel0    
    @sel0.setter
    def sel0(self,sel0):
        if isinstance(sel0,str):
            sel0=self.uni.select_atoms(sel0)
        assert isinstance(sel0,sel0.atoms.__class__),"sel0 must be a selection string or an atom group (MDAnalysis)"
        self._atoms=None
        self._sel0=sel0
    
    
    @property
    def sel1(self):
        if self.select.sel1 is None:return self.uni.atoms[:0]
        return self.select.sel1
    @sel1.setter
    def sel1(self,sel1):
        self._atoms=None
        self.select.sel1=sel1
        
    @property
    def sel2(self):
        if self.select.sel2 is None:return self.uni.atoms[:0]
        return self.select.sel2
    @sel2.setter
    def sel2(self,sel2):
        self._atoms=None
        self.select.sel2=sel2
        
    @property
    def nbonds(self):
        return len(self.sel1)

    @property
    def project(self):
        if self._project is None:
            self._project=clsDict['Project']()
        return self._project
    
    @project.setter
    def project(self,project):
        self._project=project
    
    @property
    def atoms(self):
        """
        Returns an atom group consisting of all unique atoms in self.sel1,
        self.sel2, and self.sel0

        Returns
        -------
        AtomGroup
            MDAnalysis atom group.

        """
        
        if np.any([getattr(self,name) is None for name in ['_atoms','_sel0index','_sel1index','_sel2index']]):
            sel=self.sel0+self.sel1+self.sel2
            if len(sel)==0:
                self._atoms=self.sel0
                self._sel0index=np.zeros(0,dtype=int)
                self._sel1index=np.zeros(0,dtype=int)
                self._sel2index=np.zeros(0,dtype=int)
                return self._atoms
            
            sel,i=np.unique(sel,return_inverse=True)
            self._atoms=sel.sum()
            self._sel0index=i[:len(self.sel0)]
            self._sel1index=i[len(self.sel0):len(self.sel0)+self.nbonds]
            self._sel2index=i[len(self.sel0)+self.nbonds:]
        
        return self._atoms
    
    @property
    def sel0index(self):
        """
        Returns an index connecting the atoms in PCA.atoms to the atoms in 
        PCA.sel0 such that PCA.atoms[PCA.sel0index]=PCA.sel0
        

        Returns
        -------
        np.ndarray (dtype=int)
            1D array of integers

        """
        if self._sel0index is None:self.atoms
        return self._sel0index
    
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
    
    #%% Trajectory        
    @property
    def traj(self):
        """
        Returns the trajecory object stored in self.select

        Returns
        -------
        Trajectory object.

        """
        return self.select.traj
    
    #Loading/positions    
    @property
    def t(self):
        return np.arange(len(self.traj))*self.traj.dt*1e-3
    
    @property
    def pos(self):
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
        
        pos=[]
        ref=self.uni.select_atoms(self.align_ref)
        # self.traj[0]
        # ref0=ref.positions
        # ref0-=ref0.mean(0)
        ref0=self.ref_pos
        
        for _ in self.traj:
            pos.append(self.align(ref0,ref,atoms))
        
        # pos=np.array([atoms.positions for _ in self.traj])

        self._source=clsDict['Source'](Type='PCAmode',select=copy(self.select),filename=self.traj.files,
                      status='raw')
        self.source.details.append('PCA analysis')
        self.source.details.append(self.select.details)
        self.source.project=self.project
        if self.sel0 is not None:
            self.source.details.append('PCA sel0 selection with {0} elements'.format(len(self.sel0)))
        
        self._pos=np.array(pos)
        # self.align(ref)
        self._covar=None #Clears any existing covariance matrix
        
    @property
    def ref_pos(self):
        """
        Store positions of the reference atoms

        Returns
        -------
        None.

        """
        if self._ref_pos is None:
            ref=self.uni.select_atoms(self.align_ref)
            self.traj[0]
            self._ref_pos=ref.positions
            self._ref_pos-=self._ref_pos.mean(0)
        return self._ref_pos
        
    @staticmethod
    def align(ref0,ref,atoms):
        ref=ref.positions
        pos=atoms.positions
        
        pos-=ref.mean(0)
        ref-=ref.mean(0)
        
        H=ref0.T@ref
        U,S,Vt=svd(H)
        V=Vt.T
        Ut=U.T
        
        R=V@Ut
        return (R.T@pos.T).T
        
        
    # def align(self):
    #     ref_group=self.atoms.select_atoms(self.align_ref)
    #     if len(ref_group):
    #         i=np.digitize(ref_group.indices,self.atoms.indices)-1
    #     else:
    #         i=np.arange(len(self.atoms))
        
    #     pos=self.pos.swapaxes(0,1)
    #     pos-=pos[i].mean(0)
    #     pos=pos.swapaxes(0,1)
        
    #     # ref=pos.mean(0)
        
    #     pos_ref=pos[:,i]
    #     ref=pos_ref[0]
        
    #     for k,pos0 in enumerate(pos_ref):
    #         H=pos0.T@ref
            
    #         U,S,Vt=svd(H)
    #         V=Vt.T
    #         Ut=U.T
            
    #         # d=np.linalg.det(np.dot(V,Ut))
    #         R=V@Ut
    #         pos[k]=(R@pos[k].T).T
        
    #     self._pos=pos
    
    
    
    @property
    def mean(self):
        if self._mean is None:
            self._mean=self.pos.mean(0)
        return copy(self._mean)
    
    #%% Calculating the PCA
    
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
    

    def runPCA(self,n:int=-1):
        """
        Runs the PCA with n principal components. Set n to -1 to obtain all
        principal components

        Parameters
        ----------
        n : int (or '-1')
            Number of principal components to calculate. Default is -1

        Returns
        -------
        self

        """
        if n==-1:n=self.CoVar.shape[0]
        
        if n>self.CoVar.shape[0]/2:
            w,v=eigh(self.CoVar)
        else:
            w,v=eigsh(self.CoVar,k=n,which='LM')
            
        i=np.argsort(w)[::-1]
        self._lambda,self._PC=w[i],v[:,i]
        
        keys=['_Ct','_data','_S2m','_Ctmb','_tcavg','_pcamp','_fits']
        for k in keys:setattr(self,k,None)
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
        return self._lambda[:self.nPC]
    
    @property
    def PC(self):
        if self._PC is None:
            self.runPCA()
        return self._PC[:,:self.nPC]
    
    @property
    def PCxyz(self):
        return self.PC.reshape([len(self.atoms),3,self.nPC]).swapaxes(0,1)
    
    @property
    def nPC(self) -> int:
        """
        Number of principal components that are kept after initial PCA analysis.
        
        We always want to exclude the smallest 6 principal components, because
        these represent the translational and rotational degrees of freedom.
        
        They are small because we have previously aligned the molecule, thus
        removing these motions. However, if we do not calculate all principal
        components, we may already have removed them.

        Returns
        -------
        int

        """
        if self._PC is None:
            self.runPCA()
        return min(self._PC.shape[1],self._PC.shape[0]-6)
    
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

    