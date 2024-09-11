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
from copy import copy,deepcopy
from pyDR.misc.tools import linear_ex
from .PCAmovies import PCAmovies


dtype=Defaults['dtype']

class PCA():
    def __init__(self,select,align_ref='name CA'):
        self.select=select
        self._project=None
        if select.project is None:
            select.molsys.project=Project()
            select.project=select.molsys.project
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
        
        self._select_bond=self.select.select_bond
        
        self._movie=None
        
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
        keys=['_pos','_covar','_sel1index','_sel2index','_lambda','_PC','_atoms',
              '_pcamp','_Ct','_data','_noopt','_fit','_directfit','_prodfit','_mean','_S2m','_S20m','_Am','_Ctdirect','_tcavg','_Ctprod','_mf']
        for k in keys:setattr(self,k,None)
        return self
    
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
        self.clear()
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
        self.clear()
        self._sel0=sel0
    
    
    @property
    def sel1(self):
        if self.select.sel1 is None:return self.uni.atoms[:0]
        return self.select.sel1
    @sel1.setter
    def sel1(self,sel1):
        self.clear()
        self.select.sel1=sel1
        
    @property
    def sel2(self):
        if self.select.sel2 is None:return self.uni.atoms[:0]
        return self.select.sel2
    @sel2.setter
    def sel2(self,sel2):
        self.clear()
        self.select.sel2=sel2
        
    @property
    def nbonds(self):
        return len(self.sel1)

    @property
    def project(self):
        return self.select.project if self._project is None else self._project
    
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
        
        if self._atoms is None:
            self.clear()
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
        ref_group=self.atoms.select_atoms(self.align_ref)
        if len(ref_group):
            i=np.digitize(ref_group.indices,self.atoms.indices)-1
        else:
            i=np.arange(len(self.atoms))
        
        pos=self.pos.swapaxes(0,1)
        pos-=pos[i].mean(0)
        pos=pos.swapaxes(0,1)
        
        # ref=pos.mean(0)
        
        pos_ref=pos[:,i]
        ref=pos_ref[0]
        
        for k,pos0 in enumerate(pos_ref):
            H=pos0.T@ref
            
            U,S,Vt=svd(H)
            V=Vt.T
            Ut=U.T
            
            # d=np.linalg.det(np.dot(V,Ut))
            R=V@Ut
            pos[k]=(R@pos[k].T).T
        
        self._pos=pos
    
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
        
        i=np.arange(len(self.atoms)) if sel is None else getattr(self,f'sel{sel}index')
        
        pos0=self.mean[i]
        
        pos0+=A*self.PC[:,n].reshape([self.PC.shape[0]//3,3])
        return pos0
    
    #%% Correlation functions of the principal components
    @property
    def t(self):
        """
        Time axis returned in ns

        Returns
        -------
        np.ndarray
            Time axis in ns.

        """
        return np.arange(len(self.select.traj))*self.select.traj.dt*1e-3
    
    @property    
    def Ct(self):
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
        if self._Ct is None:
            ctc=Ctcalc()
            ctc.a=self.PCamp
            ctc.add()
            ct=ctc.Return()[0]
            ct=ct.T/self.Lambda
            self._Ct=ct.T
        return self._Ct[:self.nPC]
    
    @property
    def tc_avg(self):
        """
        Returns the average correlation time for each principal component

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self._tcavg is None:
            i=np.argmax(self.Ct<0,axis=1)
            self._tcavg=np.array([(Ct[:i0]).sum() for Ct,i0 in zip(self.Ct,i)])
        return self._tcavg
    
    @property
    def tc_index(self):
        """
        Index that sorts the principal components from shortest correlation time
        to longest correlation time

        Returns
        -------
        None.

        """
        return np.argsort(self.tc_avg)
    
    @property
    def tc_rev_index(self):
        """
        Returns the index to reverse tc sorting

        Returns
        -------
        None.

        """
        return np.unique(self.tc_avg,return_inverse=True)[1]
        
    
    #%% Vector and Distance functions
    @property
    def v(self):
        """
        Calculates vectors connecting atoms in sel1 and sel2 for all loaded 
        time points

        Returns
        -------
        3D array (n time points x n atoms x 3)

        """
        self.load()
        return self.pos[:,self.sel1index,:]-self.pos[:,self.sel2index,:]
    
    @property
    def dist(self):
        """
        Returns the distance between atom pairs for all loaded time points

        Returns
        -------
        3D array (n time points x n atoms)

        """
        return np.sqrt((self.v**2).sum(-1))
        
    @property
    def v_mean_pos(self):
        """
        Returns vectors between the mean positions.
        
        These should usually be shorter than the mean of the vectors.
        
        Calculation:
            1) Calculate mean positions
            2) Calculate vectors

        Returns
        -------
        2D array (n atoms x 3)

        """
        return self.mean[self.sel1index]-self.mean[self.sel2index]
    
    @property
    def dist_mean_pos(self):
        """
        Returns distances between atoms pairs for vectors between average 
        positions

        Returns
        -------
        1D array (n atoms)

        """
        return np.sqrt((self.v_mean_pos**2).sum(-1))
    
    @property
    def dist_avg(self):
        """
        Returns the average distance between atoms pairs

        Returns
        -------
        None.

        """
        return np.sqrt((self.v**2).sum(-1)).mean(0)
    
    @property
    def d2_0m(self):
        """
        Returns vectors between sel1 and sel2 that have been calculated for
        all principal components from 0 to m

        Returns
        -------
        None.

        """
        d2=self.dist_mean_pos**2
        PC=self.PCxyz[:,:,self.tc_index]
        Lambda=self.Lambda[self.tc_index]
        
        diff2L=np.zeros([self.nbonds,PC.shape[-1]])
        for k in range(3):
            diff2L+=Lambda*(PC[k,self.sel1index]-PC[k,self.sel2index])**2
        
        return d2+np.cumsum(diff2L,axis=-1).T
    
    @property
    def dist2_avg(self):
        """
        Mean of the vector lengths squared

        Returns
        -------
        None.

        """
        return (self.dist**2).mean(0)
    
    def v_from_pca(self,t_index=0):
        """
        This is a check-sum that the bond vectors derived from PCA are the same
        as directly calculated.

        Parameters
        ----------
        t_index : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        v=self.v_mean_pos
        

        v+=(self.PCamp.T[t_index]*(self.PCxyz[:,self.sel1index]-self.PCxyz[:,self.sel2index])).sum(-1).T
        
        return v
    
    #%% Order parameter calculations
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
        assert len(self.sel1)==len(self.sel2),"sel1 and sel2 must have the same number of atoms"
        
        v=(self.pos[:,self.sel1index]-self.pos[:,self.sel2index]).T
        v/=np.sqrt((v**2).sum(0))      
        
        S2=np.ones(v.shape[1])*(-1/2)
        for k in range(3):
            for j in range(k,3):
                S2+=(v[k]*v[j]).mean(-1)**2*(3/2 if k==j else 3)
        return S2
    
    @property
    def S20m(self):
        """
        Calculates the contribution to the total order parameter arising from
        the mth principal component, noting that we number according to sorting
        of the correlation times (so zero is the fastest PC, and -1 is the 
        slowest PC)
        
        Note this is the product of all S2 up to mode m

        Returns
        -------
        2D array (n PCs x n bonds)

        """
        
        if self._S20m is None:
            d20m=self.d2_0m
            
            PC=self.PCxyz[:,:,self.tc_index]
            Lambda=self.Lambda[self.tc_index]
            
                    
            
            X=np.zeros([self.PC.shape[1],self.nbonds])
            for k in range(3):
                for j in range(k,3):
                    P=(self.mean[self.sel1index,k]-self.mean[self.sel2index,k])*\
                        (self.mean[self.sel1index,j]-self.mean[self.sel2index,j])
                        
                        
                    a=Lambda*(PC[k,self.sel1index]-PC[k,self.sel2index])*\
                        (PC[j,self.sel1index]-PC[j,self.sel2index])
                        
                    b=(P+np.cumsum(a.T,axis=0))/d20m
                    
                    # b=P/d20m[0]
                    
                    X+=(b**2)*(1 if k==j else 2)
            S20m=-1/2+3/2*X
            #Cleanup: make the S20m sorted
            while np.any(S20m[:-1]<S20m[1:]):
                i=S20m[:-1]<S20m[1:]
                S20m[:-1][i]=S20m[1:][i]+1e-7
                
            self._S20m=S20m
            
        return self._S20m
    
    @property
    def S2m(self):
        """
        Calculates the contribution to the total order parameter arising from
        the mth principal component, noting that we number according to sorting
        of the correlation times (so zero is the fastest PC, and -1 is the 
        slowest PC)
        

        Returns
        -------
        2D array (n PCs x n bonds)

        """
        if self._S2m is None:
            S20m=self.S20m
            S20m=np.concatenate((np.ones([1,self.nbonds]),S20m),axis=0)
            self._S2m=S20m[1:]/S20m[:-1]
        return self._S2m
    
    @property
    def Am(self):
        """
        Calculates the amplitude from each principal component to the bonds. 
        

        Returns
        -------
        None.

        """
        
        if self._Am is None:
            S20m=np.concatenate([np.ones((1,self.S20m.shape[1])),self.S20m],axis=0)
            self._Am=S20m[:-1]-S20m[1:]
        return self._Am
            

    #%% Correlation functions of the bonds
        
    def Ctmb(self,bond:int):
        """
        Returns the correlation functions for a given bond corresponding to
        each principal component

        Parameters
        ----------
        bond : int
            Index of the desired bond

        Returns
        -------
        2D array (n principal components x n time points)

        """
        S2m=self.S2m[:,bond]
        Ct=self.Ct[self.tc_index]
        
        return (Ct.T*(1-S2m)+S2m).T
    
    @property
    def Ctdirect(self):
        """
        Returns the directly calculated correlation functions for each bond

        Returns
        -------
        2D array (n bonds x n time points)

        """
        if self._Ctdirect is None:
            v=(self.v.T/np.sqrt((self.v.T**2).sum(0)))
            ctc=Ctcalc()
            for k in range(3):
                for j in range(k,3):
                    ctc.a=v[k]*v[j]
                    ctc.c=3/2 if k==j else 3
                    ctc.add()                    
            ct=ctc.Return(-1/2)[0]
            self._Ctdirect=ct
        return self._Ctdirect
        
    @property
    def Ctprod(self):
        """
        Returns correlation functions for all bonds derived from PCA

        Returns
        -------
        2D array (n bonds x n time points)

        """
        
        if self._Ctprod is None:
            self._Ctprod=np.array([self.Ctmb(k).prod(0) for k in range(self.nbonds)])
        return self._Ctprod
        

    #%% Exporting to data
    def direct2data(self):
        """
        Exports the directly-calculated correlation functions to a data object.

        Returns
        -------
        data

        """
        
        sens=clsDict['MD'](t=self.t)
        
        data=clsDict['Data'](R=self.Ctdirect.astype(dtype),
                  Rstd=np.repeat([sens.info['stdev']],self.nbonds,axis=0).astype(dtype),
                  sens=sens,label=self.select.label)
        data.source.filename=self.select.traj.files
        data.source.status='raw'
        data.source.details=self.select.details
        data.source.Type='PCAbond'
        data.details.append('Direct analysis of the correlation function')
        data.details.append('Source is PCA module')
        data.select=self.select
        data.project=self.project
        return data
    
    def prod2data(self):
        """
        Exports the product of correlation functions calculated via PCA to
        a data object for comparison to the directly calculated result

        Returns
        -------
        data

        """
        sens=clsDict['MD'](t=self.t)
        
        data=clsDict['Data'](R=self.Ctprod.astype(dtype),
                  Rstd=np.repeat([sens.info['stdev']],self.nbonds,axis=0).astype(dtype),
                  sens=sens,label=self.select.label)
        data.source.filename=self.select.traj.files
        data.source.status='raw'
        data.source.details=self.select.details
        data.source.Type='PCAbond'
        data.details.append('Product of PCA-derived correlation functions')
        data.details.append('Source is PCA module')
        data.select=self.select
        data.project=self.project
        return data
    
    def PCA2data(self):
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
        
        if self._data is None:
            sens=clsDict['MD'](t=self.t)
            out=Data_PCA(R=self.Ct.astype(dtype),
                         Rstd=np.repeat(np.array([sens.info['stdev']],dtype=dtype),self.Ct.shape[0],axis=0),
                         sens=sens,
                         select=self.select,Type='PCAmode')
            out.source.filename=self.select.traj.files
            out.source.status='raw'
            out.source.details=self.select.details
            out.source.details.append(f'PCA based on analysis of {self.atoms.__len__()} atoms')
            out.source.details.append(f'PCA exported to data with {len(self.Lambda)} principal components')
        
        
            out.label=np.arange(out.R.shape[0],dtype=int)
            out._PCA=self
            
            self._data=out
            self.project.append_data(out)
            
            
            
        return copy(self._data)
    
    def PCA2noopt(self,n=12):
        if self._noopt is None or self._noopt.R.shape[1]!=n:
            d=self.PCA2data()
            d.detect.r_no_opt(12)
            self._noopt=d.fit()
        return self._noopt
    
    
    @property
    def n(self):
        """
        Default number of detectors used to analyze a trajectory. Can also be
        set by the user

        Returns
        -------
        int

        """
        if self._n is None:
            self._n=np.round(np.log10(self.traj.__len__())*2).astype(int)
        return self._n
    
    @n.setter
    def n(self,n):
        self._n=n
    
    def PCA2fit(self,n=None):
        if n is None:n=self.n
        if self._fit is None or self._fit.R.shape[1]!=n:
            noopt=self.PCA2noopt(max([12,n]))
            noopt.detect.r_auto(n)
            self._fit=noopt.fit()
        return self._fit
    
    def direct2fit(self,n=None):
        if n is None:n=self.n
        if self._directfit is None or self._directfit.R.shape[1]!=n:
            d=self.direct2data()
            d.detect.r_no_opt(max([12,n]))
            noopt=d.fit()
            noopt.detect.r_auto(n)
            self._directfit=noopt.fit()
        return self._directfit
    
    def prod2fit(self,n=None):
        if n is None:n=self.n
        if self._prodfit is None or self._prodfit.R.shape[1]!=n:
            d=self.prod2data()
            d.detect.r_no_opt(max([12,n]))
            noopt=d.fit()
            noopt.detect.r_auto(n)
            self._prodfit=noopt.fit()
        return self._prodfit
            
            
    
    def prod_v_direct(self,n=None):
        """
        Plots a comparison of the directly calculated and PCA-derived detector
        analysis of the bonds

        Parameters
        ----------
        n : TYPE, optional
            DESCRIPTION. The default is 6.

        Returns
        -------
        None.

        """
        
        self.project.current_plot=self.project.current_plot+1
        
        
        fit=self.direct2fit(n=n)
        pfit=self.prod2fit(n=n)        
            
        fit.plot()
        pfit.plot()
        
        return fit,pfit
        
    
    #%% Estimating distributions
    def fit_PC(self,n:int=8,max_error=.01):
        """
        Fits the correlation functions of the principal components to each of 
        three models:
        
        0) A*exp(-t/10^z)
        1) A0+A1*exp(-t/10^z1)
        2) A0*exp(-t/10^z0)+A1*exp(-t/10^z1)
    
        We then report the simplest model for each residue satisfying the
        acceptance criteria (max_error)

        Parameters
        ----------
        nz : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        z,A,error,model

        """
        
        if self._mf is None or self._mf[0]!=n or self._mf[1]!=max_error:
            if f'p{n}:PCAMODE:{self.source.short_file.split(".")[0]}' in self.project.titles:
                fit=self.project[f'p{n}:PCAMODE:{self.source.short_file.split(".")[0]}'][0]
            else:
                data=self.PCA2data()
                data.detect.r_no_opt(n)
                fit0=data.fit()
                fit0.detect.r_auto(n)
                fit=fit0.fit()
                
    
            mf=[model_free(fit,model=k) for k in range(3)]
            
            z=np.ones([2,fit.R.shape[0]])*-14
            A=np.zeros(z.shape)
            model=np.zeros(fit.R.shape[0])
            
            
            z[1]=mf[0][0][0]
            A[1]=mf[0][1][0]
            error=mf[0][3]
            
            i=mf[0][3]>max_error
            z[:,i]=mf[1][0][:,i]
            A[:,i]=mf[1][1][:,i]
            error[i]=mf[1][3][i]
            model[i]=1

            i=mf[1][3]>max_error
            z[:,i]=mf[2][0][:,i]
            A[:,i]=mf[2][1][:,i]
            
            error[i]=mf[2][3][i]
            model[i]=2
            
            self._mf=n,max_error,z,A,error,model
        return self._mf[2:]
    
    def total_dist(self,z=None):
        """
        Returns the total distribution of correlation times as a histogram, 
        which only considers the amplitudes of the principal components.

        Parameters
        ----------
        z : array, optional
            z values for the output histogram. The default is None.

        Returns
        -------
        None.

        """
        zf,Af=deepcopy(self.fit_PC()[:2])
        Af*=self.Lambda
        
        if z is None:z=np.linspace(*Defaults['zrange'])
        
        i=np.digitize(zf.flatten(),z)
            
        A=np.array([Af.flatten()[i==k].sum() for k in range(len(z))])
        
        return z,A
    
    def bond_dist(self,z=None):
        """
        Returns the distributions for the individual bonds, which are scaled
        according to the contributing order parameters
        
        Parameters
        ----------
        z : array, optional
            z values for the output histogram. The default is None.

        Returns
        -------
        None.

        """
        
        if z is None:z=np.linspace(*Defaults['zrange'])
        
        zf,Af=deepcopy(self.fit_PC()[:2])
        zf=zf[:,self.tc_index]
        Af=Af[:,self.tc_index]
        
        Af=np.array([(Af0*(1-self.S2m.T)).T for Af0 in Af])
        
        zf=zf.reshape([self.nPC*2])
        Af=Af.reshape([self.nPC*2,self.nbonds])

        i=np.digitize(zf,z)
        
        A=np.array([Af[i==k].sum(0) for k in range(len(z))]).T
        
        return z,A
    
    #%% Chimera videos
    def rho_from_PCA(self,PCAfit=None):
        """
        Determines the contributions of each principal component to a given
        detector response.

        Parameters
        ----------
        sens : sensitivity object
            DESCRIPTION.
        rho_index : int, optional
            DESCRIPTION. The default is None.
        index : int, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if PCAfit is None:PCAfit=self.PCA2fit()
        
        
        rhoPCA=np.array([self.Am.T*R[self.tc_index] for R in PCAfit.R.T])
        return rhoPCA
            
    
        # if rho_index is not None and sens is not None:
        #     rhoz=sens.rhoz[rho_index]
        #     z0=sens.z
        # else:
        #     z0=np.linspace(*Defaults['zrange'])
        #     rhoz=np.ones(z0.shape)
        
        # zf,Af=self.fit_PC()[:2]
        # zf,Af=zf[:,self.tc_index],Af[:,self.tc_index]
        
        # Am=self.Am[:,index]
        
        
        # Afsc=Af*np.array([linear_ex(z0,rhoz,zf0) for zf0 in zf])

        # A=(Afsc*Am).sum(0)
        
        # return A
            
    
    def PCAweight(self,rho_index:int,index:int,PCAfit=None,frac:float=0.75):
        """
        Determines the weighting of the principal components that accounts for
        most of a detector response for a given bond. By default, we account
        for 75% (frac=0.75) of the given bond. 
        
        We choose the weighting by first calculating the contribution of
        each mode to the given detector response. We then sort the weightings
        and apply a Gaussian weighting, with the max at the most influential
        mode (so we only use half the Gaussian). Then we increase sigma until
        we cover 75% of the contributions to the detector response. The fraction
        can be modified by setting frac=0.75
        """
        
        A=self.rho_from_PCA(PCAfit)[rho_index][index]
        
        _,i,j=np.unique(A,return_index=True,return_inverse=True)
        N=np.argmax(np.cumsum(A[i])/A.sum()>frac)
        
        
        
        
    
    @property
    def movie(self):
        if self._movie is None:
            self._movie=PCAmovies(self)
        return self._movie
    
    #%% PDB writing / Chimera
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
        
        

    
    # def S2pca(self,n='all'):
    #     """
    #     Calculates the order parameters for the bond selections via principal
    #     component analysis

    #     Returns
    #     -------
    #     None.

    #     """
    #     if n=='all':n=self.pos.shape[1]*3
        
    #     if self._PC is None or self._PC.shape[1]<n:self.runPCA(n)
        
    #     # v0=(self.pos[:,self.sel1index]-self.pos[:,self.sel2index]).mean(0)
    #     # len2=(v0**2).sum(-1)
    #     v0=self.pos[:,self.sel1index]-self.pos[:,self.sel2index]
    #     len2=np.sqrt((v0**2).sum(-1)).mean(0)**2
        
        
        
    #     Delta=[self.PC[k::3][self.sel1index]-self.PC[k::3][self.sel2index] for k in range(3)]
        
    #     S2=np.ones(len2.shape)*(-1/2)
    #     for k in range(3):
    #         for j in range(k,3):
    #             S2+=(((Delta[k]*Delta[j])*self.Lambda).T/len2).sum(0)**2*(3/2 if k==j else 3)
                
    #     return S2
    
    # def S2covar(self):
    #     """
    #     Calculates S2 from the covariance matrix.

    #     Returns
    #     -------
    #     np.array
    #         Array of order parameters

    #     """
        
    #     v0=self.pos[:,self.sel1index]-self.pos[:,self.sel2index]
    #     len2=np.sqrt((v0**2).sum(-1)).mean(0)**2
        
    #     S2=np.ones(len2.shape)*(-1/2)
    #     M=self.CoVar
        
    #     for k in range(3):
    #         for j in range(k,3):
    #             S2+=M[k::3,j::3][self.sel1index][:,self.sel2index] #Not correct yet
    #             """
    #             We need cov(k_1,j_1)+cov(k_2,j_2)-cov(k_1,j_2)-cov(k_2,j_1)+
    #                         +(<k_1>-<k_2>)*(<j_1>-<j_2>)
                            
    #             We should also work out the impact of the length normalization,
    #             which may not be so simple.
    #             """
        

def model_free(data,model:int=0):
    """
    Performs a simple model free fit, assuming that the amplitudes must sum to
    1

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    model : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    z,A,error

    """
    rhoz=deepcopy(data.sens.rhoz[:-1])
    R=deepcopy(data.R[:,:-1])
    z0=data.sens.z
    
    if model==0:
        error=np.array([np.sqrt(((Rc-R)**2).sum(-1)) for Rc in rhoz.T])
        i=np.argmin(error,axis=0)
        error=np.array([e[i0] for e,i0 in zip(error.T,i)])     
        z=z0[i]
        A=np.ones(z.shape)
        Rc=data.sens.rhoz[:,i].T

    elif model==1:
        error=[]
        A=[]
        R[:,0]-=rhoz[0,0]
        rhoz[0]-=rhoz[0,0]
        for Rc in rhoz.T:
            sc=(((Rc**2).sum(-1)**(-1)*Rc)*R).sum(-1)
            sc[sc<0]=0
            sc[sc>1]=1
            Rc_sc=np.atleast_2d(Rc).T@np.atleast_2d(sc)
            error.append(np.sqrt((((Rc_sc-R.T))**2).sum(0)))
            A.append(sc)
        A,error=np.array(A),np.array(error)    
        i=np.argmin(error,axis=0)
        
        error=np.array([e[i0] for e,i0 in zip(error.T,i)])
        A=np.array([a[i0] for a,i0 in zip(A.T,i)])
        Rc=data.sens.rhoz[:,i]*A
        Rc[0]+=(1-A)*data.sens.rhoz[0,0]
        Rc=Rc.T
        z=np.array([-14*np.ones(i.shape),z0[i]])
        A=np.array([1-A,A])
        
    elif model==2:
        error=[]
        A=[]
        kj=[]
        for k in range(len(z0)):
            for j in range(k+1,len(z0)):
                Rc=rhoz[:,k]-rhoz[:,j]
                Rf=R-rhoz[:,j]
                sc=(((Rc**2).sum()**(-1)*Rc)*Rf).sum(-1)
                sc[sc<0]=0
                sc[sc>1]=1
                Rc_sc=np.atleast_2d(Rc).T@np.atleast_2d(sc)
                error.append(np.sqrt((((Rc_sc-Rf.T))**2).sum(0)))
                A.append(sc)
                kj.append((k,j))
                
        A,error,kj=np.array(A),np.array(error),np.array(kj)
        
        i=np.argmin(error,axis=0)
        
        error=np.array([e[i0] for e,i0 in zip(error.T,i)])
        A=np.array([a[i0] for a,i0 in zip(A.T,i)])
        k=kj.T[0][i]
        j=kj.T[1][i]
        z=np.array([z0[k],z0[j]])
        A=np.array([A,1-A])
        Rc=data.sens.rhoz[:,k]*A[0]+data.sens.rhoz[:,j]*A[1]
        
    return z,A,Rc,error
        
            
            

        
        
Data=clsDict['Data']
class Data_PCA(Data):
    def __init__(self, R=None, Rstd=None, label=None, sens=None, select=None, 
                 src_data=None, Type=None,S2=None, S2std=None, Rc=None):
        
        super().__init__(R=R,Rstd=Rstd,label=label,sens=sens,select=select,
                         src_data=src_data,Type=Type,S2=S2,S2std=S2std,Rc=Rc)
        self._PCA=None
        if self.src_data is not None:
            self._PCA=self.src_data.PCA
    
    
    @property
    def PCA(self):
        return self._PCA
        
    @property
    def select(self):
        """
        Returns the selection if iRED in bond mode

        Returns
        -------
        molselect
            Selection object.

        """
        if 'PCAmode'==self.source.Type:
            return None
        return self.source.select
        
        