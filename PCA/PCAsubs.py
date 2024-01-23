#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:10:58 2024

@author: albertsmith
"""

import numpy as np
from ..MDtools import Ctcalc

#%% Correlation functions of the principal components
class PCA_Ct():
    def __init__(self,pca):
        self.pca=pca
        self._tcavg=None
        self._Ct=None
        
        
    
    @property
    def select(self):
        return self.pca.select
    
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
            ctc.a=self.pca.PCamp
            ctc.add()
            ct=ctc.Return()[0]
            ct=ct.T/self.pca.Lambda
            self._Ct=ct.T
        return self._Ct[:self.pca.nPC]
    
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
            self._tcavg=np.array([(Ct[:i0]).sum() for Ct,i0 in zip(self.Ct,i)],dtype=np.float64)
            self._tcavg+=np.arange(len(self._tcavg))*1e-12
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
        return np.unique(self.tc_avg,return_index=True)[1]
    
    @property
    def tc_rev_index(self):
        """
        Returns the index to reverse tc sorting

        Returns
        -------
        None.

        """
        return np.unique(self.tc_avg,return_inverse=True)[1]
    
    
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
        S2m=self.pca.S2.S2m[:,bond]
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
    
    
    
class PCA2Amps():
    def __init__(self,pca):
        self.pca=pca