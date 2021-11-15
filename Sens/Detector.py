#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:32:58 2021

@author: albertsmith
"""

import numpy as np
from pyDR.Sens import Sens,Info
from pyDR.Sens import NMRexper
from scipy.sparse.linalg import eigs
from numpy.linalg import svd

class Detector(Sens):
    def __init__(self,sens):
        """
        Initiate a detector object with a sensitivity object.
        """
        
        super().__init__()
        
        "Parameters for detectors"
        self.info.new_parameter(par='z0')
        self.info.new_parameter(par='Del_z')
        self.info.new_parameter(par='stdev')
        
        self.__r=None         
        self.T=None
        
        self.SVD=SVD(sens) #Initiate the SVD class
        
        self.sens=sens
        _=sens._rho_eff   #Run the calculation of the input once to finalize the input rhoz values
        "We'll throw a warning if the input sensitivity gets updated"
        
        "If bond-specific, initiate all for all bonds"
        if len(sens)>1:
            for s in sens:
                self.append(Detector(s))
                
                
    @property
    def r(self):
        """
        Obtain the r matrix for fitting with detectors
        """
        assert self.__r is not None,"First optimize detectors (r_auto, r_target, r_no_opt)"
        
        if self.sens.info.edited:
            print('Warning: the input sensitivities may have been edited, but the detectors have not been updated')

        return self.__r.copy()
    
    def r_no_opt(self,n):
        """
        Generate detectors based only on the singular value decomposition (do not
        use any optimization)
        """
        self.SVD(n)     #Run the SVD
        T=np.eye(n) #No optimization
        self.update_det(T) #Calculate sensitivities, etc.
        
    
    def update_det(self,T):
        """
        Updates the detector sensitivities, R matrix, info, etc. for a given
        T matrix
        """
        
        if self.T is not None and np.all(T==self.T):return
        
        self.T=T
        self.__rho=T@self.SVD.Vt
        self.__rhoCSA=T@self.SVD.VtCSA
        self.__r=((1/self.sens.norm)*(self.SVD.U@np.diag(self.SVD.S)).T).T
        
        dz=self.z[1]-self.z[0]
        self.info=Info()
        self.info.new_parameter(z0=np.array([(self.z*rz).sum()*dz for rz in self.__rho]))
        self.info.new_parameter(Del_z=np.array([(rz*dz)/rz.max() for rz in self.__rho]))
        self.info.new_parameter((np.linalg.pinv(self.__r)**2)@self.sens.info['stdev']**2)**0.5
    
    def _rho(self):
        return self.__rho
    
    def _rhoCSA(self):
        return self.__rhoCSA


class SVD(Sens):
    def __init__(self,sens):
        """
        Initiate an object responsible for carrying out singular value decomposition
        and sensitivity reconstruction 
        """
        self.sens=sens
        
        self._U=None
        self._S=None
        self._Vt=None
        self._VtCSA=None
        self.n=0
        
    @property
    def Vt(self):
        return self._Vt[:self.n] if self._Vt is not None else None
    
    @property
    def VtCSA(self):
        return self._VtCSA[:self.n] if self._VtCSA is not None else None
    
    @property
    def S(self):
        return self._S[:self.n] if self._S is not None else None
    
    @property
    def U(self):
        return self._U[:,:self.n] if self._U is not None else None
    
    def run(self,n):
        """
        Runs the singular value decomposition for the largest n singular values
        """
        
        if self.sens.info.edited:
            print('Warning: the input sensitivities have been edited-the detector sensitivities will be updated accordingly')
            self.S=None #SVD needs to be re-run
        
        self.n=n
        
        if self.S is None or len(self.S)<n:
            norm=self.sens.norm
            X=(self.sens._rho_eff[0].T*norm).T
            if np.shape(X)[0]>np.shape(X)[1]: #Use sparse version of SVD
                S2,V=eigs(X.T@X,k=n)
                self._S=np.sqrt(S2.real)
                self._U=(X@V)@np.diag(1/self.S)
                self._Vt=V.real.T
            else:
                self._U,self._S,self._Vt=svd(X) #Full calculation
            
            self._VtCSA=np.diag(1/self._S)@(self._U.T@(self.sens._rho_effCSA[0].T*norm).T)
            self.T=np.eye(n)        
            
    def __call__(self,n):
        """
        Runs the singular value decomposition for the largest n singular values
        """
        self.run(n)
                