#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:32:58 2021

@author: albertsmith
"""

import numpy as np
from pyDR.Sens import Sens,Info
from scipy.sparse.linalg import eigs
from scipy.optimize import lsq_linear as lsqlin
from scipy.optimize import linprog
from pyDR.misc.tools import linear_ex

class Detector(Sens):
    def __init__(self,sens):
        """
        Initiate a detector object with a sensitivity object.
        """
        
        super().__init__(z=sens.z)
        
        "Parameters for detectors"
        self.info.new_exper(z0=0,zmax=0,Del_z=0,stdev=0)
        self.info.del_exp(0)
        
        self.__r=None  #Storage for r matrix        
        
        self.sens=sens
        _=sens._rho_eff   #Run the calculation of the input once to finalize the input rhoz values
        "We'll throw a warning if the input sensitivity gets updated"
        
        self.r_opt=r_opt(self)
        
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
        
        if self.sens.edited:
            print('Warning: the input sensitivities may have been edited, but the detectors have not been updated')

        return self.__r.copy()
    
    def r_no_opt(self,n):
        """
        Generate detectors based only on the singular value decomposition (do not
        use any optimization)
        """
        self.r_opt.no_opt(n)
    
    def r_target(self,target,n):
        """
        Generate n detectors where the first m detectors are approximately equal
        to the rows in target (n>=m). Using larger n than m will result in n-m
        un-optimized detectors in addition to the m detectors optimized to the
        target function.
        
        Target may be input as a list of vectors having the same length as sens.z,
        such that each element corresponds to the correlation time already used
        in this sensitivity object. One may also input a sensitivity object itself,
        in which case we will try to match the sensitivity of the two objects.
        Finally, one may input a dictionary with keys 'z' and 'rhoz', to use a
        different correlation time axis than is used in this sensitivity object
        (in the latter cases, we use linear extrapolation to match the target
        sensitivities to the correlation time axis used here)
        """
        self.r_opt.target(target,n)
        
    
    def update_det(self):
        """
        Updates the detector sensitivities, R matrix, info, etc. for a given
        T matrix
        """
        
        SVD=self.r_opt.SVD
        T=self.r_opt.T        
        
        self.__rho=T@SVD.Vt
        self.__rhoCSA=T@SVD.VtCSA
        # self.__r[bond]=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),\
        #     np.dot(U,np.linalg.solve(T.T,np.diag(S)).T)
        self.__r=((1/self.sens.norm)*(SVD.U@np.linalg.solve(T.T,np.diag(SVD.S))).T).T
        # self.__r=((1/self.sens.norm)*(self.SVD.U@np.diag(self.SVD.S)).T).T
        
        dz=self.z[1]-self.z[0]
        self.info=Info()
        self.info.new_parameter(z0=np.array([(self.z*rz).sum()/rz.sum() for rz in self.__rho]))
        self.info.new_parameter(zmax=np.array([self.z[np.argmax(rz)] for rz in self.__rho]))
        self.info.new_parameter(Del_z=np.array([rz.sum()*dz/rz.max() for rz in self.__rho]))
        self.info.new_parameter(stdev=((np.linalg.pinv(self.__r)**2)@self.sens.info['stdev']**2)**0.5)
        
        self.sens.updated()
    
    #%% Detector optimization
    def _rho(self):
        return self.__rho
    
    def _rhoCSA(self):
        return self.__rhoCSA
    
    def r_auto(self,n,NegAllow=False):
        """
        Generate n detectors that are automatically selected based on the results
        of SVD
        """
        self.r_opt.auto(n=n,NegAllow=NegAllow)
        
    def r_zmax(self,zmax,NegAllow=False):
        """
        Generate n detectors defined by the correlation time of their maximum. 
        Specify the list of maxima (zmax)
        """
        self.r_opt.zmax(zmax=zmax,NegAllow=NegAllow)

class r_opt():
    """
    Class for optimizing detectors sensitivities
    """
    def __init__(self,detect):
        """
        Initiate the detector optimization object
        """
        self.detect=detect
        self.SVD=SVD(detect.sens)
        self.T=None

        
    def no_opt(self,n):
        """
        Generate detectors based only on the singular value decomposition (do not
        use any optimization)
        """
        self.SVD(n)     #Run the SVD
        self.T=np.eye(n) #No optimization
        self.detect.update_det() ##Re-calculate detectors based on the new T matrix

    def opt_z(self,n,z=None,index=None,min_target=None):
        """
        Optimize a detector using n singular values to have amplitude 1 at correlation
        time z (or with index, z=self.z[index]). Used for self.auto(n) and
        self.zmax
        
        n: Number of singular values to use
        z: Target correlation time (this value set to 1)
        index: Use instead of z (z=self.z[index])
        min_target: Vector defining the minimum allowed value of the resulting
        detector sensitivity (default is np.zeros(self.z.shape))
        """
        
        self.SVD(n)
        if min_target is None:min_target=np.zeros(self.detect.z.shape)
        assert z is not None or index is not None,"z or index must be provided"
        if index is None:index=np.argmin(np.abs(self.detect.z-z))
        Vt=self.SVD.Vt
        return linprog(Vt.sum(1),-Vt.T,-min_target,[Vt[:,index]],1,bounds=(-500,500),\
                  method='interior-point',options={'disp':False})['x']
    
    def zmax(self,zmax,NegAllow=0):
        """
        Re-optimize detectors based on a previous set of detectors (where the 
        maximum of the detectors has been recorded)
        """
        zmax=np.atleast_1d(zmax)
        n=zmax.size        
        self.SVD(n)
        self.T=np.eye(n)
        for k,z in enumerate(zmax):
            self.T[k]=self.opt_z(n=n,z=z)
        self.detect.update_det()
        
    
    def auto(self,n,NegAllow=False):
        """
        Generate n detectors that are automatically selected based on the results
        of SVD
        """
        self.SVD(n)
        Vt=self.SVD.Vt
        
        def true_range(k,untried):
            "Find the range around k in untried where all values are True"
            i=np.nonzero(np.logical_not(untried[k:]))[0]
            right=(k+i[0]) if len(i)!=0 else len(untried)
            i=np.nonzero(np.logical_not(untried[:k]))[0]
            left=(i[-1]+1) if len(i)!=0 else 0
            
            return left,right
        
        def find_nearest(Vt,k,untried,error=None,endpoints=False):
            """Finds the location of the best detector near index k. Note that the
            vector untried indicates where detectors may still exist. k must fall 
            inside a range of True elements in untried, and we will only search within
            that range. Note that by default, finding the best detector at the
            end of that range will be disallowed, since the range is usually bound
            by detectors that have already been identified. Exceptions are the first
            and last positions. untried will be modified in-place
            """
            
            left,right=true_range(k,untried)
            
            maxi=100000
            test=k
            while k!=maxi:
                if not(np.any(untried[left:right])):return     #Give up if the whole range of untried around k is False
                k=test
                x=self.opt_z(n=n,index=k)
                rhoz0=(self.SVD.Vt.T@x).T
                maxi=np.argmax(np.abs(rhoz0))
                error[k]=np.abs(k-maxi)
                if k<=maxi:untried[k:maxi+1]=False  #Update the untried index
                else:untried[maxi:k+1]=False
                test=maxi
            
            if (k<=left or k>=right-1) and not(endpoints):
                return None #Don't return ends of the range unless 0 or ntc
            else:
                return rhoz0,x,k
         
        def biggest_gap(untried):
            """Finds the longest range of True values in the untried index
            """
            k=np.nonzero(untried)[0][0]
            gap=0
            biggest=0
            while True:
                left,right=true_range(k,untried)
                if right-left>gap:
                    gap=right-left
                    biggest=np.mean([left,right],dtype=int)
                i0=np.nonzero(untried[right:])[0]
                if len(i0)>0:
                    k=right+np.nonzero(untried[right:])[0][0]
                else:
                    break
            return biggest
        
        #Locate where the Vt are sufficiently large for maxima
        i0=np.nonzero(np.any(np.abs(Vt.T)>(np.abs(Vt).max(1)*.75),1))[0]
        ntc=self.detect.z.size
        untried=np.ones(ntc,dtype=bool)
        untried[:i0[0]]=False
        untried[i0[-1]+1:]=False
        count=0     #How many detectors have we found?
        index=list()    #List of indices where detectors are found
        rhoz=list()     #Optimized sensitivity
        X=list()        #Columns of the T-matrix
        err=np.ones(ntc,dtype=int)*ntc #Keep track of error at all time points tried
            
        "Locate the left-most detector"
        if untried[0]:
            rhoz0,x,k=find_nearest(Vt,0,untried,error=err,endpoints=True)
            rhoz.append(rhoz0)
            X.append(x)
            index.append(k)
            count+=1
        "Locate the right-most detector"
        if untried[-1] and n>1:
            rhoz0,x,k=find_nearest(Vt,ntc-1,untried,error=err,endpoints=True)
            rhoz.append(rhoz0)
            X.append(x)
            index.append(k)
            count+=1
        "Locate remaining detectors"
        while count<n:  
            "Look in the middle of the first untried range"
            k=biggest_gap(untried)
            out=find_nearest(Vt,k,untried,error=err)  #Try to find detectors
            if out: #Store if succesful
                rhoz.append(out[0])
                X.append(out[1])
                index.append(out[2])
#                untried[out[2]-1:out[2]+2]=False #No neighboring detectors
                count+=1
        
        
        i=np.argsort(index).astype(int)
#        pks=np.array(index)[i]
        rhoz=np.array(rhoz)[i]
        self.T=np.array(X)[i]    
        self.detect.update_det()
        if NegAllow:self.allowNeg()
        
    def allowNeg(self):
        """
        Allows detectors that extend to infinite or 0 tc to dip below 0 where
        oscillations occur. Only applied to first and last detector
        """
        update=False
        z,rhoz=self.detect.z,self.detect.rhoz
        if rhoz[0,0]/rhoz[0].max()>.95:
            i=np.argwhere(rhoz[0]<.01)[0,0]
            M=np.concatenate(([np.ones(z.size)],[np.arange(z.size)]),axis=0).T
            x=np.linalg.lstsq(M[i:],rhoz[0,i:],rcond=None)[0]
            min_target=-(M@x)
            self.T[0]=self.opt_z(n=self.T.shape[0],z=self.detect.info['zmax',0],min_target=min_target)
            update=True
        if rhoz[-1,-1]/rhoz[-1].max()>.95:
            i=np.argwhere(rhoz[-1]<.01)[-1,0]
            M=np.concatenate(([np.ones(z.size)],[np.arange(z.size)]),axis=0).T
            x=np.linalg.lstsq(M[:i],rhoz[-1,:i],rcond=None)[0]
            min_target=-(M@x)
            self.T[-1]=self.opt_z(n=self.T.shape[0],z=self.detect.info['zmax',-1],min_target=min_target)
            update=True
        if update:self.detect.update_det()
            
    def target(self,target,n=None):
        """
        Generate n detectors where the first m detectors are approximately equal
        to the rows in target (n>=m). Using larger n than m will result in n-m
        un-optimized detectors in addition to the m detectors optimized to the
        target function.
        
        Target may be input as a list of vectors having the same length as sens.z,
        such that each element corresponds to the correlation time already used
        in this sensitivity object. One may also input a sensitivity object itself,
        in which case we will try to match the sensitivity of the two objects.
        Finally, one may input a dictionary with keys 'z' and 'rhoz', to use a
        different correlation time axis than is used in this sensitivity object
        (in the latter cases, we use linear extrapolation to match the target
        sensitivities to the correlation time axis used here)
        """
        
        if hasattr(target,'z') and hasattr(target,'_rho_eff'):
            z,target=target.z,target._rho_eff[0]
            target=linear_ex(z,target,self.detect.z)
        elif isinstance(target,dict):
            z,target=target['z'],target['rhoz']
            target=linear_ex(z,target,self.detect.z)
        if n is None:n=target.shape[0]
        self.SVD(n)

        self.T=np.eye(n)
        for k,t in enumerate(target):
            self.T[k]=lsqlin(self.SVD.Vt.T,t,lsq_solver='exact')['x']
        
        self.detect.update_det()    #Re-calculate detectors based on the new T matrix
        
    


class SVD(Sens):
    """
    Class responsible for performing and storing/returning the results of singular
    value decomposition of a given set of sensitivities
    """
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
    
    @property
    def M(self):
        """
        Matrix on which to perform the SVD
        """
        return (self.sens._rho_eff[0].T*self.sens.norm).T
    
    @property
    def Mn(self):
        """
        Estimate of M based on SVD.n singular values
        """
        assert self.S is not None,'Mn cannot be calculated without first running the SVD'
        return self.U@(np.diag(self.S)@self.Vt)
    
    
    def run(self,n):
        """
        Runs the singular value decomposition for the largest n singular values
        """
        
        if self.sens.edited:
            print('Warning: the input sensitivities have been edited-the detector sensitivities will be updated accordingly')
            self._S=None #SVD needs to be re-run
            self.sens.updated()
        
        self.n=n
        
        if self.S is None or len(self.S)<n:
            X=self.M
            if np.shape(X)[0]>np.shape(X)[1]: #Use sparse version of SVD
                S2,V=eigs(X.T@X,k=n)
                self._S=np.sqrt(S2.real)
                self._U=((X@V)@np.diag(1/self.S)).real
                self._Vt=V.real.T
            else:
                self._U,self._S,self._Vt=[x.real for x in np.linalg.svd(X)] #Full calculation
            
            self._VtCSA=np.diag(1/self._S)@(self._U.T@(self.sens._rho_effCSA[0].T*self.sens.norm).T)      
            
    def __call__(self,n):
        """
        Runs the singular value decomposition for the largest n singular values
        """
        self.run(n)
                