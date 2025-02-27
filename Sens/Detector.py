#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:32:58 2021

@author: albertsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from pyDR.misc.disp_tools import set_plot_attr,NiceStr
from pyDR.misc import ProgressBar
from pyDR import Sens
from scipy.sparse.linalg import eigs
from scipy.optimize import lsq_linear as lsqlin
from scipy.optimize import linprog
from pyDR.misc.tools import linear_ex
from copy import copy

import warnings
from scipy.linalg import LinAlgWarning
from scipy.optimize import OptimizeWarning
warnings.filterwarnings(action='ignore', category=LinAlgWarning)
warnings.filterwarnings(action='ignore', category=OptimizeWarning)

class Detector(Sens.Sens):
    def __init__(self,sens):
        """
        Initiate a detector object with a sensitivity object.
        """
        
        super().__init__(z=sens.z)
        
        "Parameters for detectors"
        self.info.new_exper(z0=0,zmax=0,Del_z=0,stdev=0)
        self.info.del_exp(0)
        self.info.updated(deactivate=True) #Detectors are not calculated from parameters in info, so we don't use this feature
        
        self.__r=None  #Storage for r matrix  
        self.__locked=False
        
        self.sens=sens
        _=sens.rhoz   #Run the calculation of the input once to finalize the input rhoz values
        "We'll throw a warning if the input sensitivity gets updated"
        
        self.SVD=SVD(sens)
        self.T=None
        self.opt_pars={'options':[]}
        
        "If bond-specific, initiate all for all bonds"
        if len(sens)>1:
            for s in sens:
                self.append(Detector(s))
    
    def __eq__(self,ob):
        if self is ob:return True        #If same object, then equal
        if len(self)!=len(ob):return False  #If different lengths, then not equal
        if str(self.__class__)!=str(ob.__class__):return False
        if ('n' in self.opt_pars and 'n' not in ob.opt_pars) or ('n' not in self.opt_pars and 'n' in ob.opt_pars):
            return False
        elif 'n' not in self.opt_pars and 'n' not in ob.opt_pars:
            return self.sens==ob.sens
        else:
            return super().__eq__(ob)
    
    def del_exp(self,index):
        """
        Deletes detectors from the detector object

        Parameters
        ----------
        index : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not(hasattr(self,'nowarn')):
            print('Warning: Deleting detectors from the detector object will prevent fitting')
        
        i=np.ones(self.rhoz.shape[0],dtype=bool)
        i[index]=False
        self._Sens__rho=self.rhoz[i]
        self._Sens__rhoCSA=self._rhoCSA()[i]
        self.opt_pars['n']=i.sum()
        self.info.del_exp(index)
        
    
    def copy(self):
        """
        Returns a deep copy of the detector object. Note that we do not deep-copy
        the original sensitivity object
        """
        sens=self.sens  #Hold on to the old sensitivity object
        self.sens=None
        out=super().copy()
        out.sens=sens  #Put back old sensitivity object
        self.sens=sens
        return out
    
    def lock(self,locked=True):
        """
        Prevents sensitivites in this detector object from being updated. Usually,
        we run this after performing a fit and storing this detector object as
        the sensitivity of the resulting data object. We do this because it
        does not make sense to edit the sensitivities of the results of fitting
        """
        self.__locked=locked
    
    @property
    def _islocked(self):
        if self.__locked:
            print('Detector object is locked. Re-optimization is not allowed')
            return True
        return False
    
            
    @property
    def r(self):
        """
        Obtain the r matrix for fitting with detectors
        """
        self.reload()
        self.match_parent()
        assert self.__r is not None,"First optimize detectors (r_auto, r_target, r_no_opt)"
        
        if not(self.SVD.up2date):
            print('Warning: detector sensitivities should be updated due to a change in input sensitivities')

        return self.__r.copy()
    
    def reload(self):
        """
        If detector loaded from file, we may need to re-calculate some parameters
        """
        if self.__r is None and self._Sens__rho is not None:
            opt_pars=copy(self.opt_pars)
            inclS2='inclS2' in opt_pars['options']
            R2ex='R2ex' in opt_pars
            target=self._Sens__rho[inclS2:-1] if R2ex else self._Sens__rho[inclS2:]
            
            self.r_target(target)
            if np.max(np.abs(target-self.rhoz))>1e-6:
                print('Warning: Detector reoptimization failed')
            self.opt_pars=copy(opt_pars)
            self.opt_pars['options']=[]
            for o in opt_pars['options']:
                getattr(self,o)()
                
    def match_parent(self):
        """
        If this is a detector sub-object, then match_parent can be run in order
        when the sensitivities or r-matrix is required of the object.

        Returns
        -------
        self

        """
        if self._parent is None:return
        
        self.opt_pars=opts=self._parent.opt_pars
        
        inclS2='inclS2' in opts['options']
        R2ex='R2ex' in opts
        
        Type=opts['Type']
        if Type in ['auto','target','zmax']:
            target=self._parent.rhoz[inclS2:-1] if R2ex else self._parent.rhoz[inclS2:]
            self.r_target(target)
        else:
            self.r_no_opt(opts['n'])
        
        o0=copy(opts['options'])  #What options are used?
        opts['options']=[]   #Clear the list (options only run if not already in list)
        for o in o0:getattr(self,o)()  #Run the option (adds back to the list)
            
    
    def update_det(self):
        """
        Updates the detector sensitivities, R matrix, info, etc. for a given
        T matrix
        """
        
        SVD=self.SVD
        T=self.T        
        
        self._Sens__rho=T@SVD.Vt
        self._Sens__rhoCSA=T@SVD.VtCSA
#        self.__r=((1/self.sens.norm)*(SVD.U@np.diag(SVD.S)@np.linalg.inv(T)).T).T      #Same as below, but in theory slower(?)
        self.__r=((1/self.sens.norm)*(SVD.U@np.linalg.solve(T.T,np.diag(SVD.S)).T).T).T
        
        dz=self.z[1]-self.z[0]
        for k in self.info.keys.copy():self.info.del_parameter(k)
        self.info.new_parameter(z0=np.array([(self.z*rz).sum()/rz.sum() for rz in self.rhoz]))
        self.info.new_parameter(zmax=np.array([self.z[np.argmax(rz)] for rz in self.rhoz]))
        self.info.new_parameter(Del_z=np.array([rz.sum()*dz/rz.max() for rz in self.rhoz]))
        # self.info.new_parameter(stdev=((np.linalg.pinv(self.__r)**2)@self.sens.info['stdev']**2)**0.5)
        if not(np.any(self.sens.info['stdev']==0)):
            self.info.new_parameter(stdev=((np.linalg.pinv((self.__r.T/self.sens.info['stdev'].astype(float)).T)**2).sum(1))**0.5)
        else:
            self.info.new_parameter(stdev=np.zeros(self.rhoz.shape[0]))
        
    #%% Detector optimization
    def _rho(self):
        """
        This works differently than the other sub-classes. update_det finalizes
        the value of _rho, and this function just returns that value.
        """
        self.match_parent()
        assert 'n' in self.opt_pars,"First, optimize detectors before calling Detector._rho"
        if not(self.SVD.up2date):
            print('Warning: detector sensitivities should be updated due to a change in the input sensitivities')
        return self.rhoz
    
    def _rhoCSA(self):
        return self._Sens__rhoCSA
    
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
        assert z is not None or index is not None,"z or index must be provided"
        if min_target is None:min_target=np.zeros(self.z.shape)
        
        if index is None:index=np.argmin(np.abs(self.z-z))
        Vt=self.SVD.Vt

        #TODO I got some errors using r_auto from the linprog. It is solved using the cvxpy. However, couldnt see any
        # differences in the results?
        # I would like to add an argument which asks for a mode, where you can put in cvxpy if necessary
        if False:
            ntc = np.shape(Vt)[1]
            bounds = np.zeros(ntc)
            import cvxpy
            x = cvxpy.Variable(Vt[:, index].shape)
            prob = cvxpy.Problem(cvxpy.Minimize(np.sum(Vt, axis=1) @ x), [-Vt.T @ x <= -bounds, Vt[:, index] @ x == 1])
            prob.solve(verbose=False)
            x = x.value
            return x

        return linprog(Vt.sum(1),A_ub=-Vt.T,b_ub=-min_target,A_eq=[Vt[:,index]],b_eq=1,bounds=(-500,500),\
                  method='interior-point',options={'disp':False})['x']
    
    def r_zmax(self,zmax,Normalization='MP',NegAllow=False):
        """
        Re-optimize detectors based on a previous set of detectors (where the 
        maximum of the detectors has been recorded)
        """
        if self._islocked:return
        
        zmax=np.atleast_1d(zmax)
        zmax.sort()
        n=zmax.size        
        self.SVD(n)
        self.T=np.eye(n)
        for k,z in enumerate(zmax):
            self.T[k]=self.opt_z(n=n,z=z)
        self.opt_pars={'n':n,'Type':'zmax','Normalization':None,'NegAllow':False,'options':[]}
        self.update_det()
        if NegAllow:self.allowNeg()
        
        if Normalization:self.ApplyNorm(Normalization)
        
        return self
    
    def r_no_opt(self,n):
        """
        Generate detectors based only on the singular value decomposition (do not
        use any optimization)
        """
        if self._islocked:return
        
        self.SVD(n)     #Run the SVD
        self.T=np.eye(n) #No optimization
        self.opt_pars={'n':n,'Type':'no_opt','Normalization':None,'NegAllow':False,'options':[]}
        self.update_det() ##Re-calculate detectors based on the new T matrix
        
        return self
        
    
    def r_target(self,target,n=None,Normalization=None):
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
        if self._islocked:return
        
        if hasattr(target,'z') and hasattr(target,'rhoz'):
            z,target=target.z,target.rhoz
            target=linear_ex(z,target,self.z)
        elif isinstance(target,dict):
            z,target=target['z'],target['rhoz']
            target=linear_ex(z,target,self.z)
        if n is None:n=target.shape[0]
        self.SVD(n)

        self.T=np.eye(n)
        for k,t in enumerate(target):
            self.T[k]=lsqlin(self.SVD.Vt.T,t,lsq_solver='exact')['x']    
        self.opt_pars={'n':n,'Type':'target','Normalization':Normalization,'NegAllow':False,'options':[]}
        self.update_det()    #Re-calculate detectors based on the new T matrix
        if Normalization:self.ApplyNorm(Normalization)
        
        return self
    
    def r_auto(self,n:int,Normalization:str='MP',NegAllow:bool=False):
        """
        Generate n detectors that are automatically selected based on the results
        of SVD

        Parameters
        ----------
        n : int
            Number of detectors.
        Normalization : str, optional
            Normalization mode. 'I' yields integral-normalized detectors, 'M'
            yields maximum-normalized detectors that sum to one if S2 is include,
            'MP' yields max-positive normalized detectors, where all detectors,
            including the S2 detector, have maxima of 1 (but do not sum to 1).
            The default is 'MP'.
        NegAllow : bool, optional
            Allows the first/last detectors to oscillate below zero. 
            The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self._islocked:return
        
        self.SVD(n)
        Vt=self.SVD.Vt
        #todo add alternative for linux, because the scipy linalg is to slow -K
        """
        def linprog_cvxpy(Y):
            import cvxpy
            
            Vt = Y[0]
            k = Y[1]
            ntc = np.shape(Vt)[1]
            if np.size(Y) == 3:
                bounds = Y[2]
            else:
                bounds = np.zeros(ntc)
            try:
                x = cvxpy.Variable(Vt[:, k].shape)
                prob = cvxpy.Problem(cvxpy.Minimize(np.sum(Vt, axis=1) @ x), [-Vt.T @ x <= -bounds, Vt[:, k] @ x == 1])
                prob.solve(verbose=False)
                x = x.value
            except Exception as e:
                print("linprog failed:",e)
                x = np.ones(n)
            return x
        """
        
        
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
                if k<=maxi:
                    untried[k:maxi+1]=False  #Update the untried index
                else:
                    untried[maxi:k+1]=False
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
        
        def sweep(error):
            print('Standard optimization failed: ')
            rhoz=list()
            X=list()
            for k in range(len(self.z)):
                ProgressBar(k+1,ntc,'Optimizing:','',0,40)
                x=self.opt_z(n=n,index=k)
                rhoz0=(self.SVD.Vt.T@x).T
                maxi=np.argmax(np.abs(rhoz0))
                error[k]=np.abs(k-maxi)
                X.append(x)
                rhoz.append(rhoz0)
            
            index=np.argsort(error)[:n]
            return index,[X[i] for i in index]
            
        #Locate where the Vt are sufficiently large for maxima
        i0=np.nonzero(np.any(np.abs(Vt.T)>(np.abs(Vt).max(1)*.75),1))[0]
        ntc=self.z.size
        untried=np.ones(ntc,dtype=bool)
        untried[:i0[0]]=False
        untried[i0[-1]+1:]=False
        count=0     #How many detectors have we found?
        index=list()    #List of indices where detectors are found
        rhoz=list()     #Optimized sensitivity
        X=list()        #Columns of the T-matrix
        err=np.ones(ntc,dtype=int)*ntc #Keep track of error at all time points tried
            
        try:
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
        except:
            index,X=sweep(err)
        
        i=np.argsort(index).astype(int)
#        pks=np.array(index)[i]
        # rhoz=np.array(rhoz)[i]
        self.T=np.array(X)[i]    
        # TODO previously, Normalization and NegAllow were not set to the input values. Why??
        self.opt_pars={'n':n,'Type':'auto','Normalization':Normalization,'NegAllow':NegAllow,'options':[]}
        self.update_det()


        if NegAllow:self.allowNeg()
        if Normalization:self.ApplyNorm(Normalization)
        
        return self
        
    def allowNeg(self):
        """
        Allows detectors that extend to infinite or 0 tc to dip below 0 where
        oscillations occur. Only applied to first and last detector
        """
        if self._islocked:return
        
        update=False
        z,rhoz=self.z,self.rhoz
        if rhoz[0,0]/rhoz[0].max()>.95:
            i=np.argwhere(rhoz[0]<.01)[0,0]
            M=np.concatenate(([np.ones(z.size)],[np.arange(z.size)]),axis=0).T
            x=np.linalg.lstsq(M[i:],rhoz[0,i:],rcond=None)[0]
            min_target=-(M@x)
            self.T[0]=self.opt_z(n=self.T.shape[0],z=self.info['zmax',0],min_target=min_target)
            update=True
        if rhoz[-1,-1]/rhoz[-1].max()>.95:
            i=np.argwhere(rhoz[-1]<.01)[-1,0]
            M=np.concatenate(([np.ones(z.size)],[np.arange(z.size)]),axis=0).T
            x=np.linalg.lstsq(M[:i],rhoz[-1,:i],rcond=None)[0]
            min_target=-(M@x)
            self.T[-1]=self.opt_z(n=self.T.shape[0],z=self.info['zmax',-1],min_target=min_target)
            update=True
        if update:self.update_det()
        self.opt_pars['NegAllow']=True
        
        return self
        
                    
    def inclS2(self,Normalization=None):
        """
        Creates an additional detector from S2 measurements, where that detector
        returns the difference between (1-S2) and an optimized sum of the other
        detectors.

        Parameters
        ----------
        Normalization : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self._islocked:return

        assert 'n' in self.opt_pars.keys(),'First perform initial detector optimization before including S2'
        
        if 'inclS2' in self.opt_pars['options']:self.removeS2()
        # if 'inclS2' in self.opt_pars['options']:return #Function already run
        self.opt_pars['options'].append('inclS2')
        
        norm=Normalization if Normalization is not None else self.opt_pars['Normalization']
        if norm is None:norm='MP'
        # if norm is not None:
        #     self.ApplyNorm(norm)
        #     self.opt_pars['Normalization']=norm.upper()

        if norm.lower()=='mp' or norm.lower()=='i':
            wt=linprog(-(self.rhoz.sum(axis=1)).T,self.rhoz.T,np.ones(self.rhoz.shape[1]),\
                        bounds=(-500,500),method='interior-point',options={'disp' :False,})['x']
            rhoz0=[1-(self.rhoz.T@wt).T]
            rhoz0CSA=[1-(self._rhozCSA.T@wt).T]
            sc=np.atleast_1d(rhoz0[0].max() if norm.lower()=='mp' else rhoz0[0].sum()*(self.z[1]-self.z[0]))
            self._Sens__rho=np.concatenate((rhoz0/sc,self.rhoz))
            self._Sens__rhoCSA=np.concatenate((rhoz0CSA/sc,self._rhozCSA))
            mat1=np.concatenate((np.zeros([self.__r.shape[0],1]),self.__r),axis=1)
            mat2=np.atleast_2d(np.concatenate((sc,wt.T),axis=0))
            self.__r=np.concatenate((mat1,mat2),axis=0)
        elif norm.lower()=='m':
            self.__r=np.concatenate((\
                         np.concatenate((np.zeros([self.__r.shape[0],1]),self.__r),axis=1),\
                               np.ones([1,self.__r.shape[1]+1])),axis=0)
            self._Sens__rho=np.concatenate(([1-self._Sens__rho.sum(0)],self._Sens__rho),axis=0)
            self._Sens__rhoCSA=np.concatenate(([1-self._Sens__rhoCSA.sum(0)],self._Sens__rhoCSA),axis=0)
        else:
            assert 0,"Unknown normalization (use 'M','MP', or 'I')"
            
        pars={'z0':np.nan,'zmax':self.z[np.argmax(self.rhoz[0])],'Del_z':np.nan,
                    'stdev':((np.linalg.pinv(self.__r)[0,:-1]**2)@self.sens.info['stdev']**2)**0.5}
        self.info.new_exper(**pars)
        ne=len(self.info)
        self.info._Info__values=self.info._Info__values.T[np.concatenate(([-1],np.arange(ne-1)))].T
    
        return self
    
    def removeS2(self):
        if self._islocked:return
        
        if 'inclS2' not in self.opt_pars['options']:
            print('S2 not included')
            return self
        self._Sens__rho=self.rhoz[1:]
        self._Sens__rhoCSA=self._Sens__rhoCSA[1:]
        self.__r=self.r[:-1,1:]
        self.opt_pars['options'].remove('inclS2')
        
        return self
    
    def R2ex(self,vref=None):
        """
        Includes a detector that will remove influence of fast exchange on the R2
        measurements. Detector responses then correspond to the R2 exchange 
        contribution at the field given by v_ref.

    

        Returns
        -------
        self

        """
        if self._islocked:return
        
        
        v0=self.sens.info['v0'].astype(float)*(self.sens.info['Type']=='R2')
        if vref is None:
            vref=v0[self.sens.info['Type']=='R2'].min()
        
        r_ex_vec=v0**2/vref**2
        
        rhoz=np.zeros(self.tc.size)
        rhoz[-1]=1e6
        
        self.__r=np.concatenate([self.r,np.transpose([r_ex_vec])],axis=1)
        self._Sens__rho=np.concatenate([self._Sens__rho,[rhoz]],axis=0)
        self._Sens__rhoCSA=np.concatenate([self._Sens__rhoCSA,[np.zeros(self.tc.size)]],axis=0)
        
        
        
        pars={'z0':np.nan,'zmax':np.nan,'Del_z':np.nan,
                    'stdev':((np.linalg.pinv(self.__r)[-1]**2)@self.sens.info['stdev']**2)**0.5}
        self.info.new_exper(**pars)
        
        self.opt_pars['options'].append('R2ex')
        self.opt_pars['R2ex_vref']=vref
        
        return self
    
    def removeR2ex(self):
        """
        Removes the detector for R2 exchange

        Returns
        -------
        None.

        """
        if self._islocked:return
        
        if 'R2ex' not in self.opt_pars['options']:
            print('S2 not included')
            return self
        
        self._Sens__rho=self.rhoz[:-1]
        self._Sens_rhoCSA=self._Sens__rhoCSA[:-1]
        self.__r=self.r[:,:-1]
        self.opt_pars['options'].remove('R2ex')
        self.opt_pars.pop('R2ex_vref')
        self.info.del_exp(-1)
        
        return self
        
    
    def ApplyNorm(self,Normalization='MP'):
        """
        Applies normalization to the set of detectors. Options are 'I' with forces
        the integrals to all equal 1, 'M' which yields detectors with maxima
        of 1, and 'MP' which also yields detectors with maxima of 1, but prevents
        the detector derived from S2 from being negative (if set to 'M', the sum
        of the detectors is 1, but the first detector, derived from S2, may 
        become negative)
        """
        if self._islocked:return
        
        assert Normalization[0].lower() in ['m','i'],'Normalization should be "M", "MP", or "I"'
        for k,rhoz in enumerate(self._Sens__rho):
            if Normalization.lower()[0]=='m':
                self.T[k]/=rhoz.max()
            else:
                self.T[k]/=rhoz.sum()*self.dz
        self.opt_pars['Normalization']=Normalization
        self.update_det()  
    
    
    def plot_rhoz(self,index=None,ax=None,norm=False,**kwargs):
        """
        Plots the detector sensitivities

        Parameters
        ----------
        index : TYPE, optional
            DESCRIPTION. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        norm : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
    
        if index is None:
            index=np.arange(self.info.N)
        else:
            index=np.array(index)

        hdl=super().plot_rhoz(index=index,ax=ax,norm=norm,**kwargs)
        
        if 'R2ex' in self.opt_pars['options'] and self.info.N-1 in index:
            hdl[-1].set_alpha(0)
            hdl[-1].axes.set_ylim([self.rhoz[index][:-1].min(),self.rhoz[index][:-1].max()])
            
        return hdl
            
    
        
    def plot_fit(self,index=None,ax=None,norm=False,**kwargs):
        """
        Plots the sensitivities of the data object.
        """
        
        if index is None:index=np.ones(self.SVD.M.shape[0],dtype=bool)
        index=np.atleast_1d(index)
            
        assert np.issubdtype(index.dtype,int) or np.issubdtype(index.dtype,bool),"index must be integer or boolean"
    
        a=self.SVD.M[index].T #Get sensitivities
        norm_vec=np.abs(a).max(0) if norm else self.sens.norm
        a/=norm_vec
        fit=self.SVD.Mn[index].T
        fit/=norm_vec
   
        if ax is None:
            fig=plt.figure()
            ax=fig.add_subplot(111)

        hdl=[*ax.plot(self.z,a,color='red'),*ax.plot(self.z,fit,color='black',linestyle=':')]

        set_plot_attr(hdl,**kwargs)
        ax.legend([hdl[0],hdl[hdl.__len__()>>1]],['Input','Fit'])
        
        ax.set_xlim(self.z[[0,-1]])
        ticks=ax.get_xticks()
        nlbls=4
        step=int(len(ticks)/(nlbls-1))
        start=0 if step*nlbls==len(ticks) else 1
        lbl_str=NiceStr('{:q1}',unit='s')
        ticklabels=['' for _ in range(len(ticks))]
        for k in range(start,len(ticks),step):ticklabels[k]=lbl_str.format(10**ticks[k])
        
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
        
#        ax.set_xticklabels(ticklabels)
        ax.set_xlabel(r'$\tau_\mathrm{c}$')
        
        ax.set_ylabel(r'$\rho_n(z)$')
               
        return hdl

    def _hash(self):
        #todo get rid of that later
        return hash(self)
    
    def __hash__(self):
        if 'n' not in self.opt_pars:   #Unoptimized set of detectors (hash not defined)
            return hash(self.sens)
        return hash(self.rhoz.tobytes())

    # def __hash__(self):
    #     warnings.warn("implement hash value for Detector!")
    #     return 0

class SVD():
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
        self._sens_hash=None
        
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
        return (self.sens.rhoz.T*self.sens.norm).T
    
    @property
    def Mn(self):
        """
        Estimate of M based on SVD.n singular values
        """
        assert self.S is not None,'Mn cannot be calculated without first running the SVD'
        return self.U@(np.diag(self.S)@self.Vt)
    
    
    @property
    def up2date(self):
        """
        Returns True if the SVD has been performed on up-to-date sensitivities.
        Returns False if the SVD has been perormed on out-of-date sensitivities
        or if SVD has not been run
        """
        if self._sens_hash is None or self._sens_hash!=self.sens._hash:return False
        return True
    
    def update(self):
        """
        Updates the SVD calculation in case the sensitivities have changed

        Returns
        -------
        None.

        """
        if not(self.up2date):self(self.n)
        
    
    def run(self,n):
        """
        Runs the singular value decomposition for the largest n singular values
        """
        
        if self._sens_hash is not None and not(self.up2date):
            print('Warning: the input sensitivities have been edited-the detector sensitivities will be updated accordingly')
            self._S=None #SVD needs to be re-run
        
        self.n=n
        
        if self.S is None or len(self.S)<n:
            X=self.M
            if np.shape(X)[0]>np.shape(X)[1]: #Use sparse version of SVD
                S2,V=eigs(X.T@X,k=n)
                self._S=np.sqrt(S2.real)
                self._U=((X@V)@np.diag(1/self._S)).real
                self._Vt=V.real.T
            else:
                self._U,self._S,self._Vt=np.linalg.svd(X) #Full calculation
                self._Vt=self._Vt[:X.shape[0]]
            
            sign=[np.sign(V[np.argwhere(np.abs(V)>np.abs(V).max()*0.5)[0,0]]) for V in self._Vt]
            #By default, the SVD (or eigs) does not return vectors with consistent signs
            #It's basically random. This wastes a lot of time for us, since we
            #can't average together data using r_no_opt. Furthermore, we can't
            #use the same optimization of detectors resulting from different runs
            #of r_no_opt. So, we come up with some means of always having the same
            #sign. It does not matter what it is, as long as it's consistent. Then,
            #we say that the first time the abs(Vt) exceeds half the max(abs(Vt)),
            #the sign of Vt should always be positive.
            self._Vt=(self._Vt.T/sign).T
            self._U/=sign
            
            self._VtCSA=np.diag(1/self._S)@(self._U.T@(self.sens._rho_effCSA[0].T*self.sens.norm).T)
            
            self._sens_hash=self.sens._hash     #Store the current hash value of the sensitivity
            
    def __call__(self,n):
        """
        Runs the singular value decomposition for the largest n singular values
        """
        self.run(n)
                