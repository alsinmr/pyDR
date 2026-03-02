#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:10:58 2024

@author: albertsmith
"""

import numpy as np
from ..MDtools import Ctcalc
from .. import clsDict
from .. import Defaults
from ..MDtools import vft
from copy import copy
import os
import matplotlib.pyplot as plt
dtype=Defaults['dtype']

#%% Correlation functions of the principal components
class PCA_Ct():
    def __init__(self,pca):
        self.pca=pca
        self._tcavg=None
        self._Ct=None
        self._Ctdirect=None
        self._Ctprod=None
        self._Ctsum=None
        
        
    
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
        return self.pca.t
    
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
            dt=self.t[1]
            i=np.argmax(self.Ct<0,axis=1)
            self._tcavg=np.array([(Ct[:i0]).sum()*dt for Ct,i0 in zip(self.Ct,i)],dtype=np.float64)
            self._tcavg+=np.arange(len(self._tcavg))*1e-12 #Enforce unique values
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
        Ct=self.Ct
        
        return (Ct.T*(1-S2m)+S2m).T
    
    def CtmA(self):
        pass
        
    
    @property
    def Ctdirect(self):
        """
        Returns the directly calculated correlation functions for each bond

        Returns
        -------
        2D array (n bonds x n time points)

        """
        if self._Ctdirect is None:
            v=self.pca.Vecs.v.T
            v=(v/np.sqrt((v**2).sum(0)))
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
            self._Ctprod=np.array([self.Ctmb(k).prod(0) for k in range(self.pca.nbonds)])
        return self._Ctprod
    
    @property
    def Ctsum(self):
        """
        Returns the sum of correlation functions for all bonds, weighted by
        the Am.
        
        In principle, this is an approximation, but we include it since
        it should be faster, and if relatively accurate, then we can use it.

        Returns
        -------
        2D array (n bonds x n time points)

        """
        if self._Ctsum is None:
            self._Ctsum=((self.pca.S2.Am.T@self.Ct).T+(1-self.pca.S2.Am.sum(0))).T
        return self._Ctsum
            
    

#%% Vector and Distance functions    
class PCAvecs():    
    def __init__(self,pca):
        self.pca=pca
        
    
    @property
    def v(self):
        """
        Calculates vectors connecting atoms in sel1 and sel2 for all loaded 
        time points

        Returns
        -------
        3D array (n time points x n atoms x 3)

        """
        pca=self.pca
        pca.load()
        return pca.pos[:,pca.sel1index,:]-pca.pos[:,pca.sel2index,:]
    
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
        pca=self.pca
        return pca.mean[pca.sel1index]-pca.mean[pca.sel2index]
    
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
        pca=self.pca
        d2=self.dist_mean_pos**2
        PC=pca.PCxyz[:,:,pca.Ct.tc_index]
        Lambda=pca.Lambda[pca.Ct.tc_index]
        
        diff2L=np.zeros([pca.nbonds,PC.shape[-1]])
        for k in range(3):
            diff2L+=Lambda*(PC[k,pca.sel1index]-PC[k,pca.sel2index])**2
        
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
    
    
    
class PCA_S2():
    def __init__(self,pca):
        self.pca=pca
        self._S20m=None
        self._S2m=None
        self._Am=None
        self._AmCC=None
        self._S20mCC=None
        self._S2mCC=None
    
        
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
        assert len(self.pca.sel1)==len(self.pca.sel2),"sel1 and sel2 must have the same number of atoms"
        
        v=self.pca.Vecs.v.T
        v/=np.sqrt((v**2).sum(0))      
        
        S2=np.ones(v.shape[1])*(-1/2)
        for k in range(3):
            for j in range(k,3):
                S2+=(v[k]*v[j]).mean(-1)**2*(3/2 if k==j else 3)
        return S2
    
    
    
    @property
    def d2_0m(self):
        """
        Returns vectors between sel1 and sel2 that have been calculated for
        all principal components from 0 to m

        This is <(v^{0->m})^2> in our derivation.

        Returns
        -------
        None.

        """
        pca=self.pca
        d2=pca.Vecs.dist_mean_pos**2
        PC=pca.PCxyz[:,:,pca.Ct.tc_index]
        Lambda=pca.Lambda[pca.Ct.tc_index]
        
        diff2L=np.zeros([pca.nbonds,PC.shape[-1]])
        for k in range(3):
            diff2L+=Lambda*(PC[k,pca.sel1index]-PC[k,pca.sel2index])**2
        
        return d2+np.cumsum(diff2L,axis=-1).T
    
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
            pca=self.pca
            d20m=self.d2_0m   #This is already sorted!
            
            PC=pca.PCxyz[:,:,pca.Ct.tc_index]
            Lambda=pca.Lambda[pca.Ct.tc_index]
            
                    
            
            X=np.zeros([pca.PC.shape[1],pca.nbonds])
            for k in range(3):
                for j in range(k,3):
                    P=(pca.mean[pca.sel1index,k]-pca.mean[pca.sel2index,k])*\
                        (pca.mean[pca.sel1index,j]-pca.mean[pca.sel2index,j])
                        
                        
                    a=Lambda*(PC[k,pca.sel1index]-PC[k,pca.sel2index])*\
                        (PC[j,pca.sel1index]-PC[j,pca.sel2index])
                        
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
            S20m=np.concatenate((np.ones([1,self.pca.nbonds]),S20m),axis=0)
            self._S2m=S20m[1:]/S20m[:-1]
        return self._S2m[self.pca.Ct.tc_rev_index]
    

    
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
        return self._Am[self.pca.Ct.tc_rev_index]
    
    # def S20mCC(self,q):
    #     """
        
    #     Calculates the contribution to the total order parameter arising from
    #     the mth principal component, noting that we number according to sorting
    #     of the correlation times (so zero is the fastest PC, and -1 is the 
    #     slowest PC)
        
    #     Note this is the product of all S2 up to mode m

    #     Returns
    #     -------
    #     2D array (n PCs x n bonds)

    #     """
        

    #     pca=self.pca
    #     d20m=self.d2_0m
        
    #     PC=pca.PCxyz[:,:,pca.Ct.tc_index]
    #     Lambda=pca.Lambda[pca.Ct.tc_index]
        
                
        
    #     X=np.zeros([pca.PC.shape[1],pca.nbonds])
    #     for k in range(3):
    #         for j in range(3):
    #             P=(pca.mean[pca.sel1index[q],k]-pca.mean[pca.sel2index[q],k])*\
    #                 (pca.mean[pca.sel1index,j]-pca.mean[pca.sel2index,j])
                    
                    
    #             a=Lambda*(PC[k,pca.sel1index[q]]-PC[k,pca.sel2index[q]])*\
    #                 (PC[j,pca.sel1index]-PC[j,pca.sel2index])
                    
    #             b=(P+np.cumsum(a.T,axis=0))/np.sqrt(d20m[:,q]*d20m.T).T
                
    #             # b=P/d20m[0]
                
    #             X+=(b**2)
    #     S20m=-1/2+3/2*X
    #     #Cleanup: make the S20m sorted
    #     while np.any(S20m[:-1]<S20m[1:]):
    #         i=S20m[:-1]<S20m[1:]
    #         S20m[:-1][i]=S20m[1:][i]+1e-7
            
            
    #     return S20m
    @property
    def S20mCC(self):
        """
        Calculates contributions to the cross-correlated amplitudes from all
        motions up to mode m

        Returns
        -------
        np.array

        """
        if self._S20mCC is None:
            pca=self.pca
            Lambda=pca.Lambda[pca.Ct.tc_index] #Resorted Lambda
            PC=pca.PCxyz[:,:,pca.Ct.tc_index] #Resorted PCs
            d20m=self.d2_0m   #Mean-squared distances vs. m (already resorted)
            
            out=np.zeros([pca.PC.shape[1],pca.nbonds,pca.nbonds])
            
            for q in range(pca.nbonds):
                X=np.zeros([pca.PC.shape[1],pca.nbonds])
                for k in range(3):
                    # TODO could we run from k to 3 instead of  0 to 3?
                    for j in range(3):
                        
                        P=(pca.mean[pca.sel1index[q],k]-pca.mean[pca.sel2index[q],k])*\
                            (pca.mean[pca.sel1index,j]-pca.mean[pca.sel2index,j])
                            
                            
                        a=Lambda*(PC[k,pca.sel1index[q]]-PC[k,pca.sel2index[q]])*\
                            (PC[j,pca.sel1index]-PC[j,pca.sel2index])
                        b=(P+np.cumsum(a.T,axis=0))/np.sqrt(d20m[:,q]*d20m.T).T
                        
                        X+=b**2
                        
                out[:,:,q]=-1/2+3/2*X
                
            self._S20mCC=out
            
        return self._S20mCC
              
    @property
    def S2mCC(self):
        """
        Calculates contributions to the cross-correlated amplitudes from mode m

        Returns
        -------
        np.array

        """
        
        if self._S2mCC is None:
            S20mCC=self.S20mCC
            S20mCC=np.concatenate((np.ones([1,self.pca.nbonds,self.pca.nbonds]),S20mCC),axis=0)
            self._S2mCC=(S20mCC[1:]/S20mCC[:-1])[self.pca.Ct.tc_rev_index]
        return self._S2mCC
    
    @property
    def AmCC(self):
        if self._AmCC is None:
            S20mCC=np.concatenate([np.ones((1,*self.S20mCC.shape[1:])),self.S20mCC],axis=0)
            self._AmCC=(S20mCC[:-1]-S20mCC[1:])[self.pca.Ct.tc_rev_index]
        return self._AmCC

    
    def plotS2_direct_v_prod(self,ax=None):
        """
        Make a plot of S2 comparing the direct calculation and the product of
        the S2m

        Parameters
        ----------
        ax : matplotlib axis, optional
            The default is None.

        Returns
        -------
        ax

        """
        if ax is None:ax=plt.subplots()[1]
        
        ax.plot(np.arange(len(self.S2direct)),self.S2direct,color='red',label='Direct')
        ax.plot(np.arange(len(self.S2direct)),self.S2m.prod(0),color='black',linestyle=':',label='Prod.')
        
        ax.legend()
        
        ax.set_xticklabels([],rotation=90)
        def fun(i,pos,xlabel=self.pca.select.label):
            i=int(i)
            if i>=len(xlabel):return ''
            if i<0:return ''
            return xlabel[i]
        ax.xaxis.set_major_locator(plt.MaxNLocator(30,integer=True))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fun))
        return ax
    
#%% Class for re-weighting the PCA to show different motions
class Weighting():
    def __init__(self,pca):
        """
        Calculates how to reweight the principal components to show
        specific timescales and motion of particular bonds.

        Parameters
        ----------
        pca : PCA
            PCA object

        Returns
        -------
        None.

        """
        self.pca=pca
        
    
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
        
        pca=self.pca
        if PCAfit is None:PCAfit=pca.Data.PCARef
        
        
        rhoPCA=np.array([pca.S2.Am.T*R for R in PCAfit.R.T])
        return rhoPCA
    
    def timescale(self,timescale,PCAfit=None):
        """
        Returns a weighting of the principal components for a given timescale
        or timescales.
        
        This is determined from the stored sensitity object and the relative
        amplitudes of the detectors at the given timescale. 

        Parameters
        ----------
        timescale : float,array
            timescale or timescales given on a log-scale. The default is None.

        Returns
        -------
        np.array

        """
        if PCAfit is None:PCAfit=self.pca.Data.PCARef
        sens=PCAfit.sens
        
        oneD=not(np.array(timescale).ndim)
    
        timescale=np.atleast_1d(timescale)
        
        i=np.array([np.argmin(np.abs(ts-sens.z)) for ts in timescale],dtype=int)
        wt0=sens.rhoz[:,i]
        # wt0/=wt0.sum(0)
        
        return (PCAfit.R@wt0).flatten() if oneD else PCAfit.R@wt0
    
    def rho_spec(self,rho_index:int,PCAfit=None,frac:float=0.75):
        """
        Determines which principal components account for most of a detector 
        response for all bonds. By default, we account for 75% (frac=0.75) of 
        the given bond. 
        
        At the moment, we just select N principle components until we account
        for 75% of the motion. We have considered using some kind of
        variable function, rather than just a boolean (include/don't include)

        Parameters
        ----------
        rho_index : int, optional
            Index of the detector. If set to None (the default), then this will
            return the an index to cover the whole timescale range, i.e. all
            of S2 for that bond
        PCAfit : TYPE, optional
            DESCRIPTION. The default is None.
        frac : float, optional
            DESCRIPTION. The default is 0.75.

        Returns
        -------
        None.

        """
        if PCAfit is None:PCAfit=self.pca.Data.PCARef
        A=self.rho_from_PCA(PCAfit)
        
        A=self.rho_from_PCA(PCAfit)[rho_index].sum(0).astype(np.float64)
        A+=np.arange(A.size)*1e-12 #Hack to avoid non-unique values of A
        
        Asort,i=np.unique(A,return_inverse=True)
        N=np.argmax(np.cumsum(Asort[::-1])/A.sum()>frac)+1
        
        out=np.zeros(self.pca.nPC,dtype=bool)
        out[-N:]=True
        out=out[i]
        
        return out
        
    
    def bond(self,index:int,rho_index:int=None,PCAfit=None,frac:float=0.75):
        """
        Determines which principal components account for most of a detector 
        response for a given bond. By default, we account for 75% (frac=0.75) 
        of the given bond. 
        
        At the moment, we just select N principle components until we account
        for 75% of the motion. We have considered using some kind of
        variable function, rather than just a boolean (include/don't include)

        Parameters
        ----------
        index : int
            Index of the bond
        rho_index : int, optional
            Index of the detector. If set to None (the default), then this will
            return the an index to cover the whole timescale range, i.e. all
            of S2 for that bond
        PCAfit : TYPE, optional
            DESCRIPTION. The default is None.
        frac : float, optional
            DESCRIPTION. The default is 0.75.

        Returns
        -------
        None.

        """
        if PCAfit is None:PCAfit=self.pca.Data.PCARef
        if rho_index is None:
            A=self.pca.S2.Am[:,index].astype(np.float64)
        else:
            A=self.rho_from_PCA(PCAfit)[rho_index][index].astype(np.float64)

        A+=np.arange(A.size)*1e-12 #Hack to avoid non-unique values of A
        
        Asort,i=np.unique(A,return_inverse=True)
        N=np.argmax(np.cumsum(Asort[::-1])/A.sum()>frac)+1
        
        out=np.zeros(self.pca.nPC,dtype=bool)
        out[-N:]=True
        out=out[i]
        
        return out

#%% Impulse response: A last attempt

class Impulse():
    def __init__(self,pca):
        self.pca=pca
        self._PCamp=None
        self._v_dev=None
        
    @property
    def PCamp(self):
        """
        PC amplitudes starting from the max or min of each principal component
        occuring during the first half of the stored trajectory

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self._PCamp is None:
            l2=self.pca.PCamp.shape[1]//2
            i=np.argmax(np.abs(self.pca.PCamp[:,:l2]),axis=-1)
            self._PCamp=np.array([a[i0:i0+l2] for a,i0 in zip(self.pca.PCamp,i)])
        return self._PCamp
    
    @property
    def v_dev(self):
        """
        Vector pointing in the direction of the max deviation of each bond

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self._v_dev is None:
            v0=vft.norm(self.pca.Vecs.v_mean_pos.T)
            euler=vft.getFrame(v0)
            v=vft.R(vft.norm(np.swapaxes(self.pca.Vecs.v.T,1,2)),*vft.pass2act(*euler))
            # v=vft.applyFrame(,nuZ_F=v0)
            _,beta,gamma=vft.getFrame(v,return_angles=True)
            beta=beta.mean(0)
            gamma=np.mod(gamma,np.pi).mean(0)
            self._v_dev=vft.R(vft.R([0,0,1],0,beta,gamma),*euler)
            
        return self._v_dev
        
    
    def sign(self,index:int):
        """
        Returns the sign to apply to each principal component to get the Impulse
        response function for a given bond (puts all PCs in phase)

        Parameters
        ----------
        index : int
            Bond index

        Returns
        -------
        np.array
            Array of 1 or -1 for each principal component.

        """
        v0=vft.norm((self.pca.Vecs.v_mean_pos[index]+(self.PCamp[:,0]*\
            (self.pca.PCxyz[:,self.pca.sel1index[index]]-self.pca.PCxyz[:,self.pca.sel2index[index]])).T).T)
        v1=vft.norm((self.pca.Vecs.v_mean_pos[index]-(self.PCamp[:,0]*\
            (self.pca.PCxyz[:,self.pca.sel1index[index]]-self.pca.PCxyz[:,self.pca.sel2index[index]])).T).T)
            
        vref=self.v_dev[:,index]
        
        return (2*np.argmax([(v0.T*vref).sum(1),(v1.T*vref).sum(1)],axis=0)-1)
    
    def PCamp_bond(self,index:int):
        """
        Returns the PC amplitudes with signs shifted to maximize motion of the
        selected bond

        Parameters
        ----------
        index : int
            Bond index

        Returns
        -------
        np.array
            Array of amplitudes for each principal component with signs 
            switched to produced the impulse response

        """
        
        return (self.PCamp.T*self.sign(index)).T
        
        
        
#%% Histograms
class Hist():
    def __init__(self,pca):
        self.pca=pca
        
    @property
    def project(self):
        return self.pca.project
        
    def plot(self,n0:int=0,n1:int=1,ax=None,maxbin:float=None,nbins:int=None,cmap='nipy_spectral',index=None,**kwargs):
        """
        Creates a 2D histogram of two principal components. Specify the desired
        components (n0,n1=0,1 by default)
        
        Can also create a 1D histogram, set n1 to None

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
        index : np.array
            Indexes the frames to be included in the histogram. 
            Defaults to None (all frames) 
        **kwargs : TYPE
            Plotting arguments to be passed to hist2d.

        Returns
        -------
        TYPE
            Handle for the hist2d object

        """
        PCamp=self.pca.PCamp
        
        if ax is None:ax=plt.figure().add_subplot(111)
        if maxbin is None:
            if n1 is None:
                maxbin=np.max([np.max(np.abs(PCamp[n0]))])
            else:
                maxbin=np.max([np.max(np.abs(PCamp[n0])),np.max(np.abs(PCamp[n1]))]) 
        if nbins is None:
            nbins=min([100,PCamp.shape[1]//4])
        
        if index is None:
            index=np.ones(PCamp[n0].size,dtype=bool)
        
        
        if n1 is None:
            ax.hist(PCamp[n0][index],bins=np.linspace(-maxbin,maxbin,nbins),**kwargs)
            ax.set_xlabel(f'PC {n0}')
            ax.set_ylabel('Frequency')
            return ax
        
        out=ax.hist2d(PCamp[n0][index],PCamp[n1][index],bins=np.linspace(-maxbin,maxbin,nbins),cmap=cmap,**kwargs)
        ax.set_xlabel(f'PC {n0}')
        ax.set_ylabel(f'PC {n1}')
        # return out #Should we change this back to out?
        return ax
    #%% PDB writing / Chimera
    
    def PC2index(self,PCamp):
        """
        Returns the frame in the trajectory that most closely matches the provided
        principal component amplitudes (PCamp). One must provide the first N 
        principal components. To omit a principal component from the selection, 
        set to np.nan

        Parameters
        ----------
        PCamp : list-like (usually np.array)
            Values of the first N principal components to match to a frame in the
            trajectory

        Returns
        -------
        int

        """
        PCamp=np.atleast_1d(PCamp)
        assert PCamp.ndim==1 and PCamp.dtype!=object,"PCamp cannot have more than one dimension, and can only contain numbers/np.nan"
        index=np.zeros(self.pca.PCamp.shape[0],dtype=bool)
        index[:PCamp.size]=np.logical_not(np.isnan(PCamp))
        
        PCamp=PCamp[np.logical_not(np.isnan(PCamp))]
        
        return np.argmin(((self.pca.PCamp[index].T-PCamp)**2).sum(-1))
        
        
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
        if A is None:A=np.sqrt(self.pca.Lambda[n])
        
        i=np.arange(len(self.pca.atoms)) if sel is None else getattr(self.pca,f'sel{sel}index')
        
        pos0=self.pca.mean[i]
        
        pos0+=A*self.pca.PC[:,n].reshape([self.pca.PC.shape[0]//3,3])
        return pos0
    
    def write_pdb(self,n:int=0,A:float=None,PCamp:list=None,from_traj:bool=True,select_str:str='protein',filename:str=None):
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
            If this is provided, then 'n' and 'A' are ignored.
            The default is None.
        from_traj : bool, optional
            Extracts a frame from the trajectory that is closest given PCamps
            instead of calculating the inverse position. Only active if PCamp
            is defined. The default is True.
        select_str : str, optional
            If from_traj is True, this will apply a selection to the MDanalysis
            universe. The default is 'protein'

        Returns
        -------
        None.

        """
        frame0=self.pca.traj.frame
        if filename is None:
            folder=self.project.directory if self.project is not None and self.project.directory is not None \
                else os.path.split(self.pca.select.molsys.topo)[0]
            filename=os.path.join(folder,'pca.pdb')
        atoms=self.pca.uni.atoms.select_atoms(select_str) if (from_traj and PCamp is not None) else self.pca.atoms
        if PCamp is not None:
            if from_traj:
                i=self.PC2index(PCamp)
                self.pca.traj[i]
                if hasattr(self.pca.traj,'traj_index'):
                    q=self.pca.traj.traj_index
                    print(f'Extracting frame {self.pca.traj.trajs[q].frame} from trajectory {q}')
                atoms.positions=self.pca.align(self.pca.ref_pos, 
                                               self.pca.uni.select_atoms(self.pca.align_ref),
                                               atoms)
            else:
                pos=self.pca.mean
                for k,A in enumerate(PCamp):
                    pos+=A*self.pca.PC[:,k].reshape([self.pca.PC.shape[0]//3,3])
                atoms.positions=pos
        else:
            atoms.positions=self.PC2pos(n=n,A=A)
        atoms.write(filename)
        self.pca.traj[frame0]
        return filename
    
    def chimera(self,n:int=0,std:float=1,PCamp:list=None,from_traj:bool=True,select_str:str='protein',cmap_ch='tab10'):
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
        from_traj : bool, optional
            Extracts a frame from the trajectory that is closest given PCamps
            instead of calculating the inverse position. Only active if PCamp
            is defined. The default is True.
        select_str : str, optional
            If from_traj is True, this will apply a selection to the MDanalysis
            universe. The default is 'protein'

        Returns
        -------
        self

        """            
        if isinstance(cmap_ch,str):cmap_ch=plt.get_cmap(cmap_ch)
        if PCamp is not None:
            filename=self.write_pdb(n=n,PCamp=PCamp,from_traj=from_traj,select_str=select_str)
            if self.project.chimera.current is None:self.project.chimera.current=0
            self.project.chimera.command_line('open "{0}"'.format(filename))
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=[int(c*100) for c in cmap_ch(mdls-1)[:-1]]
            if from_traj:
                self.project.chimera.command_line(['ribbon #{0}','~show #{0}','color #{0} {1},{2},{3}'.format(mdls,clr[0],clr[1],clr[2])])
            else:
                self.project.chimera.command_line(['~ribbon #{0}','show #{0}','color #{0} {1},{2},{3}'.format(mdls,clr[0],clr[1],clr[2])])
            self.project.chimera.command_line(self.project.chimera.saved_commands)
            return
        
            
        if not(hasattr(std,'__len__')):
            A=np.array([-std,std])*np.sqrt(self.pca.Lambda[n])
        elif hasattr(std,'__len__') and len(std)==1:
            A=np.array([-std[0],std[0]])*np.sqrt(self.pca.Lambda[n])
        else:
            A=np.array(std)*np.sqrt(self.pca.Lambda[n])
        if self.project.chimera.current is None:
            self.project.chimera.current=0
        for A0 in A:
            filename=self.write_pdb(n=n,A=A0,PCamp=PCamp)
            if self.project.chimera.current is None:self.project.chimera.current=0
            self.project.chimera.command_line('open "{0}"'.format(filename))       
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=[int(c*100) for c in cmap_ch(mdls-1)[:-1]]
            self.project.chimera.command_line(['~ribbon','show','color #{0} {1},{2},{3}'.format(mdls,clr[0],clr[1],clr[2])])
        self.project.chimera.command_line(self.project.chimera.saved_commands)
        
        return self
    
    def hist2struct(self,nmax:int=4,from_traj:bool=True,select_str:str='protein',ref_struct:bool=False,ax=None,cmap_ch='tab10',n_colors=10,**kwargs):
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
        from_traj : bool, optional
            Extracts a frame from the trajectory that is closest to the selected
            points instead of calculating the inverse position
            The default is True.
        select_str : str, optional
            If from_traj is True, this will apply a selection to the MDanalysis
            universe. The default is 'protein'
        ref_struct : bool, optional
            Show a reference structure in chimera (mean structure). 
            The default is False.
        **kwargs : TYPE
            Keyword arguments to be passed to the PCA.plot function.

        Returns
        -------
        None.

        """
        
        
        if isinstance('cmap_ch',str):cmap_ch=plt.get_cmap(cmap_ch).resampled(n_colors)
        
        x,y=int(np.ceil(np.sqrt(nmax))),int(np.ceil(np.sqrt(nmax)))
        if (x-1)*y>=nmax:x-=1
        
        if ax is None:
            fig=plt.figure()
            ax=[fig.add_subplot(x,y,k+1) for k in range(nmax)]
        else:
            fig=ax[0].figure
        hdls=list()
        for k,a in enumerate(ax):
            self.plot(n0=k,n1=k+1,ax=a,**kwargs)
            hdls.append([a.plot([0,0],[0,0],color='white',linestyle=':',visible=False)[0] for _ in range(2)])
        
        fig.tight_layout()    
        
        if ref_struct:
            self.chimera(PCamp=[0])
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=cmap_ch((mdls-1)%n_colors)
            for k,a in enumerate(ax):
                a.scatter(0,0,marker='x',color=clr)
        
        PCamp=[None for _ in range(nmax+1)]
        markers=['x','o','+','v','>','s','1','*']
        mkr_hdls=[]
        def onclick(event):
            if event.inaxes:
                ax0=event.inaxes
                if ax0 in ax:
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
                        if from_traj:
                            i=self.PC2index(PCamp)
                            PCamp0=self.pca.PCamp[:len(PCamp)][:,i] #Set PCamp to the nearest value in trajectory
                        else:
                            PCamp0=PCamp
                        self.chimera(PCamp=PCamp0,from_traj=from_traj,cmap_ch=cmap_ch)
                        for k,a in enumerate(ax):
                            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
                            clr=cmap_ch((mdls-1)%n_colors)
                            
                            if mdls==0:
                                while mkr_hdls:mkr_hdls.pop().remove()
                            
                            mkr_hdls.append(a.scatter(PCamp0[k],PCamp0[k+1],100,marker=markers[(mdls-1)%len(markers)],linewidth=3,color=clr))
                            
                        #Clear the positions in the plot
                        for k in range(len(PCamp)):PCamp[k]=None
                        for h in hdls:
                            for h0 in h:
                                h0.set_visible(False)
                    plt.pause(0.01)
            else: #Clicking outside the axes clears out the positions
            
                if self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)==0:
                    while mkr_hdls:mkr_hdls.pop().remove()
                    
                for k in range(len(PCamp)):PCamp[k]=None
                for h in hdls:
                    for h0 in h:
                        h0.set_visible(False)
                plt.pause(0.01)
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        self.h2s_ax=ax
        return ax


class Cluster():
    def __init__(self,pca):
        from sklearn import cluster as cluster
        self.cluster=cluster
        
        self.algorithm='MiniBatchKMeans'
        self.cluster_kwargs={'n_clusters':3,'random_state':42}  #Put in random state for reproducibility
        
        self.pca=pca
        self._index=[0,1]
        
        self._cclass=None
        self._output=None
        self._sorting=None
        self._state=None #For unusual usage: manual replacement of state definition
        
    @property
    def project(self):
        return self.pca.project
        
    @property
    def n_clusters(self):
        #Changing order here because some algorithms do not take n_clusters as argument
        if self._output is not None and hasattr(self._output,'n_clusters'):
            return self._output.n_clusters
        
        if 'n_clusters' in self.cluster_kwargs:
            return self.cluster_kwargs['n_clusters']
        
        return None
    
    @n_clusters.setter
    def n_clusters(self,value:int):
        self.cluster_kwargs['n_clusters']=value
        self._cclass=None
        self._output=None
        self._state=None
        self._sorting=None
        
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self,index):
        self._index=index
        self._cclass=None
        self._output=None
        self._state=None
        self._sorting=None
    
    
    @property
    def cclass(self):
        setup=False
        if self._cclass is None or str(self._cclass.__class__).split('.')[-1].split("'")[0]!=self.algorithm:
            setup=True
            if self.algorithm not in dir(self.cluster):
                print('Available algorithms:')
                for name in dir(self.cluster):
                    if name[0]!='_':print(name)
                assert False,'Unknown clustering algorithm'
                
        for key,value in self.cluster_kwargs.items():
            if hasattr(self._output,key) and getattr(self._output,key)!=value:
                setup=True
                break
        
        if setup:
            self._cclass=getattr(self.cluster,self.algorithm)(**self.cluster_kwargs)
            self._output=None
            
            
        return self._cclass
    
    @property
    def output(self):
        if self._output is None:
            self._sorting=None
            self._output=self.cclass.fit(self.pca.PCamp[self.index].T)
        return self._output
    
    @property
    def state(self):
        if self._state is not None:return self._state
        if self._sorting is None:
            self.PCavg
        i=self._sorting
        out=np.zeros(self.output.labels_.shape,dtype=int)
        for k,i0 in enumerate(i):
            out[self.output.labels_==i0]=k
        self._state=out
        
        return out
    
    @state.setter
    def state(self,state):
        self._state=state
    
    @property
    def populations(self):
        return np.unique(self.state,return_counts=True)[1]/self.state.size
    
    def plot(self,ax=None,skip:int=10,maxbin:float=None,nbins:int=None,cmap='binary',cmap_cl='tab10',percent:bool=True,**kwargs):
        nplots=len(self.index)-1
        if ax is not None:
            if nplots>1:
                assert hasattr(ax,'__len__'),'For nD>2, must provide a list of axes'
                ax=np.array(ax).flatten()
                fig=ax[0].figure
                
        else:
            i0=i1=int(np.ceil(np.sqrt(nplots)))
            if (i0-1)*i1>=nplots:i0-=1
            fig=plt.figure()
            ax=[fig.add_subplot(i0,i1,k+1) for k in range(nplots)]
            
        for k,a in enumerate(ax):
            self.pca.Hist.plot(self.index[k],self.index[k+1],ax=a,maxbin=maxbin,nbins=nbins,cmap=cmap,**kwargs)
            cmap0=plt.get_cmap(cmap_cl) if isinstance(cmap_cl,str) else cmap_cl
            for q in range(self.n_clusters):
                a.scatter(self.pca.PCamp[self.index[k]][self.state==q][::skip],
                          self.pca.PCamp[self.index[k+1]][self.state==q][::skip],s=.1,color=cmap0(q))
                if percent:
                    a.scatter(self.pca.PCamp[self.index[k]][self.state==q].mean(),
                           self.pca.PCamp[self.index[k+1]][self.state==q].mean(),s=25,
                           color='black',marker='^')
                
                
                    a.text(self.pca.PCamp[self.index[k]][self.state==q].mean(),
                           self.pca.PCamp[self.index[k+1]][self.state==q].mean(),
                           f'{self.populations[q]*100:.1f}%',
                           horizontalalignment='center',verticalalignment='center')
                
            if -1 in self.state:
                a.scatter(self.pca.PCamp[self.index[k]][self.state==-1][::skip],
                          self.pca.PCamp[self.index[k+1]][self.state==-1][::skip],s=.1,color=0.8)
                
        return ax
    
    def plot_t_depend(self,ax=None):
        if ax is None:ax=plt.subplots()[1]
        ax.scatter(self.pca.t,self.state,3,marker='o',color='black')
        ax.set_ylim([-.5,self.n_clusters-.5])
        if len(self.pca.traj.lengths)>1:
            lengths=self.pca.traj.lengths
            for k in range(len(lengths)-1):
                ax.plot(np.sum(lengths[:k+1])*np.ones(2),ax.get_ylim(),color='grey',linestyle=':')
                
        return ax
        
        
    
    @property
    def PCavg(self):
        """
        Returns the average PC amplitudes within each cluster

        Returns
        -------
        np.array

        """
        
        out=np.zeros([self.n_clusters,len(self.index)])
        
        for k in range(self.n_clusters):
            out[k]=self.pca.PCamp[self.index][:,self.output.labels_==k].mean(-1)
            
            
        self._sorting=np.argsort(out[:,0])
        out=out[self._sorting]        
        
        return out
    
    @property
    def cluster_index(self):
        """
        Returns an index giving the frame in the trajectory corresponding to
        the closest approach of the trajectory to the mean position of each
        cluster

        Returns
        -------
        np.array

        """
        
        PCavg=self.PCavg
        
        out=np.zeros(self.n_clusters,dtype=int)
        for k in range(self.n_clusters):
            error=((self.pca.PCamp[self.index].T-PCavg[k])**2).sum(-1)
            out[k]=np.argmin(error)
        return out
    
    def write_pdb(self,state:int=None,frame:int=None,filename:str=None,from_traj:bool=True,select_str:str='protein'):
        """
        Writes out a pdb or pdbs corresponding to the mean position of each cluster.
        If from_traj=True, then a frame will be taken from the trajectory that
        is closest from to the cluster mean. Otherwise, positions will be
        constructed by inverting PCA.
        
        Note that from_traj=True

        Parameters
        ----------
        state : int, optional
            Which PCA cluster to use. The default is None, which will write out
            all states at once. In this case, filename should be formattable
            so that the outputs do not overwrite each other. Leaving filename
            as None will also work.
        filename : str, optional
            File name to save to. The default is None (pca{state}.pdb)
        from_traj : bool, optional
            Extracts a frame from the trajectory. The default is True.
        select_str : str, optional
            Apply selection to write out PCA results. Default is 'protein'. Note
            that if from_traj=False, then only atoms that were in the original
            PCA analysis can be written out.

        Returns
        -------
        None.

        """
        
        if filename is None:
            folder=self.project.directory if self.project is not None and self.project.directory is not None \
                else os.path.split(self.pca.select.molsys.topo)[0]
            filename=os.path.join(folder,'pca{0}.pdb')
            
        if state is None and frame is None:
            return [self.write_pdb(state=k,filename=filename.format(k),from_traj=from_traj) for k in range(self.n_clusters)]
        elif frame is not None:
            state=self.state[frame]
            from_traj=True
                
        frame0=self.pca.traj.frame
        if from_traj:
            self.pca.traj[self.cluster_index[state] if frame is None else frame]
            print(self.pca.traj.frame)
        else:
            pos=self.pca.mean
            for k,A in enumerate(self.PCavg[state]):
                pos+=A*self.PC[:,self.index[k]].reshape([self.pca.PC.shape[0]//3,3])
            self.pca.atoms.positions=pos
        
        atoms=self.pca.uni.atoms.select_atoms(select_str)
        print(atoms.universe.trajectory.frame)
        atoms.write(filename.format(state))
        
        self.pca.traj[frame0]
            
            
        return filename.format(state)
        
    def chimera(self,state:int=None,frame:int=None,from_traj:bool=True,select_str:str='protein'):         
        
        
        filenames=self.write_pdb(state=state,frame=frame,from_traj=from_traj,select_str=select_str)
        if state is None and frame is None:
            state=np.arange(self.n_clusters)
        elif frame is not None:
            state=self.state[frame]
        if not(isinstance(filenames,list)):filenames=[filenames]
        if not(hasattr(state,'__len__')):state=[state]

        if self.project.chimera.current is None:self.project.chimera.current=0
        
        for state0,filename in zip(state,filenames):
            self.project.chimera.command_line('open "{0}"'.format(filename))       
            mdls=self.project.chimera.CMX.how_many_models(self.project.chimera.CMXid)
            clr=[int(c*100) for c in plt.get_cmap('tab10')(state0)[:-1]]
            self.project.chimera.command_line(['~show','ribbon','color #{0} {1},{2},{3}'.format(mdls,clr[0],clr[1],clr[2])])
        self.project.chimera.command_line(self.project.chimera.saved_commands)
        
        
                     

class Ramachandran():
    def __init__(self,pca):
        """
        Class for creating Ramachandran plots from trajectories, including
        selecting regions of the PCA

        Parameters
        ----------
        pca : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.pca=pca
        
        self.resids=None
        
        self._phi=None
        self._psi=None
        
        self._cpca="CombinedPCA" in str(pca.__class__)
        
        self._vCm1N=None
        self._vNCa=None
        self._vCaC=None
        self._vCNp1=None
            
        
    @property
    def resids(self):
        """
        List of residues for which the N, Ca, and C atoms are available in 
        pca.atoms, as well as the C from the previous residue
        

        Returns
        -------
        None.

        """
        if self._resids is None:
            if self._cpca:
                resids=self.pca.pcas[0].Ramachandran.resids
                for pca in self.pca.pcas[1:]:
                    resids=np.intersect1d(pca.Ramachandran.resids,resids)
                self._resids=resids                    
                self._icpca=[]
                for pca in self.pca.pcas:
                    self._icpca.append(np.isin(pca.Ramachandran.resids,self.resids))
                return self._resids


            iCA=self.pca.atoms.names=='CA'
            iC=self.pca.atoms.names=='C'
            iN=self.pca.atoms.names=='N'
            
            resids=np.intersect1d(self.pca.atoms.resids[iCA],self.pca.atoms.resids[iC])
            resids=np.intersect1d(resids,self.pca.atoms.resids[iN])
            resids=np.intersect1d(resids,self.pca.atoms.resids[iC]+1)
            resids=np.intersect1d(resids,self.pca.atoms.resids[iN]-1)
            
            self._resids=resids
            
            self._iCA,self._iC,self._iN,self._iCm1,self._iNp1=[copy(x) for x in (iCA,iC,iN,iC,iN)]
            
            self._iCA[iCA]=np.isin(self.pca.atoms[iCA].resids,resids)
            self._iC[iC]=np.isin(self.pca.atoms[iC].resids,resids)
            self._iN[iN]=np.isin(self.pca.atoms[iN].resids,resids)
            self._iCm1[iC]=np.isin(self.pca.atoms[iC].resids+1,resids)
            self._iNp1[iN]=np.isin(self.pca.atoms[iN].resids-1,resids)
            
        return self._resids
        
    @resids.setter
    def resids(self,value):
        self._resids=value
        
        
    @property
    def CA(self):
        return self.pca.atoms[self._iCA]
    
    @property
    def C(self):
        return self.pca.atoms[self._iC]
    
    @property
    def N(self):
        return self.pca.atoms[self._iN]
    
    @property
    def Cm1(self):
        return self.pca.atoms[self._iCm1]
    
    @property
    def Np1(self):
        return self.pca.atoms[self._iNp1]
    
    
    @property
    def vCm1N(self):
        if self._vCm1N is None:
            v=self.pca.pos[:,self._iCm1]-self.pca.pos[:,self._iN]
            self._vCm1N=(v.T/np.sqrt((v**2).sum(-1).T)).T
        return self._vCm1N
    

    @property
    def vNCa(self):
        if self._vNCa is None:
            v=self.pca.pos[:,self._iN]-self.pca.pos[:,self._iCA]
            self._vNCa=(v.T/np.sqrt((v**2).sum(-1).T)).T
        return self._vNCa
        
    @property
    def vCaC(self):
        if self._vCaC is None:
            v=self.pca.pos[:,self._iCA]-self.pca.pos[:,self._iC]
            self._vCaC=(v.T/np.sqrt((v**2).sum(-1).T)).T
        return self._vCaC
    
    @property
    def vCNp1(self):
        if self._vCNp1 is None:
            v=self.pca.pos[:,self._iC]-self.pca.pos[:,self._iNp1]
            self._vCNp1=(v.T/np.sqrt((v**2).sum(-1).T)).T
        return self._vCNp1
    
    def _PhiPsi(self):
        self.resids #Properly initialize
        
        if self._cpca:
            self._phi=np.concatenate([pca.Ramachandran.phi[i] for i,pca in zip(self._icpca,self.pca.pcas)],axis=1)
            self._psi=np.concatenate([pca.Ramachandran.psi[i] for i,pca in zip(self._icpca,self.pca.pcas)],axis=1)
            return self._phi,self._psi
                
        
        # First calculate phi
        vz=self.vNCa.T
        vxz=self.vCm1N.T
        v0=self.vCaC.T
        # Use function out of the frames tool box to apply a reference frame
        v=vft.applyFrame(v0,nuZ_F=vz,nuXZ_F=vxz)
        self._phi=np.arctan2(v[1],-v[0])
        
        # Second, calculate psi, recycle some vectors
        vz=self.vCaC.T
        vxz=self.vNCa.T
        v0=self.vCNp1.T
        
        v=vft.applyFrame(v0,nuZ_F=vz,nuXZ_F=vxz)
        self._psi=np.arctan2(v[1],-v[0])
        
        return self._phi,self._psi
    
    @property
    def phi(self):
        if self._phi is None:
            self._PhiPsi()
        return self._phi
    
    @property
    def psi(self):
        if self._psi is None:
            self._PhiPsi()
        return self._psi
    
    def plot(self,resid=None,ax=None,index=None,bins=72,cmap='cool',what='phipsi',**kwargs):
        """
        

        Parameters
        ----------
        resid : TYPE, optional
            DESCRIPTION. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        index : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if ax is None:ax=plt.subplots()[1]
        
        phi,psi=self.phi,self.psi
        
        if index is not None:
            phi,psi=phi[:,index],psi[:,index]
            
        if resid is not None:
            i=resid==self.resids
            phi,psi=phi[i].flatten(),psi[i].flatten()
        else:
            phi,psi=phi.flatten(),psi.flatten()
        
        if what.lower() in ['phi','psi']:
            q=phi if what.lower()=='phi' else psi
            y,x=np.histogram(q*180/np.pi,range=(-180,180),bins=bins)
            y=y/phi.__len__()
            x=x[:-1]+(x[1]-x[0])/2
            ax.plot(x,y,**kwargs)
            ax.set_xlabel((r'$\phi$' if what.lower()=='phi' else r'$\psi$')+r' / $^\circ$')
            ax.set_ylabel('frequency')
            if resid is not None:
                ax.set_title(f'Residue {resid}')
            return ax
            
            
        range=[[-np.pi,np.pi],[-np.pi,np.pi]]
        out=np.histogram2d(phi,psi,bins=bins,range=range)
        
        x=out[1][:-1]+(out[1][1]-out[1][0])/2
        y=out[2][:-1]+(out[2][1]-out[2][0])/2
        z=np.log10(out[0])
        z[out[0]==0]=-1
        ax.contourf(x*180/np.pi,y*180/np.pi,z.T,cmap=cmap,**kwargs)
        ax.set_xlabel(r'$\phi$ / $^\circ$')
        ax.set_ylabel(r'$\psi$ / $^\circ$')
        
        return ax
    
    def pca_plot(self,resid=None,ax0=None,ax=None,bins=72,cmap='cool',what='phipsi'):
        
        if ax0 is not None:ax0.cla()
        ax0=self.pca.Hist.plot(ax=ax0)
        
        if hasattr(resid,'__len__'):
            sz=[np.sqrt(len(resid)).astype(int),np.sqrt(len(resid)).astype(int)]
            if np.prod(sz)<len(resid):
                sz[1]+=1
            if ax is None:
                fig=plt.figure()
                ax=[fig.add_subplot(*sz,k+1) for k in range(len(resid))]
        else:
            if ax is None:ax=plt.subplots()[1]
            ax=[ax]
            resid=[resid]
            
        
        for resi,a in zip(resid,ax):
            self.plot(resid=resi,ax=a,bins=bins,cmap=cmap,what=what)
        
        
        
        def callback(*args,**kwargs):
            pc0=ax0.get_xlim()
            pc1=ax0.get_ylim()
            
            index=np.logical_and(np.logical_and(self.pca.PCamp[0]>=pc0[0],self.pca.PCamp[0]<=pc0[1]),
                                 np.logical_and(self.pca.PCamp[1]>=pc1[0],self.pca.PCamp[1]<=pc1[1]))

            for resi,a in zip(resid,ax):
                a.cla()
                self.plot(resid=resi,ax=a,bins=bins,cmap=cmap,what=what,color='black',linestyle=':')
                self.plot(resid=resi,ax=a,bins=bins,cmap=cmap,index=index,what=what,**kwargs)
            plt.show()
            return None
        
        ax0.callbacks.connect('xlim_changed',callback)
        ax0.callbacks.connect('ylim_changed',callback)
        ax[0].figure.set_size_inches([10.6,7.1])
        ax[0].figure.tight_layout()
        
        return ax
               
            
        
    
    
    
        
        
    
        
        
        
    
    
        
        
        
        
        
#%% Port various correlation functions to data objects        
class PCA2Data():
    def __init__(self,pca):
        """
        Data manager for the pca class. Uses a project to store the resulting
        data objects

        Parameters
        ----------
        pca : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.pca=pca
        self.project=clsDict['Project']() if pca.project is None else pca.project
        self._detect=None
        self._n=None
        self._n_noopt=12
        self._locs={}
        self._BondRef=None
        self._PCARef=None
        
    def clear(self):
        """
        Resets the stored data. This will clear the project if not shared with
        pca.

        Returns
        -------
        None.

        """
        if self.project is not self.pca.project:
            self.project=clsDict['Project']()
        self._locs={}
    
    @property
    def BondRef(self):
        """
        Default data set when a directly-calculated data object is required.
        
        Should have proc or opt_fit status. Defaults to the output of 
        PCA2data.direct

        Returns
        -------
        data

        """
        if self._BondRef is not None:
            return self._BondRef
        return self.direct()
    
    @BondRef.setter
    def BondRef(self,data):
        assert data.source.status in ['proc','opt_fit'],'Data needs to have status proc or opt_fit'
        self._BondRef=data
        
    @property
    def PCARef(self):
        """
        Default data set when a PCA data object is required.
        
        Should have proc or opt_fit status. Defaults to the output of 
        PCA2data.PCA

        Returns
        -------
        data

        """
        if self._PCARef is not None:
            return self._PCARef
        return self.PCA()
    
    @PCARef.setter
    def PCARef(self,data):
        assert data.source.status in ['proc','opt_fit'],'Data needs to have status proc or opt_fit'
        self._PCARef=data
        
        
    @property
    def n_noopt(self):
        """
        Number of unoptimized detectors for initial analysis

        Returns
        -------
        int

        """
        return self._n_noopt
    
    @n_noopt.setter
    def n_noopt(self,n):
        if n!=self._n:
            self._detect=self.sens.Detector().r_no_opt(n) #Update the detect object
        self._n_noopt=n
    
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
            self._n=np.round(np.log10(self.pca.traj.__len__())*2).astype(int)
        return self._n
    
    @n.setter
    def n(self,n):
        for k in self._locs:
            if 'proc' in self._locs[k]:self._locs[k].pop('proc')
            if 'opt_fit' in self._locs[k]:self._locs[k].pop('opt_fit')
        self._n=n
    

    
    
    @property
    def sens(self):
        """
        Since we always start from the same trajectory, the sensitivity and
        initial detector objects can always be the same. The sensitivity is 
        returned here

        Returns
        -------
        MD

        """
        if self._detect is None:
            sens=clsDict['MD'](t=self.pca.Ct.t)
            self._detect=sens.Detector().r_no_opt(self.n)
        return self._detect.sens
            
    @property
    def detect(self):
        """
        Since we always start from the same trajectory, the sensitivity and
        initial detector objects can always be the same. The detector object is 
        returned here

        Returns
        -------
        Detector

        """
        if self._detect is None:
            self.sens
        return self._detect
    
    def _find(self,key,status='proc'):
        """
        Finds a data object based on its index in the project and/or its title
        
        If both are provided, a check will be performed that both yield the
        same data object. If not, a message will be shown, and title will be
        given priority

        Parameters
        ----------
        index : int, optional
            DESCRIPTION. The default is None.
        title : str, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        data

        """
        
        if key in self._locs and status in self._locs[key]:
            index,title=self._locs[key][status]
        else:
            return None
        
        if title is not None:
            assert title in self.project.titles,f'Data "{title}" is missing from project'
            
        if index is None:
            return self.project[title][0]
        elif title is None:
            assert index<len(self.project),f"Index {index} is too large for project of length {len(self.project)}"
            return self.project[index]
        
        if self.project[index].title==title:
            return self.project[index]
        else:
            print('Warning: index and titles did not agree in PC data project')
            return self.project[title][0]
        
        
    def _raw(self,key):
        """
        Exports a matrix of correlation functions to a data object
        
        

        Parameters
        ----------
        Ct : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if key in self._locs:return self._find(key,'raw')
        
        if key=='direct':
            DataObj=clsDict['Data']
            loc='Ctdirect'
            details=['Direct analysis of the correlation function','Source: PCA module']
            Type='MD'
        elif key=='prod':
            DataObj=clsDict['Data']
            loc='Ctprod'
            details=['Product of PCA correlation functions to reconstruct bond motion','Source: PCA module']
            Type='PCAbond'
        elif key=='sum':
            DataObj=clsDict['Data']
            loc='Ctsum'
            details=['Sum of PCA correlations functions to reconstruct bond motion','Source: PCA module']
            Type='PCAbond'
        elif key=='PCA':
            DataObj=Data_PCA
            loc='Ct'
            details=[f'PCA correlation functions ({self.pca.nPC} principal components)','Source: PCA module']
            Type='PCAmode'
        else:
            assert 0,f'Unrecognized correlation function: {key}'
        
        Ct=getattr(self.pca.Ct,loc)
        
        sens=self.sens
        out=DataObj(R=Ct,
                     Rstd=np.repeat(np.array([sens.info['stdev']],dtype=dtype),Ct.shape[0],axis=0),
                     sens=sens,Type=Type)
        out._pca=self.pca
        out.source.filename=self.pca.select.traj.files
        out.source.status='raw'
        out.details=self.pca.select.details
        out.details.extend(details)
    
        out.label=np.arange(out.R.shape[0],dtype=int) if key=='PCA' else self.pca.select.label
        
        out.detect=self.detect
        
        self.project.append_data(out)
        out.select=copy(self.pca.select)
        
        if key not in self._locs:self._locs[key]={}
        self._locs[key]['raw']=len(self.project)-1,out.title
        
        return self.project[-1]
    
    def _no_opt(self,key):
        if key in self._locs and 'no_opt' in self._locs[key]:return self._find(key,'no_opt')
        if key not in self._locs:self._raw(key)
        self._find(key,'raw').fit()
        
        self._locs[key]['no_opt']=len(self.project)-1,self.project[-1].title
        return self.project[-1]
        
    def _proc(self,key):
        if key in self._locs and 'proc' in self._locs[key]:return self._find(key,'proc')
        if key not in self._locs or 'no_opt' not in self._locs[key]:self._no_opt(key)
        data=self._find(key,'no_opt')
        data.detect.r_auto(self.n)
        data.fit()
        
        self._locs[key]['proc']=len(self.project)-1,self.project[-1].title
        return self.project[-1]
        
    def _opt_fit(self,key):
        if key in self._locs and 'opt_fit' in self._locs[key]:return self._find(key,'opt_fit')
        if key not in self._locs or 'proc' not in self._locs[key]:self._proc(key)
        data=self._find(key,'proc')
        data.opt2dist(rhoz_cleanup=True)
        
        self._locs[key]['opt_fit']=len(self.project)-1,self.project[-1].title
        return self.project[-1]
    
    def direct(self,status='proc'):
        getattr(self,'_'+status)('direct')
        return self._find('direct',status)
    
    def prod(self,status='proc'):
        getattr(self,'_'+status)('prod')
        return self._find('prod',status)
    
    def sum(self,status='proc'):
        getattr(self,'_'+status)('sum')
        return self._find('sum',status)
    
    def PCA(self,status='proc'):
        getattr(self,'_'+status)('PCA')
        return self._find('PCA',status)
    
    def prod_v_direct(self,errorbars=False, style='canvas', fig=None, index=None, 
             rho_index=None, plot_sens=True, split=True,status='proc',**kwargs):
        """
        Plot comparing

        Parameters
        ----------
        errorbars:  Show the errorbars of the canvas (False/True or int)
                    (default 1 standard deviation, or insert a constant to multiply the stdev.)
        style:      Plot style ('canvas','scatter','bar')
        fig:        Provide the desired figure object (matplotlib.pyplot.figure) or provide
                    an integer specifying which of the project's figures to append
                    the canvas to (if data attached to a project)
        index:      Index to specify which residues to canvas (None or logical/integer indx)
        rho_index:  Index to specify which detectors to canvas (None or logical/integer index)
        plot_sens:  Plot the sensitivity as the first canvas (True/False)
        split:      Break the plots where discontinuities in data.label exist (True/False)  
        status:     Which status to plot ('proc','no_opt','opt_fit')
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        po : TYPE
            DESCRIPTION.

        """
        assert status!='raw','Raw data plotting not currently supported'
        
        po=self.direct(status).plot(errorbars=errorbars, style=style, fig=fig, index=index, 
             rho_index=rho_index, plot_sens=plot_sens, split=split,**kwargs)
        
        self.prod(status).plot(errorbars=errorbars, style=style, fig=fig, index=index, 
             rho_index=rho_index, plot_sens=plot_sens, split=split,**kwargs)
        
        
        return po
    
    def prod_v_sum(self,errorbars=False, style='canvas', fig=None, index=None, 
             rho_index=None, plot_sens=True, split=True,status='proc',**kwargs):
        """
        Plot comparing

        Parameters
        ----------
        errorbars:  Show the errorbars of the canvas (False/True or int)
                    (default 1 standard deviation, or insert a constant to multiply the stdev.)
        style:      Plot style ('canvas','scatter','bar')
        fig:        Provide the desired figure object (matplotlib.pyplot.figure) or provide
                    an integer specifying which of the project's figures to append
                    the canvas to (if data attached to a project)
        index:      Index to specify which residues to canvas (None or logical/integer indx)
        rho_index:  Index to specify which detectors to canvas (None or logical/integer index)
        plot_sens:  Plot the sensitivity as the first canvas (True/False)
        split:      Break the plots where discontinuities in data.label exist (True/False)  
        status:     Which status to plot ('proc','no_opt','opt_fit')
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        po : TYPE
            DESCRIPTION.

        """
        assert status!='raw','Raw data plotting not currently supported'
        
        po=self.sum(status).plot(errorbars=False, style='canvas', fig=None, index=None, 
             rho_index=None, plot_sens=True, split=True,**kwargs)
        
        self.prod(status).plot(errorbars=False, style='canvas', fig=None, index=None, 
             rho_index=None, plot_sens=True, split=True,**kwargs)
        
        
        return po
    


Data=clsDict['Data_iRED']
class Data_PCA(Data):
    def __init__(self, R=None, Rstd=None, label=None, sens=None, select=None, 
                 src_data=None, Type=None,S2=None, S2std=None, Rc=None):
        
        super().__init__(R=R,Rstd=Rstd,label=label,sens=sens,select=select,
                         src_data=src_data,Type=Type,S2=S2,S2std=S2std,Rc=Rc)
        self._pca=None
        if self.src_data is not None:
            self._pca=self.src_data.pca
        
        self._CC=None
        self._CCnorm=None
    

        
    
    def opt2dist(self,rhoz=None,rhoz_cleanup:bool=False,parallel:bool=False):
        """
        Forces a set of detector responses to be consistent with some given distribution
        of motion. Achieved by performing a linear-least squares fit of the set
        of detector responses to a distribution of motion, and then back-calculating
        the detectors from that fit. Set rhoz_cleanup to True to obtain monotonic
        detector sensitivities: this option eliminates unusual detector due to 
        oscilation and negative values in the detector sensitivities. However, the
        detectors are no longer considered "DIstortion Free".
                                
    
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        rhoz_cleanup : TYPE, optional
            DESCRIPTION. The default is False. If true, we use a threshold for cleanup
            of 0.1, although rhoz_cleanup can be set to a value between 0 and 1 to
            assign the threshold manually
    
        Returns
        -------
        data object
    
        """
        # print('checkpoint')
        out=super().opt2dist(rhoz=rhoz,rhoz_cleanup=rhoz_cleanup,parallel=parallel)
        out._pca=self.pca
       
        
        return out
    
    @property
    def CC(self):
        if self.source.Type!='PCAbond':
            return None
        
        if self._CC is None:
            self._CC=np.zeros([self.R.shape[1],self.R.shape[0],self.R.shape[0]],dtype=dtype)
            for k in range(self._CC.shape[0]):
                self._CC[k]=self.pca.S2.AmCC.T@self.src_data.R[:,k]
        
        return self._CC
        
        #     out.CC=np.zeros([out.R.shape[1],out.R.shape[0],out.R.shape[0]],dtype=dtype)
        #     for k in range(out.CC.shape[-1]):
        #         A=self.pca.S2.Am
        #         A[A<0]=0
        #         A0=self.R.T*np.sqrt(A[:,k])
        #         out.CC[:,:,k]=A0@np.sqrt(A)
        #     out.CC[out.CC<0]=0
    
    # @property
    # def CCnorm(self) -> np.ndarray:
    #     """
    #     Calculates and returns the normalized cross-correlation matrices for
    #     each detector

    #     Returns
    #     -------
    #     np.ndarray

    #     """
    #     if self.CC is None:
    #         print('Warning: Cross-correlation not calculated for this data object')
    #         return

    #     if self._CCnorm is None:
    #         self._CCnorm=np.zeros(self.CC.shape,dtype=dtype)
    #         for k,CC in enumerate(self.CC):
    #             dg=np.sqrt([np.diag(CC)])
    #             self._CCnorm[k]=CC/(dg.T@dg)
    #         # self._CCnorm[np.isnan(self._CCnorm)]=0
    #         # self._CCnorm[np.isinf(self._CCnorm)]=0
    #     return self._CCnorm
    
    @property
    def pca(self):
        return self._pca
        
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
    
    
    def modes2bonds(self) -> Data:
        """
        Converts PCA mode detector responses into bond-specific detector 
        responses, including calculation of cross-correlation matrices for each 
        detector. These are stored in CC and CCnorm, where CC is the unnormalized
        correlation and CCnorm is the correlation coefficient, i.e. Pearson's r

        Returns
        -------
        data_PCA

        """
        
        out=self.__class__(sens=self.sens,src_data=self)
        out.details.extend(self.details)
        out.details.append('Converted from PCA modes to PCA bond data')
        out.R=self.pca.S2.Am.T@self.R
        S2=1-self.pca.S2.Am.sum(0)
        out.R+=np.array([rhoz[-1]*S2 for rhoz in self.sens.rhoz]).T
        out.Rstd=np.sqrt(self.pca.S2.Am.T@self.Rstd**2)
        
        out.source.Type='PCAbond'
        out.source.n_det=self.source.n_det
        out.label=self.pca.select.label
        
        if self.source.project is not None:self.source.project.append_data(out)
        
        return out
    
    # def CCchimera(self,index=None,rho_index:int=None,indexCC:int=None,scaling:float=None,norm:bool=True) -> None:
    #     """
    #     Plots the cross correlation of motion for a given detector window in 
    #     chimera. 

    #     Parameters
    #     ----------
    #     index : list-like, optional
    #         Select which residues to plot. The default is None.
    #     rho_index : int, optional
    #         Select which detector to initially show. The default is None.
    #     indexCC : int,optional
    #         Select which row of the CC matrix to show. Must be used in combination
    #         with rho_index. Note that if index is also used, indexCC is applied
    #         AFTER index.
    #     scaling : float, optional
    #         Scale the display size of the detectors. If not provided, a scaling
    #         will be automatically selected based on the size of the detectors.
    #         The default is None.
    #     norm : bool, optional
    #         Normalizes the data to the amplitude of the corresponding detector
    #         responses (makes diagonal of CC matrix equal to 1).
    #         The default is True

    #     Returns
    #     -------
    #     None

    #     """
        
    #     CMXRemote=clsDict['CMXRemote']

    #     index=np.arange(self.R.shape[0]) if index is None else np.array(index)

    #     if rho_index is None:rho_index=np.arange(self.R.shape[1])
    #     if not(hasattr(rho_index, '__len__')):
    #         rho_index = np.array([rho_index], dtype=int)
    #     # R = self.CCnorm[:,index,bond_index].T
    #     #TODO add some options for including the sign of the correlation (??)
    #     R=np.abs(getattr(self,'CCnorm' if norm else 'CC')[:,index][:,:,index].T)
    #     R *= 1/R.T[rho_index].max() if scaling is None else scaling
    #     # R[R < 0] = 0 

    #     if self.source.project is not None:
    #         ID=self.source.project.chimera.CMXid
    #         if ID is None:
    #             self.source.project.chimera.current=0
    #             ID=self.source.project.chimera.CMXid
    #             print(ID)
    #     else: #Hmm....how should this work?
    #         ID=CMXRemote.launch()
    #         cmds=[]


    #     ids=np.array([s.indices for s in self.select.repr_sel[index]],dtype=object)


    #     if len(rho_index)==1 and indexCC is not None:
    #         x=R[indexCC].squeeze()[:,rho_index].squeeze()
    #         self.select.chimera(color=plt.get_cmap('tab10')(rho_index[0]),x=x,index=index)
    #         sel0=self.select.repr_sel[index][indexCC]
    #         if hasattr(sel0,'size'):sel0=sel0[0] #sel0 may still be a np.array
    #         mn=CMXRemote.valid_models(ID)[-1]
    #         CMXRemote.send_command(ID,'color '+'|'.join(['#{0}/{1}:{2}@{3}'.format(mn,s.segid,s.resid,s.name) for s in sel0])+' black')
    #         # print('color '+'|'.join(['#{0}/{1}:{2}@{3}'.format(mn,s.segid,s.resid,s.name) for s in sel0])+' black')
    #         return sel0
    #     else:

    #         self.select.chimera()
    #         mn=CMXRemote.valid_models(ID)[-1]
    #         CMXRemote.send_command(ID,f'color #{mn} tan')
                
            
    #         out=dict(R=R,rho_index=rho_index,ids=ids)
    #         CMXRemote.add_event(ID,'DetCC',out)
        
    #         if self.source.project is not None:
    #             self.source.project.chimera.command_line(self.source.project.chimera.saved_commands)
        
                
        
        
        
        
        
        