#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:41:16 2021

@author: albertsmith
"""

import numpy as np
from pyDR import clsDict
from pyDR.FRET import FRETdipole
from pyDR.MDtools import Ctcalc as Ct_calc
import matplotlib.pyplot as plt

class FRETefcy():
    """
    Object for calculating a variety of parameters relevant to FRET transfer 
    efficiencies. 
    
    Formulae taken from:
    A. Barth,..., C.A.M. Seidel. Unraveling multi-state molecular dynamics in 
    single-molsys FRET experiments. arXiv. 2021
    https://arxiv.org/abs/2107.14770v1
    
    Orientation dependence (kappa):
    H. Kashida, H. Asanuma. Orientation-dependent FRET system reveals differences
    in structures and flexibilities of nicked and gapped DNA duplexes. 
    Nucleic Acids Research. 2017, 45, e105. 
    https://doi.org/10.1093/nar/gkx200
    
    Dimensional orders are
    0: x,y,z (if coordinate)
    1: time
    2: FRET pair index
    """
    
    
    def __init__(self,molsys=None,R0=None,**kwargs):
        self.f_rD=None  #Function defining the center of the Donor dipole
        self.f_rA=None  #Function defining the center of the Acceptor dipole
        self.f_vD=None  #Function defining the direction of the Donor dipole
        self.f_vA=None #Function defining the direction of the Acceptor dipole
        
        self.__r=None #Storage for the dipole centers, etc.
        self.__t=None #Storage for the time axis
        self.__index=None #Storage for the time axis index
        
        self.bin_index=None #Index to average over particular time points
        
        
        """Default parameters (often set to 1, where these influence the absolute efficiency,
        but not the relative efficiency, such that they do not influence the correlation functions)
        """
        self.ThetaD=1   #Donor quantum yield
        self.n=1.4      #Refractive index
        self.R0iso=R0 if R0 else 5      #Forster radius (in nm, for kappa2=2/3, i.e. isotropic donor/acceptor reorientation)
        self.tauD=1     #Fluoresent lifetime of the donor / ns
        
        self.Dt=None
        self.Dt_tauDA=None
        
        for k,v in kwargs:  #Overwrite default values if provided
            if hasattr(self,k):setattr(self,k,v) #Ignore values that aren't provided
            
        self.__R0=None #Storage for R0 so we don't have to keep calculating it
        self.__Et=None #Storage for Et 
        self.__tauDAt=None #Storage for tauDAt 
        
        self.isotropic_avg=False #Set to true if using the isotropic average of kappa
        
        self.__up2date=False    #Set to False if one of the functions is changed after loading
        self.__vecs_up2date=False
        
        self.__g={'gAA':None,'gDD':None,'gDA':None,'gAD':None}
        
        self._ctnorm=True
        self._label=None
        
        self.molsys=molsys if molsys else clsDict['MolSys']() #molsys object
    
    def __setattr__(self,name,value):
        if name in ['R0iso','tauD','n','ThetaD','isotropic_avg']:
            self.__up2date=False
        if name=='label':name,value='_label',np.array(value)
        super().__setattr__(name,value)
        
        
    
    def set_dipole(self,Type,**kwargs):
        assert hasattr(FRETdipole,Type),'Dipole definition ({}) not found in FRETdipole.py'.format(Type)
        out=getattr(FRETdipole,Type)(self.molsys,**kwargs)
        assert len(out)==4 or len(out)==2,'Dipole definition should return two or four arguments'
        for o in out:
            try:
                result=o()
            except:
                assert False,'Dipole function does not run'
            assert result.ndim==2 and result.shape[1]==3,'Dipole function returns incorrect sizes'
        if len(out)==4:
            self.f_rD,self.f_rA,self.f_vD,self.f_vA=out
        else:
            self.f_rD,self.f_rA=out
        self.__vecs_up2date=False
    
    def load(self,t0=0,tf=-1,n=10,nr=10,dt=None,index=None):
        """
        Load the vectors from a trajectory stored in the molsys object
        
        Currently, sparse sampling of trajectory unused (n,nr unused)
        """
        traj=self.molsys.traj
        
        tf=len(traj) if (tf==-1 or tf is None) else int(tf)  #Final frame to use
        index=np.arange(t0,tf,dtype=int)
        dt=dt if dt else traj.dt/(1e3 if traj.mda_traj.units['time']=='ps' else 1)
        
        if self.f_vD is None or self.f_vA is None:
            print("""Warning: If vectors defining the directions of the dipole are not defined,
then we use the isotropic average, kappa=2/3""")
            self.isotropic_avg=True
            
        
        assert self.f_rD is not None and self.f_rA is not None,\
            "Functions defining the center of the dipoles must be provided"
            
        if self.__r is None or not(self.__vecs_up2date):
            self.__t=index*dt
            self.__index=index
            self.__r={'rD':list(),'rA':list(),'vD':list(),'vA':list()}
            traj.ProgressBar=True
            for _ in traj[index]:
                self.__r['rD'].append(self.f_rD())
                self.__r['rA'].append(self.f_rA())
                if not(self.isotropic_avg):
                    self.__r['vD'].append(self.f_vD())
                    self.__r['vA'].append(self.f_vA())
            traj.ProgressBar=False
                    
            for k,v in self.__r.items():
                self.__r[k]=np.array(v).T
            if not(self.isotropic_avg):
                self.__r['vD']/=np.sqrt((self.__r['vD']**2).sum(0))
                self.__r['vA']/=np.sqrt((self.__r['vA']**2).sum(0))
            self.__vecs_up2date=True
    
    def recalculate(self):
        """
        Force stored parameters to be recalculated on next call
        """
        self.__up2date=False
        
    @property
    def _up2date(self):
        if not(self.__up2date):
            self.__g={'gAA':None,'gDD':None,'gDA':None,'gAD':None}
            self.Dt=None
            self.Dt_tauDA=None
            self.__Et=None
            self.__R0=None
            self.__tauDAt=None
            
        return self.__up2date and self.__vecs_up2date
    @property
    def NA(self):
        return 6.0221409e23
    @property
    def Jlambda(self):
        "Overlap of the donor emission/acceptor absorption normalized spectra, int(F(lambda)*epsilon(lambda)*lambda^4dlambda)"
        return (self.R0iso/1e9)**6*128*np.pi**5*self.NA*self.n**4/(9*np.log(10)*2/3*self.ThetaD)    
    
    @property
    def label(self):
        if self._label is not None:
            if self._up2date and len(self._label)!=len(self.E):
                print('Warning: label has the wrong length')
                return np.arange(len(self.E))
            return self._label
        return np.arange(len(self.E))

    @property
    def rD(self):
        "Position of the donor"
        assert self.__vecs_up2date,"First load the trajectory"
        return self.__r['rD'].copy()
    @property
    def rA(self):
        "Position of the acceptor"
        assert self.__vecs_up2date,"First load the trajectory"
        return self.__r['rA'].copy()
    @property
    def vD(self):
        "Orientation of the donor"
        assert self.__vecs_up2date,"First load the trajectory"
        assert self.__r['vD'].size>0,"Donor vector not loaded"
        return self.__r['vD'].copy()
    @property
    def vA(self):
        "Orientation of the acceptor"
        assert self.__vecs_up2date,"First load the trajectory"
        assert self.__r['vA'].size>0,"Acceptor vector not loaded"
        return self.__r['vA'].copy()
    @property
    def vDA(self):
        "Vector between dipoles (rD-rA)"
        return self.rD-self.rA
    @property
    def t(self):
        "Time axis"
        assert self.__t is not None and self.__vecs_up2date,"First load the trajectory to obtain the time axis"
        return self.__t
    @property
    def index(self):
        "Sampling index"
        assert self.__index is not None and self.__vecs_up2date,"First load the trajectory to obtain the index"
        return self.__index
    @property
    def kappa2(self):
        "Orientation factor"
        assert self.__vecs_up2date,"First load the trajectory"
        if self.isotropic_avg:return 2/3
        vDA=self.vDA/np.sqrt((self.vDA**2).sum(0))
        vA,vD=self.__r['vA'],self.__r['vD']
        return ((vA*vD).sum(0)-3*(vA*vDA).sum(0)*(vD*vDA).sum(0))**2
    @property
    def R0(self):
        "Foerster radius"
        if self.__R0 is None or not(self._up2date):
            self.__R0=self.R0iso*(self.kappa2/(2/3))**(1/6)
        return self.__R0
    @property
    def Et(self):
        "FRET efficiency as a function of time"
        if self.__Et is None or not(self._up2date):
            self.__Et=1/(1+((self.rA-self.rD)**2).sum(0)**3/self.R0**6)
            self.__up2date=True
        return self.__Et
    @property
    def _bi(self):
        assert self.__index is not None and self.__vecs_up2date,"First load the trajectory"
        if self.bin_index is None:return np.ones(self.index.shape,dtype=bool)
        assert hasattr(self.bin_index,'__len__') and len(self.bin_index)==2,"bin_index should have 2 entries (start,stop)"
        return np.logical_and(self.index>=self.bin_index[0],self.index<self.bin_index[1])
        
    @property
    def E(self):
        return self.Et.T[self._bi].mean(0) #Average over time 
    @property
    def tauDAt(self):
        if self.__tauDAt is None or not(self._up2date):
            self.__tauDAt=(1-self.Et)*self.tauD
        return self.__tauDAt
    @property
    def sigma_c2(self):
        kD=1/self.tauD
        kET=1/self.tauDAt.T[self._bi]
        return (kET**2/(kD+kET)**2).mean(0)-(kET/(kD+kET)).mean(0)**2
    @property
    def tauDA(self):
        return self.tauDAt.mean(-1)
    
    def Ebin(self,Dt:float=None):
        """
        E averaged over a bin of length Dt. If Dt is not given, then Dt will
        correspond to 100*dt (dt being the timestep).

        Parameters
        ----------
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt

        Returns
        -------
        np.array
            2D array of E for bins of length Dt.

        """
        # step=np.floor(self.index[-1]/nbins).astype(int)
        if (Dt is None or Dt==self.Dt) and self.Dt is not None :
            return self._Ebin  #Just take the stored value
        
        if Dt is None:Dt=(self.t[1]-self.t[0])*100
        
        step=int(Dt/(self.t[1]-self.t[0]))
        nbins=len(self.t)//step
        
        
        # A=np.zeros((nbins,self.Et.shape[-1]),dtype=bool)
        # for k in range(nbins):A[k,k*step:(k+1)*step]=True
        
        # E=np.array([(Et*A).sum(1)/step for Et in self.Et])

        E=np.zeros([self.Et.shape[0],nbins],dtype=self.Et.dtype)
        for k in range(step):
            E+=self.Et[:,k:nbins*step:step]
        E/=step
        self._Ebin=E
        self.Dt=Dt
        return E
    
    def tauDAbin(self,Dt:float=None):
        """
        E averaged over a bin of length Dt. If Dt is not given, then Dt will
        correspond to 100*dt (dt being the timestep).

        Parameters
        ----------
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt

        Returns
        -------
        np.array
            2D array of E for bins of length Dt.

        """
        # step=np.floor(self.index[-1]/nbins).astype(int)
        if (Dt is None or Dt==self.Dt_tauDA) and self.Dt_tauDA:
            return self._tauDAbin  #Just take the stored value
        
        if Dt is None:Dt=(self.t[1]-self.t[0])*100
        
        step=int(Dt/(self.t[1]-self.t[0]))
        nbins=len(self.t)//step

        tauDA=np.zeros([self.Et.shape[0],nbins],dtype=self.Et.dtype)
        for k in range(step):
            tauDA+=self.tauDAt[:,k:nbins*step:step]
        tauDA/=step
        self._tauDAbin=tauDA
        self.Dt_tauDA=Dt
        return tauDA
    
    def Ehist(self,Dt:float=None,nbins:int=None,bin_range:tuple=None,index=None):
        """
        Returns an axis and histogram for E

        Parameters
        ----------
        index : list, optional
            List of FRET pairs to calculate
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt
        nbins : int, optional
            Number of bins in the histogram. The default is None.
        bin_range : tuple, optional
            Range of values (min,max) for which to calculate the histogram. The 
            default is None, which defaults to the minimum and maximum 
            efficiencies found.

        Returns
        -------
        tuple (np.array,np.array)
            Returns the x-axis (E) and the count for each value on the x-axis

        """
        
        if index is None:index=np.arange(len(self.E))
        index=np.atleast_1d(index)
        
        E=self.Ebin(Dt=Dt)[index]
        if nbins is None:nbins=E.shape[1]//20
        if bin_range is None:bin_range=(E.min(),E.max())
        
        bins=np.linspace(*bin_range,nbins+1)
        count=np.array([np.histogram(E0,bins)[0] for E0 in E])
        
        return (bins[:-1]+bins[1:])/2,count

    def tauDAhist(self,Dt:float=None,nbins:int=None,bin_range:tuple=None,index=None):
        """
        Returns an axis and histogram for tauDA

        Parameters
        ----------
        index : list, optional
            List of FRET pairs to calculate
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt
        nbins : int, optional
            Number of bins in the histogram. The default is None.
        bin_range : tuple, optional
            Range of values (min,max) for which to calculate the histogram. The 
            default is None, which defaults to the minimum and maximum 
            efficiencies found.

        Returns
        -------
        tuple (np.array,np.array)
            Returns the x-axis (E) and the count for each value on the x-axis

        """
        if index is None:index=np.arange(len(self.E))
        index=np.atleast_1d(index)
        
        tauDA=self.tauDAbin(Dt=Dt)[index]
        if nbins is None:nbins=tauDA.shape[1]//20
        if bin_range is None:bin_range=(tauDA.min(),tauDA.max())
        
        bins=np.linspace(*bin_range,nbins+1)
        count=np.array([np.histogram(t0,bins)[0] for t0 in tauDA])
        
        return (bins[:-1]+bins[1:])/2,count
    
    def E_tauDA_hist(self,Dt:float=None,nbins=50,index=None,tau_range:tuple=None,E_range:tuple=None):
        """
        2D histograme of both tauDA and E

        Parameters
        ----------
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt
        nbins : int, optional
            Number of bins in the histogram. The default is None.
        index : list, optional
            List of FRET pairs to calculate
        tau_range : tuple, optional
            Range of values (min,max) for which to calculate the histogram for 
            tauDA. The default is None, which defaults to the minimum and maximum.
        E_range : tuple, optional
            Range of values (min,max) for which to calculate the histogram for 
            E. The default is None, which defaults to the minimum and maximum.

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        H : TYPE
            DESCRIPTION.

        """
        if index is None:index=np.ones(self.E.shape[0],dtype=bool)
        E=self.Ebin(Dt=Dt)[index]
        if Dt is None:Dt=self.Dt if self.Dt is not None else self.Dt_tauDA
        tauDA=self.tauDAbin(Dt=Dt)[index]
        
        if tau_range is None:tau_range=tauDA.min(),tauDA.max()
        if E_range is None:E_range=E.min(),E.max()
        
        Range=tau_range,E_range
        
        H=list()
        for E0,tauDA0 in zip(E,tauDA):
            out=np.histogram2d(tauDA0,E0,nbins,range=Range)
            H.append(out[0])
            x=out[1]
            y=out[2]
        
        return (x[:-1]+x[1:])/2,(y[:-1]+y[1:])/2,np.array(H)


    def _g(self,x,y=None):
        """
        Performs the linear correlation function between x and itself or y

        Parameters
        ----------
        x : np.array
            2D data for which the correlation function is performed on the 2nd 
            dimension
        y : np.array, optional
            Optional 2D data (same size as x), for which the cross-correlation
            to x is calculated. The default is None.

        Returns
        -------
        ct : np.array
            Auto or cross-correlation function.

        """
        ctcalc=Ct_calc(sym='p0' if y is None else 'p0')
        ctcalc.a=x
        if y is not None:ctcalc.b=y
        ctcalc.add()
        ct=ctcalc.Return()[0].T
        if self._ctnorm:
            norm=x.mean(axis=-1)**2 if y is None else x.mean(axis=-1)*y.mean(axis=-1)
            ct-=norm
            ct/=norm
        return ct.T
    @property
    def gDD(self):
        if self.__g['gDD'] is None or not(self._up2date):
            self.__g['gDD']=self._g(self.ThetaD*(1-self.Et))
        return self.__g['gDD']
    @property
    def gAA(self):
        if self.__g['gAA'] is None or not(self._up2date):
            self.__g['gAA']=self._g(self.Et)
        return self.__g['gAA']
    @property
    def gDA(self):
        if self.__g['gDA'] is None or not(self._up2date):
            self.__g['gDA']=self._g(self.ThetaD*(1-self.Et),self.Et)
        return self.__g['gDA']
    @property
    def gAD(self):
        if self.__g['gAD'] is None or not(self._up2date):
            self.__g['gAD']=self._g(self.Et,self.ThetaD*(1-self.Et))
        return self.__g['gAD']
        
    
    #%% Plotting functions
    def plot_E_hist(self,index=None,Dt:float=None,nbins:int=None,bin_range:tuple=None,ax=None):
        """
        Plots the histograms for E

        Parameters
        ----------
        index : list, optional
            List of FRET pairs to display
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt
        nbins : int, optional
            Number of bins in the histogram. The default is None.
        bin_range : tuple, optional
            Range of values (min,max) for which to calculate the histogram. The 
            default is None, which defaults to the minimum and maximum 
            efficiencies found.
        ax : Type, optional
            Axis or list of axes on which to plot histogram(s)

        Returns
        -------
        Type
            handles for the bar plot object

        """
        
        
        
        if index is None:index=np.arange(len(self.E))
        index=np.atleast_1d(index)
        
        bins,count=self.Ehist(Dt=Dt,nbins=nbins,bin_range=bin_range,index=index)
        
        if ax is None:
            nfp=self.E[index].shape[0]
            sz=(np.ceil(np.sqrt(nfp))*np.ones(2)).astype(int)
            if (sz[0]-1)*sz[1]>=nfp:sz[0]-=1
            ax=plt.subplots(sz[0],sz[1])[1]
        ax=np.atleast_1d(np.array(ax).flatten())
        
        hdl=list()
        for a,c0,lbl in zip(ax,count,self.label[index]):
            hdl.append(a.bar(bins,c0,linestyle='-',edgecolor='black',linewidth=.5,width=bins[1]-bins[0]))
            if a.is_last_row():
                a.set_xlabel('E')
            else:
                a.set_xticklabels('')
            if a.is_first_col():
                a.set_ylabel('counts')
            else:
                a.set_yticklabels('')
            a.set_title(lbl)
                
        return hdl
    
    def plot_tauDA_hist(self,index=None,Dt:float=None,nbins:int=None,bin_range:tuple=None,ax=None):
        """
        Plots the histograms for tauDA

        Parameters
        ----------
        index : list, optional
            List of FRET pairs to display
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt
        nbins : int, optional
            Number of bins in the histogram. The default is None.
        bin_range : tuple, optional
            Range of values (min,max) for which to calculate the histogram. The 
            default is None, which defaults to the minimum and maximum 
            efficiencies found.
        ax : Type, optional
            Axis or list of axes on which to plot histogram(s)

        Returns
        -------
        Type
            handles for the bar plot object

        """
        
        
        
        if index is None:index=np.arange(len(self.E))
        index=np.atleast_1d(index)
        
        bins,count=self.tauDAhist(Dt=Dt,nbins=nbins,bin_range=bin_range,index=index)
        
        if ax is None:
            nfp=self.E[index].shape[0]
            sz=(np.ceil(np.sqrt(nfp))*np.ones(2)).astype(int)
            if (sz[0]-1)*sz[1]>=nfp:sz[0]-=1
            ax=plt.subplots(sz[0],sz[1])[1]
        ax=np.atleast_1d(np.array(ax).flatten())
        
        hdl=list()
        for a,c0,lbl in zip(ax,count,self.label[index]):
            hdl.append(a.bar(bins/self.tauD,c0,linestyle='-',edgecolor='black',linewidth=.5,width=bins[1]-bins[0]))
            if a.is_last_row() or len(index)==1:
                a.set_xlabel(r'$\tau/\tau_D$')
            else:
                a.set_xticklabels('')
            if a.is_first_col() or len(index)==1:
                a.set_ylabel('counts')
            else:
                a.set_yticklabels('')
            a.set_title(lbl)
                
        return hdl
    
    def plot_2D_hist(self,index=None,Dt:float=None,nbins:int=50,tau_range:tuple=[0,1],E_range:tuple=[0,1],ax=None):
        """
        Plots the histograms for tauDA

        Parameters
        ----------
        index : list, optional
            List of FRET pairs to display
        Dt : float, optional
            Length of averaging bin. The default is None, yielding 100*dt
        nbins : int, optional
            Number of bins in the histogram. The default is None.
        bin_range : tuple, optional
            Range of values (min,max) for which to calculate the histogram. The 
            default is None, which defaults to the minimum and maximum 
            efficiencies found.
        ax : Type, optional
            Axis or list of axes on which to plot histogram(s)

        Returns
        -------
        Type
            handles for the bar plot object

        """
        if index is None:index=np.arange(len(self.E))
        index=np.atleast_1d(index)
        x,y,H=self.E_tauDA_hist(index=index,Dt=Dt,nbins=nbins,tau_range=tau_range,E_range=E_range)
        
        if ax is None:
            nfp=self.E[index].shape[0]
            sz=(np.ceil(np.sqrt(nfp))*np.ones(2)).astype(int)
            if (sz[0]-1)*sz[1]>=nfp:sz[0]-=1
            ax=plt.subplots(sz[0],sz[1])[1]
        ax=np.atleast_1d(np.array(ax).flatten())
            
        hdl=list()
        for a,c0,lbl in zip(ax,H,self.label[index]):
            hdl.append(a.contourf(y/self.tauD,x,c0.T,cmap='jet'))
            if a.is_last_row() or len(index)==1:
                a.set_xlabel('E')
            else:
                a.set_xticklabels('')
            if a.is_first_col() or len(index)==1:
                
                a.set_ylabel(r'$\tau/\tau_{D}$')
            else:
                a.set_yticklabels('')
            a.set_title(lbl)
                
        return hdl
            
        

    

            
        