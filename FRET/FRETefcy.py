#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:41:16 2021

@author: albertsmith
"""

import numpy as np
from pyDR.FRET import FRETdipole
from pyDR import MolSys

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
    
    
    def __init__(self,molsys=None,**kwargs):
        self.f_rD=None  #Function defining the center of the Donor dipole
        self.f_rA=None  #Function defining the center of the Acceptor dipole
        self.f_vD=None  #Function defining the direction of the Donor dipole
        self.f_vA=None #Function defining the direction of the Acceptor dipole
        
        self.__r=None #Storage for the dipole centers, etc.
        self.__t=None #Storage for the time axis
        self.__index=None #Storage for the time axis index
        
        self.bin_index=None #Index to average over particular time points
        
        self.Na=6.0221409e23
        
        """Default parameters (often set to 1, where these influence the absolute efficiency,
        but not the relative efficiency, such that they do not influence the correlation functions)
        """
        self.thetaD=1   #Donor quantum yield
        self.n=1.4      #Refractive index
        self.Jlambda=1  #Integral of spectral overlap of donor emission and acceptor absorption
        self.tauD=1     #Fluoresent lifetime of the donor / ns
        
        
        for k,v in kwargs:  #Overwrite default values if provided
            if hasattr(self,k):setattr(self,k,v) #Ignore values that aren't provided
            
        self.__R0=None #Storage for R0 so we don't have to keep calculating it
        self.__Et=None #Storage for Et 
        self.__tauDAt=None #Storage for tauDAt 
        
        self.isotropic_avg=False #Set to true if using the isotropic average of kappa
        
        self.__vecs_up2date=False    #Set to False if one of the function is changed after loading
        
        self.molsys=molsys if molsys else MolSys() #molsys object
    
    def set_dipole(self,Type,**kwargs):
        assert hasattr(FRETdipole,Type),'Dipole definition ({}) not found in FRETdipole.py'.format(Type)
        out=getattr(FRETdipole,Type)(self.molsys,**kwargs)
        assert len(out)==4 or len(out)==2,'Dipole definition should return two or four arguments'
        for o in out:
            try:
                result=o()
                assert result.ndim==2 and result.shape[0]==3,'Dipole definition failed'
            except:
                assert False,'Dipole definition failed'
        if len(out)==4:
            self.f_rD,self.f_rA,self._vD,self._vA=out
        else:
            self.f_rD,self.f_rA=out
    
    def load(self,t0=0,tf=-1,n=10,nr=10,dt=None,index=None):
        """
        Load the vectors from a trajectory stored in the molsys object
        
        Currently, sparse sampling of trajectory unused (n,nr unused)
        """
        traj=self.molsys.mda_object.trajectory
        
        tf=traj.n_frames if (tf==-1 or tf is None) else int(tf)  #Final frame to use
        index=np.arange(t0,tf)
        dt=dt if dt else traj.dt
        
        if self.f_vD is None or self.f_vA is None:
            print("""Warning: If vectors defining the directions of the dipole are not defined,
then we use the isotropic average, kappa=2/3""")
            self.isotropic_avg=True
            
        
        assert self.f_rD is not None and self.f_rA is not None,\
            "Functions defining the center of the dipoles must be provided"
            
        if self.__r is None or self.__updated_vecs:
            self.__t=index*dt
            self.__index=index
            self.__r={'rD':list(),'rA':list(),'vD':list(),'vA':list()}
            for _ in traj[index]:
                self.__r['rD'].append(self.f_rD())
                self.__r['rA'].append(self.f_rA())
                if not(self.isotropic_avg):
                    self.__r['vD'].append(self.f_vD())
                    self.__r['vA'].append(self.f_vA())
                    
            for k,v in self.__r:
                self.__r[k]=np.array(v)
            if not(self.isotropic_avg):
                self.__r['vD']/=np.sqrt((self.__r['vD']**2).sum(0))
                self.__r['vA']/=np.sqrt((self.__r['vA']**2).sum(0))
            self.__vecs_up2date
    
    @property
    def rD(self):
        "Position of the donor"
        assert self.__vecs_up2date,"First load the trajectory"
        return self.__r['rD']
    @property
    def rA(self):
        "Position of the acceptor"
        assert self.__vecs_up2date,"First load the trajectory"
        return self.__r['rA']
    @property
    def vD(self):
        "Orientation of the donor"
        assert self.__vecs_up2date,"First load the trajectory"
        assert self.__r['vD'].size>0,"Donor vector not loaded"
        return self.__r['vD']
    @property
    def vA(self):
        "Orientation of the acceptor"
        assert self.__vecs_up2date,"First load the trajectory"
        assert self.__r['vA'].size>0,"Acceptor vector not loaded"
        return self.__r['vA']
    @property
    def t(self):
        "Time axis"
        assert self.__t and self.__vecs_up2date,"First load the trajectory to obtain the time axis"
        return self.__t
    @property
    def index(self):
        "Sampling index"
        assert self.__index and self.__vecs_up2date,"First load the trajectory to obtain the index"
        return self.__index
    @property
    def kappa2(self):
        "Orientation factor"
        assert self.__vecs_up2date,"First load the trajectory"
        if self.isotropic_avg:return 2/3
        vAD=self.__r['rA']-self.__r['rD']
        vAD/=np.sqrt((vAD**2).sum(0))    #Normalize vector
        vA,vD=self.__r['vA'],self.__r['vD']
        return ((vA*vD).sum(0)-3*(vA*vAD).sum(0)*(vD*vAD).sum(0))**2
    @property
    def R0(self):
        "Foerster radius"
        if self.__R0 is None or not(self.__vecs_up2date):
            self.__R0=(9/128*np.log(10)/np.pi**5*6.022e23*self.kappa2*self.thetaD*self.Jlambda/self.n**4)**(1/6)*1e9 #nm
        return self.__R0
    @property
    def Et(self):
        "FRET efficiency as a function of time"
        if self.__Et is None or not(self.__vecs_up2date):
            self.__Et=1/(1+((self.rA-self.rD)**2).sum(0)**3/self.R0**6)
        return self.__Et
    @property
    def _bi(self):
        assert self.__index and self.__vecs_up2date,"First load the trajectory"
        if self.bin_index is None:return self.ones(self.index.shape,dtype=bool)
        assert hasattr(self.bin_index,'__len__') and len(self.bin_index)==2,"bin_index should have 2 entries (start,stop)"
        return self.index>=self.bin_index[0] and self.index<self.bin_index[1]
        
    @property
    def E(self):
        return self.Et[self._bi].mean(0) #Average over time 
    @property
    def tauADt(self):
        if self.__tauDAt is None or not(self.__vecs_up2date):
            self.__tauDAt=(1-self.E)*self.tauD
        return self.__tauDAt
    @property
    def sigma_c2(self):
        kD=1/self.tauD
        kET=1/self.tauADt[self._bi]
        return (kET**2/(kD+kET)**2).mean(0)-(kET/(kD+kET)).mean(0)**2
    @property
    def tauAD(self):
        return 1-self.E+self.sigma_c2/(1)
    
    def Ebin(self,nbins):
        step=np.floor(self.index[-1]/nbins)
        E=list()
        for k in range(0,self.index[-1],step):
            self.bin_index=[k,k+step]
            E.append(self.E)
        return np.array(E)
    def tauABbin(self,nbins):
        step=np.floor(self.index[-1]/nbins)
        tauAB=list()
        for k in range(0,self.index[-1],step):
            self.bin_index=[k,k+step]
            tauAB.append(self.tauAB)
        return np.array(tauAB)
            
        