#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:23:01 2022

@author: albertsmith
"""

from pyDR.Sens import NMR
from pyDR.Sens.NMR import defaults
import numpy as np
from pyDR.misc.tools import linear_ex

class SolnNMR(NMR):
    def __init__(self,tc=None,z=None,info=None,**kwargs):
        """
        Initializes a solution-state NMR sensitivity object. 

        Parameters
        ----------
        tc : list-like, optional
            Correlation time. Can be provided as a vector with 2, 3, or N elements.
            2 elements: Defines the first and last correlation time in the 
                        vector of correlation times (200 elements total)
            3 elements: Defines first, last, and number of correlation times
            N elements: List of the correlation times to be used
            
            The default is None, in which case 200 elements from 10 fs to 1 ms
            are used.
        z : list-like, optional
            Alternative to inputing tc, where we provide log-correlation times
            instead of the correlation times themselves.
            The default is None. See above for default behavior
        info : TYPE, optional
            Provide the info object directly for defining the experiments.
            The default is None.
        **kwargs : TYPE
            Keyword arguments for defining the NMR experiments. Each experiment
            should include an argument for tM for solution-state NMR. SolnNMR
            may be initialized without any experiments included, so tM is in
            principle optional.
            
            tM : Rotational correlation time of the molecule ()
            zeta_rot : Anisotropy of rotational diffusion tensor (optional)
            eta_rot : Rhombicity of rotational diffusion tensor
            
            See:
            Ghose, Fushman, Cowburn. J. Magn. Reson. (2001), 149, 204-217 for
            definitions used for anisotropic diffusion tensors
            (THIS IS NOT IMPLEMENTED YET!!)
            
            
            

        Returns
        -------
        None.

        """
        super().__init__(tc=tc,z=z)
        for par in ['tM','zeta_rot','eta_rot','euler']:self.info.new_parameter(par)
        
        self.new_exper(info=info,**kwargs)
        
        
        
        self.index=None
        self.vecs=None
        self.vecsCSA=None
        
        #iso will store the sensitivities corresponding to isotropic tumbling
        #bonds will store bond-specific sensitivities (if existing)
        self._bonds={'iso':None,'bonds':list()}
    
    def new_exper(self,info=None,**kwargs):
        """
        Define new experiments using a list of parameters. 
        For each parameter, one should provide either
        a single entry or a list of entries corresponding to all experiments 
        (single entry will be copied to all experiments). Note that providing a
        list of entries for dXY and/or Nuc1 will be interpreted as all experiments
        having multiple dipole-coupled nuclei. A list of lists will correct this
        behavior if what is desired is a different set of couplings for each 
        experiment.
        

        Parameters
        ----------
        info : pyDR.Sens.Info
            Append experiments defined by an Info object.
        **kwargs : TYPE
            Parameters describing the individual experiments.

        Returns
        -------
        None.

        """
        
        super().new_exper(info=info,**kwargs)
        #Set some defaults
        if 'zeta_rot' not in kwargs:  
            for k in range(len(self.info)):
                self.info['zeta_rot',k]=1
        if 'euler' not in kwargs:
            for k in range(len(self.info)):
                self.info['euler',k]=[0,0,0]
    
    def __getitem__(self,index):
        """
        Returns a copy of the SolnNMR object for the specific residue index

        Parameters
        ----------
        index : int
            Index of desired bond.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        if self.vecs is None:return self
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.index=index
        return out
    
    @property
    def vXY(self):
        """
        Mean direction of the dipole between spins X and Y for the current index.

        Returns
        -------
        None.

        """
        
        if self.index is None:return
        return self.vecs[self.index]

    @property
    def vCSA(self):
        """
        Mean direction of the CSA on spin X for the current index

        NOT CURRENTLY IMPLEMENTED!! Returns a vector along z

        Returns
        -------
        None.

        """
        if self.vecsCSA is None:return self.vXY
        
        return self.vecsCSA[self.index]
    
    def __len__(self):
        if self.vecs is None or self.index is not None:return 1
        return self.vecs.shape[0]
    
    
    def zeff(self,zM:float):
        """
        Returns the log-effective correlation time for the internal z vector,
        given the log-rotational correlation time, zM

        Parameters
        ----------
        zM : float
            log-rotational correlation time.

        Returns
        -------
        zeff, an array of the effective correlation time

        """
        return self.z+zM-np.log10(10**self.z+10**zM)
    
    @property
    def rhoz(self):
        """
        Return the sensitivities stored in this sensitivity object
        """
        if self.info.edited:
            self._bonds={'iso':None,'bonds':list() if self.vecs is None else \
                         [None for _ in range(self.vecs.shape[0])]}
        
        rhoz=super().rhoz
        
        if self.index is None:
            if self._bonds['iso'] is None:
                R0=np.zeros(rhoz.shape[0])
                Reff=np.zeros(rhoz.shape)
                for k,(rhoz0,tM) in enumerate(zip(rhoz,self.info['tM'])):
                    R0[k]=linear_ex(self.z,rhoz0,np.log10(tM))
                    Reff[k]=linear_ex(self.z,rhoz0,self.zeff(np.log10(tM)))-R0[k]
                self._bonds['iso']=Reff,R0
            return self._bonds['iso'][0]
        else:
            if len(self._bonds['bonds'])!=self.vecs.shape[0]:self._bonds['bonds']=[None for _ in range(self.vecs.shape[0])]
            
            if self._bonds['bonds'][self.index] is None:
                R0=np.zeros(rhoz.shape[0])
                Reff=np.zeros(rhoz.shape)
                print('Warning: Bond-specific relaxation not fully tested')
                for k,(rhoz0,tM,zeta,eta,euler) in enumerate(zip(rhoz,
                            *[self.info[key] for key in ['tM','zeta_rot','eta_rot','euler']])):
                    for tM0,A0 in zip(*AnisoDif(tM,zeta,eta,euler,self.vXY)):
                        R0[k]+=A0*linear_ex(self.z,rhoz0,np.log10(tM0))        
                        Reff[k]+=A0*linear_ex(self.z,rhoz0,self.zeff(np.log10(tM0)))
                    Reff[k]-=R0[k]
                    
                self._bonds['bonds'][self.index]=Reff,R0
            return self._bonds['bonds'][self.index][0]
        
         
    @property
    def _rhozCSA(self):
        """
        Return the sensitivities due to CSA relaxation in this sensitivity object
        """
        # self._update_rho()
        # return self.__rhoCSA.copy()
        return np.zeros([len(self.info),len(self.z)])
    @property
    def _rho_eff(self):
        """
        This will be used generally to obtain the sensitivity of the object, plus
        offsets of the parameter if required, whereas rhoz, rhoz_eff, etc. may
        change depending on the subclass.
        """
        if self.index is None:
            return self.rhoz,self._bonds['iso'][1]
        else:
            return self.rhoz,self._bonds['bonds'][self.index][1]
    
    @property
    def _rho_effCSA(self):
        """
        This will be used generally to obtain the sensitivity of the object due
        to CSA
        """
        return self._rhozCSA,np.zeros(self._rhozCSA.shape[0])
    

#%% Function to calculate amplitudes for anisotropic diffusion
from pyDR.MDtools.vft import R
   
def AnisoDif(tM,zeta=1,eta=0,euler=[0,0,0],v=[0,0,1]):
   
    """First we get the diffusion tensor, and also Diso and D2. This can be 
    input either as the principle values, Dxx, Dyy, and Dzz, or as the trace of
    the tensor (the isotropic value, tM), plus optionally the anisotropy, xi, 
    and the asymmetry, eta
    """
        
    v=np.array(v,dtype=float)
    v/=np.sqrt((v**2).sum())
    
    Diso=1/(6*tM)
    Dzz=3*Diso*zeta/(2+zeta)
    Dxx=(3*Diso-(2/3*eta*(zeta-1)/zeta+1)*Dzz)/2
    Dyy=2/3*eta*Dzz*(zeta-1)/zeta+Dxx
    Dsq=(Dxx*Dyy+Dyy*Dzz+Dzz*Dxx)/3
        
    "We the relaxation rates"    
    D1=4*Dxx+Dyy+Dzz;
    D2=Dxx+4*Dyy+Dzz;
    D3=Dxx+Dyy+4*Dzz;
    D4=6*Diso+6*np.sqrt(Diso**2-Dsq);
    D5=6*Diso-6*np.sqrt(Diso**2-Dsq);
    


    dx=(Dxx-Diso)/np.sqrt(Diso**2-Dsq);
    dy=(Dyy-Diso)/np.sqrt(Diso**2-Dsq);
    dz=(Dzz-Diso)/np.sqrt(Diso**2-Dsq);
    
    
    "We rotate the vectors in structure"
    vec=R(np.array(v),*euler)
    print('Warning: Check rotation directions')
    #TODO 
    """There is a difference in rotation between my previous implementation of
    this and the current implementation (sign on Ry/Rz switched). I am not
    sure which one is correct, so for the moment, this may very well be incorrect!
    """
        
    
    tM=np.zeros(5)
    A=np.zeros(5)
    
    m=vec
    res1=(1/4)*(3*(m[0]**4+m[1]**4+m[2]**4)-1)
    res2=(1/12)*(dx*(3*m[0]**4+6*m[1]**2*m[2]**2-1)\
    +dy*(3*m[1]**4+6*m[2]**2*m[0]**2-1)\
    +dz*(3*m[2]**4+6*m[0]**2*m[1]**2-1))
    
    A[0]=3*(m[1]**2)*(m[2]**2);
    A[1]=3*(m[0]**2)*(m[2]**2); 
    A[2]=3*(m[0]**2)*(m[1]**2); 
    A[3]=res1-res2;
    A[4]=res1+res2;
        
    tM[0]=1/D1
    tM[1]=1/D2
    tM[2]=1/D3
    tM[3]=1/D4
    tM[4]=1/D5
    
    return tM,A
        
        
    