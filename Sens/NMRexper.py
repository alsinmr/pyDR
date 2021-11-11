#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:21:34 2021

@author: albertsmith
"""

import numpy as np
from pyDR.misc.tools import NucInfo

#%% Functions for calculation of relaxation
def J(tc,v0):
    """
    Returns the spectral density at frequency v for the correlation time axis tc
    """
    return 2/5*tc/(1+(2*np.pi*v0*tc)**2)

def R1(tc,Nuc,v0,Nuc1=None,CSA=0,dXY=0,eta=0,vr=0,CSoff=0,QC=0,etaQ=0):
    """
    Returns the T1 (as R1) relaxation rate constant. Sources of relaxation are:
        
    Quadrupole coupling: Provide Nuc, v0, QC, and etaQ
    Dipole coupling (heteronuclear): Provide Nuc, Nuc1, v0, dXY
    Dipole coupling (homonuclear): Provide Nuc, Nuc1, v0, dXY, vr, CSoff
    CSA: Provide Nuc, v0, CSA, eta
    
    CSA and quadrupole relaxation use the same parameter, eta
    
    Defaults of most parameters are 0 (Nuc1 is None). 
    Multiple dipole couplings may be considered. Provide dXY and Nuc1 as a list
    
    All provided contributions will be included in the total rate constant.
    """

    v0=np.array(v0)*1e6     #1H resonance frequency (convert MHz to Hz)
    vr=np.array(vr)*1e3     #MAS frequency (convert kHz to Hz)
    dXY=np.atleast_1d(dXY)
    Nuc1=np.atleast_1d(Nuc1)
    assert Nuc1.size==dXY.size,"Nuc1 and dXY must have the same size"
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA=CSA*vX/1e6
    Delv=np.array(CSoff)*vX/1e6
    R=np.zeros(tc.shape)

    "Dipole relaxation"
    for N1, dXY1 in zip(Nuc1,dXY):
        if N1 is not None:
            vY=NucInfo(N1)/NucInfo('1H')*v0
            S=NucInfo(N1,'spin')
            sc=S*(S+1)*4/3 # Scaling factor depending on the spin, =1 for spin 1/2            
            if vX==vY:  #Homonuclear
                R+=sc*(np.pi*dXY1/2)**2*(1/6*J(tc,Delv+2*vr)+1/6*J(tc,Delv-2*vr)\
                   +1/3*J(tc,Delv+vr)+1/3*J(tc,Delv-vr)+3*J(tc,vX)+6*J(tc,2*vX))
            else:       #Heteronuclear
                R+=sc*(np.pi*dXY1/2)**2*(J(tc,vX-vY)+3*J(tc,vX)+6*J(tc,vY+vX))
    
    print(R)
    "Quadrupole Relaxation"
    """
    Note that we calculate the orientationally averaged, initial relaxation
    rate constant. We do not account for multi-exponentiality as occuring from
    either different orientations or from relaxation through multiple spin
    states.
    
    Use R1Q for relaxation of quadrupolar order
    """
    S=NucInfo(Nuc,'spin')     
    if S>=1:      
        deltaQ=1/(2*S*(2*S-1))*QC*2*np.pi
        C=(deltaQ/2)**2*(1+etaQ**2/3) #Constant that scales the relaxation        
        if S==1:
            R+=C*(3*J(tc,vX)+12*J(tc,2*vX))
        elif S==1.5:
            R+=C*(36/5*J(tc,vX)+144/5*J(tc,2*vX))
        elif S==2.5:
            R+=C*(96/5*J(tc,vX)+384/5*J(tc,2*vX))
        else:
            print('Spin={0} not implemented for quadrupolar relaxation'.format(S))
    "CSA relaxation"
    R+=3/4*(2*np.pi*CSA)**2*(1+eta**2/3)*J(tc,vX)
    return R
