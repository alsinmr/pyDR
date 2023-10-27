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
    Returns 1/T1 (R1) relaxation rate constant. Sources of relaxation are:
        
    Quadrupole coupling: Provide Nuc, v0, QC, and etaQ
    Dipole coupling (heteronuclear): Provide Nuc, Nuc1, v0, dXY
    Dipole coupling (homonuclear): Provide Nuc, Nuc1, v0, dXY, vr, CSoff
    CSA: Provide Nuc, v0, CSA, eta
    
    CSA and quadrupole relaxation use the same parameter, eta
    
    Defaults of most parameters are 0 (Nuc1 is None). 
    Multiple dipole couplings may be considered. Provide dXY and Nuc1 as a list
    
    All provided contributions will be included in the total rate constant.
    """


    v0=v0*1e6     #1H resonance frequency (convert MHz to Hz)
    vr=vr*1e3     #MAS frequency (convert kHz to Hz)
    dXY=np.atleast_1d(dXY)
    Nuc1=np.atleast_1d(Nuc1)
    assert Nuc1.size==dXY.size,"Nuc1 and dXY must have the same size"
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA=CSA*vX/1e6
    Delv=np.array(CSoff)*vX/1e6
    R=np.zeros(np.shape(tc))

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

def NOE(tc,Nuc,v0,Nuc1,dXY):
    """
    Returns the NOE rate constant between two nuclei (usually a 1H and heteronucleus).
    
    Note, this is the rate constant, not the enhancement or eta.
    """
    v0*=1e6
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    R=np.zeros(np.shape(tc))
    
    dXY=np.atleast_1d(dXY)
    Nuc1=np.atleast_1d(Nuc1)
    assert Nuc1.size==dXY.size,"Nuc1 and dXY must have the same size"
    if len(dXY)>1:print('Warning: The behavior of NOE has been changed to accept multiple dipole couplings.',
                        '\nPlease check that you intended to use this behavior',
                        '\nThis warning will be removed at a later point')
    
    
    for N1,dXY1 in zip(Nuc1,dXY):
        if N1!=None:
            vY=NucInfo(N1)/NucInfo('1H')*v0
            S=NucInfo(N1,'spin')
            sc=S*(S+1)*4/3 # Scaling factor depending on the spin, =1 for spin 1/2
            R+=sc*(np.pi*dXY1/2)**2*(-J(tc,vX-vY)+6*J(tc,vY+vX))
        
    return R

def R1Q(tc,Nuc,v0,QC=0,etaQ=0):
    """This function calculates the relaxation rate constant for relaxation of
    quadrupolar order
    """
    v0*=1e6
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    
    S=NucInfo(Nuc,'spin')
    deltaQ=1/(2*S*(2*S-1))*QC*2*np.pi
    C=(deltaQ/2)**2*(1+etaQ**2/3)    #Constant scaling the relaxation
    if S==0.5:
        print('No quadruple coupling for spin=1/2')
    elif S==1:
        R=C*9*J(tc,vX)
    elif S==1.5:
        R=C*(36*J(tc,vX)+36*J(tc,2*vX))
    elif S==2.5:
        R=C*(792/7*J(tc,vX)+972/7*J(tc,2*vX))
    else:
        print('Spin not implemented')
        
    return R

def R1p(tc,Nuc,v0,Nuc1=None,CSA=0,dXY=0,eta=0,vr=0,v1=0,offset=0,QC=0,etaQ=0):
    #Calc R1 contributions before scaling input values
    R10=R1(tc,Nuc,v0,Nuc1=Nuc1,CSA=CSA,dXY=dXY,eta=eta,vr=vr,CSoff=0,QC=QC,etaQ=etaQ)    #We do this first, because it includes all R1 contributions
    
    v0*=1e6 #Input in MHz
    v1*=1e3 #Input in kHz
    vr*=1e3 #Input in kHz
    v1Y=0
    v1Y*=1e3
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    
    CSA=CSA*vX/1e6 #Input in ppm
    R=np.zeros(np.shape(tc))
    
    "Treat off-resonance spin-lock"
    ve=np.sqrt(v1**2+offset**2)
    if ve==0:
        theta=np.pi/2
    else:
        theta=np.arccos(offset/ve)
    

    "Start here with the dipole contributions"
    R1del=np.zeros(np.shape(tc))
    if Nuc1 is not None:
        dXY=np.atleast_1d(dXY)
        Nuc1=np.atleast_1d(Nuc1)
        
        # if np.size(dXY)==1:
        #     vY=NucInfo(Nuc1)/NucInfo('1H')*v0
        #     S=NucInfo(Nuc1,'spin')
        #     sc=S*(S+1)*4/3 #Scaling depending on spin of second nucleus
        #     R1del=sc*(np.pi*dXY/2)**2*(3*J(tc,vY)+
        #               1/6*J(tc,2*vr-ve+v1Y)+2/6*J(tc,vr-ve+v1Y)+2/6*J(tc,vr+ve+v1Y)+1/6*J(tc,2*vr+ve+v1Y)+
        #               1/6*J(tc,2*vr-ve-v1Y)+2/6*J(tc,vr-ve-v1Y)+2/6*J(tc,vr+ve-v1Y)+1/6*J(tc,2*vr+ve-v1Y))
        # else:            
        for k in range(0,np.size(dXY)):
            vY=NucInfo(Nuc1[k])/NucInfo('1H')*v0
            S=NucInfo(Nuc1[k],'spin')
            sc=S*(S+1)*4/3 #Scaling depending on spin of second nucleus
            if vX==vY:
                print('homonuclear')
                R1del+=sc*(np.pi*dXY[k]/2)**2*(1/24*(1+3*np.cos(2*theta)**2)*J(tc,vr)+
                                              1/48*(1+3*np.cos(2*theta)**2)*J(tc,2*vr)+
                                              3/4*np.sin(theta)**4*(J(tc,2*ve+vr)+0.5*J(tc,2*ve+2*vr)+
                                                        1/2*J(tc,2*ve-2*vr)+J(tc,2*ve-vr))+
                                              3/8*np.sin(2*theta)**2*(J(tc,ve+vr)+1/2*J(tc,ve+2*vr)+
                                                        1/2*J(tc,ve-2*vr)+J(tc,ve-vr)))
                # TODO We should double check that the T1 contribution is correct  
            else:
                R1del+=sc*(np.pi*dXY[k]/2)**2*(3*J(tc,vY)+
                          1/6*J(tc,2*vr-ve+v1Y)+2/6*J(tc,vr-ve+v1Y)+2/6*J(tc,vr+ve+v1Y)+1/6*J(tc,2*vr+ve+v1Y)+
                          1/6*J(tc,2*vr-ve-v1Y)+2/6*J(tc,vr-ve-v1Y)+2/6*J(tc,vr+ve-v1Y)+1/6*J(tc,2*vr+ve-v1Y))
                
    "CSA contributions"
    R1del+=1/6*(2*np.pi*CSA)**2*(1/2*J(tc,2*vr-ve)+J(tc,vr-ve)+J(tc,vr+ve)+1/2*J(tc,2*vr+ve))
    "Here should follow the quadrupole treatment!!!"    
    
    "Add together R1 and R1p contributions, depending on the offset"
    R+=R10+np.sin(theta)**2*(R1del-R10/2) #Add together the transverse and longitudinal contributions   
    return R

def R2(tc,Nuc,v0,Nuc1=None,CSA=0,dXY=0,eta=0,vr=0,v1=0,offset=0,QC=0,etaQ=0):    
    return R1p(tc,Nuc,v0,Nuc1=Nuc1,CSA=CSA,dXY=dXY,eta=eta,vr=0,v1=0,offset=offset,QC=QC,etaQ=etaQ)  

def S2(tc):
    """
    Order parameter (note- one must provide 1-S2 into the data.R matrix!)
    
    Returns a uniform sensitivity, independent of correlation time.
    
    One could consider explicitely simulating the REDOR/DIPSHIFT experiment and
    determining how sensitive it really is w.r.t. correlation time and output
    the results in this function.
    """
    return np.ones(np.shape(tc))  

def ccXY(tc,v0,Nuc,dXY=0,Nuc1=None,CSA=0,theta=0):
    """
    CSA-dipole cross-correlated transverse relaxation
    """
    v0*=1e6 #v0 in MHz
    theta*=np.pi/180  #theta in degrees
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA*=vX/1e6 #CSA in ppm

    if Nuc1 is not None:
        S=NucInfo(Nuc1,'spin')
        if S!=0.5:
            print('Warning: Formulas for cross-correlated relaxation have only been checked for S=1/2')
        sc=S*(S+1)*4/3  
        R=np.sqrt(sc)*1/8*(2*np.pi*dXY)*(2*np.pi*CSA)*(3*np.cos(theta)**2-1)/2.*(4*J(tc,0)+3*J(tc,vX))
    else:
        R=np.zeros(np.shape(tc))
    return R

def ccZ(tc,v0,Nuc,dXY=0,Nuc1=None,CSA=0,theta=0):
    """
    CSA-dipole cross-correlated longitudinal relaxation
    """
    v0*=1e6 #v0 in MHz
    theta*=np.pi/180  #theta in degrees
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA*=vX/1e6 #CSA in ppm
    
    if Nuc1 is not None:
        S=NucInfo(Nuc1,'spin')
        if S!=0.5:
            print('Warning: Formulas for cross-correlated relaxation have only been checked for S=1/2')
        sc=S*(S+1)*4/3  
        R=np.sqrt(sc)*1/8*(2*np.pi*dXY)*(2*np.pi*CSA)*(3*np.cos(theta)**2-1)/2.*6*J(tc,vX)
    else:
        R=R=np.zeros(np.shape(tc))
    return R
