#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:12:40 2021

@author: albertsmith
"""

import numpy as np
from scipy.linalg import eig

def Kex(n_beta=25,n_gamma=50,beta_max=np.pi,beta_min=0):
    """
    Builds an exchange matrix for diffusion on a sphere. We use equally spaced
    angles, and assume that the diffusion rate constant for diffusion between
    two sites is proportional to the arc length between the two angles. We only
    allow hopping between nearest neighbors (including diagonal hops).
    
    Matrix will be normalized such that the mean correlation time is 1. One may
    restrict the beta angles to produce wobbling in a cone/wobbling on a cone
    models.
    
    Returns beta, gamma, and the exchange matrix
    """
    
    if beta_max==beta_min:
        n_beta=1
        n_gamma=200
    
    n=n_beta*n_gamma
    
    "Setup the angles"
    gamma0=np.linspace(0,2*np.pi,n_gamma,endpoint=False)    #Gamma angles
    
#    if beta_min==0 and beta_max==np.pi:        
#        beta0=np.linspace(beta_min,beta_max,n_beta,endpoint=False)
#        beta0+=beta0[1]/2
#    elif beta_min==0:
#        db=beta_max/(n_beta+1/2)
#        beta0=np.linspace(db/2,beta_max,n_beta)
#    else:
#        db=(beta_max-beta_min)/(n_beta+1/2)
#        beta0=np.linspace(beta_min,beta_max-db/2,n_beta)
    
    if beta_min==0:beta_min+=.02
    if beta_max==np.pi:beta_max+=-.02
    beta0=np.linspace(beta_min,beta_max,n_beta)
        
    beta,gamma=[x.reshape(n) for x in np.meshgrid(beta0,gamma0)]
        
    
    kex=np.zeros([n,n])    
    
    dgamma=gamma0[1]-gamma0[0]
    
    if len(beta0)!=1:
        dbeta=beta0[1]-beta0[0]
        for k in range(n_gamma):
            "Hops over beta angle"
            kex[np.arange(k*n_beta,(k+1)*n_beta-1),np.arange(k*n_beta+1,(k+1)*n_beta)]+=1/dbeta*np.sqrt(np.sin(beta0[:-1])/np.sin(beta0[1:]))
            kex[np.arange(k*n_beta+1,(k+1)*n_beta),np.arange(k*n_beta,(k+1)*n_beta-1)]+=1/dbeta*np.sqrt(np.sin(beta0[1:])/np.sin(beta0[:-1]))
    
    for k,b in enumerate(beta0):
        "Hops over gamma angle"
        b12=np.arccos(np.sin(b)**2*np.cos(dgamma)+np.cos(b)**2)
        kex[np.arange(k,n,n_beta)[:-1],np.arange(k+n_beta,n,n_beta)]+=1/b12
        kex[np.arange(k+n_beta,n,n_beta),np.arange(k,n,n_beta)[:-1]]+=1/b12
        kex[k,-n_beta+k]=1/b12
        kex[-n_beta+k,k]=1/b12
        
    kex[np.arange(n),np.arange(n)]+=-kex.sum(0)
    
    return beta,gamma,kex
    
   
    
    
def ex2Ct(beta,gamma,kex):   
    """
    Takes an exchange matrix, and the corresponding beta and gamma angles, and
    calculates the correlation populations at equilibrium, the resulting order 
    parameter, and the correlation times and amplitudes contributing to the
    correlation function
    
    Ct=S^2+(1-S^2)sum(A_m*exp(-t/tau_m))
    
    returns peq,S2,tau,A
    """
    
    w,v=eig(kex)
    
        
    i=np.argsort(w)[::-1]
    w=w[i]
    v=v[:,i]
    peq=v[:,0].real/v[:,0].real.sum() #Equilibrium, normalized to sum to one
    
    vi=np.linalg.pinv(v)
#    vi=v.T/v.sum(1) #Renormalized Inverse of v
    
    x,y,z=np.sin(beta)*np.cos(gamma),np.sin(beta)*np.sin(gamma),np.cos(beta)
    P2=-1/2+3/2*(np.sum([np.dot(np.atleast_2d(q).T,np.atleast_2d(q)) for q in [x,y,z]],0)**2)
    
    pp=np.dot(np.atleast_2d(peq).T,np.atleast_2d(peq))
    
    S2=(P2*pp).sum()    #Order parameter
    
    tau=-1/w.real    #Correlation times
    
    A=list()
    for vim,vm in zip(vi,v.T):
        A.append((np.dot(np.atleast_2d(vm).T,np.atleast_2d(vim*peq))*P2).sum())
    A=np.array(A).real
    
    tau=tau[1:]
    A=A[1:]/(1-S2)
    
    return peq,S2,tau,A
    
def degen_tau(tau_in,Ain):
    """
    Takes a list of tau and A values, and re-bins the result to eliminate tau
    with vanishing amplitudes and degenerate tau
    """
    
    i=Ain>1e-10
    tau=tau_in.copy()[i]
    A=Ain.copy()[i]
    
    skip=np.zeros(tau.shape,dtype=bool)
    for k,t in enumerate(tau):
        i=np.logical_and(tau>t*.99999,tau<t*1.00001)
        skip[i]=True
        sumA=A[i].sum()
        A[i]=0
        A[k]=sumA
        
    i=A>1e-10
    A=A[i]
    tau=tau[i]
    return tau,A
        