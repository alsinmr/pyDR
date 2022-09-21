#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:59:53 2022

@author: albertsmith
"""

from numpy.linalg import pinv as PINV
from numpy import sqrt,atleast_1d,atleast_2d,concatenate,repeat,ones,abs,sum
from scipy.optimize import lsq_linear as lsq

def fit0(X):
    """
    Used for parallel fitting of data. Single argument in the input should
    include data, the r matrix, and the upper and lower bounds
    """
    if X[2] is None:
        pinv=PINV(X[0])   #Simple pinv fit if no bounds required
        rho=pinv@X[1]
        Rc=X[0]@rho
        stdev=sqrt((pinv**2).sum(1))
    else:
        Y=lsq(X[0],X[1],bounds=X[2])
        rho=Y['x']
        Rc=Y['fun']+X[1]
        stdev=sqrt((PINV(X[0])**2).sum(1))
    return rho,stdev,Rc 

def dist_opt(X):
    """
    Optimizes a distribution that yields detector responses, R, where the 
    distribution is required to be positive, and have an integral of 1-S2
    
    Ropt,dist=dist_opt((R,R_std,rhoz,S2,dz))
    
    Note- input is via tuple (or iterable)
    """
    
    R,R_std,rhoz,S2=X
    total=atleast_1d(1-S2)
    
    ntc=rhoz.shape[1]
    rhoz=concatenate((rhoz/repeat(atleast_2d(R_std).T,ntc,axis=1),
        atleast_2d(ones(ntc))),axis=0)
    Rin=concatenate((R/R_std,total))
    
    dist=0
    while abs(sum(dist)-total)>1e-3:  #This is a check to see that the sum condition has been satisfied
        dist=lsq(rhoz,Rin,bounds=(0,1))['x']
        Rin[-1]=Rin[-1]*10 #Increase the weighting of the total if sum condition not satisfied
        rhoz[-1]=rhoz[-1]*10 
    Ropt=(rhoz[:-1]@dist)*R_std
    
    return Ropt,dist