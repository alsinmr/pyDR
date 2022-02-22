#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:59:53 2022

@author: albertsmith
"""

from numpy.linalg import pinv as PINV
from numpy import sqrt
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
        stdev=sqrt((pinv**2).sum())
    else:
        Y=lsq(X[0],X[1],bounds=X[2])
        rho=Y['x']
        Rc=Y['fun']+X[1]
        stdev=sqrt((PINV(X[0])**2).sum(1))
    return rho,stdev,Rc 