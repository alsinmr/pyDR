#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:06:58 2022

@author: albertsmith
"""

import pyDR
import numpy as np
from pyDR.Fitting.fit import model_free
from pyDR.misc.tools import linear_ex
import matplotlib.pyplot as plt
from copy import copy

frames=pyDR.Project('/Users/albertsmith/Documents/Dynamics/HETs_Methyl_Loquet/pyDR_proc/frames')
sims=['HETs_MET_4pw','HETs_4pw']

# A=np.concatenate([model_free(frames[sim]['MethylLib_md'][0],nz=1,fixz=-14)[1][0] for sim in sims])
# theta=np.arccos(np.sqrt(((1-A)-1/4)*4/3))*180/np.pi
# z=np.concatenate([model_free(frames[sim]['MethylHop_md'][0],nz=1,fixA=8/9)[0][0] for sim in sims])


# z=[z[:40],z[40:],z]
# theta=[theta[:40],theta[40:],theta]

# b=list()
# M=list()
# for k,(theta0,z0) in enumerate(zip(theta,z)):
#     zstd=np.std(z0)
#     i=np.logical_and(z0<z0.mean()+2*zstd,z0>z0.mean()-2*zstd)
#     z0,theta0=z0[i],theta0[i]
#     z[k]=z0
#     theta[k]=theta0
#     M.append(np.concatenate(([np.ones(z0.shape)],[z0],[z0**2])).T)
#     b.append(np.linalg.pinv(M[-1])@np.atleast_2d((theta0)**0.5).T)

# if __name__=='__main__':
#     ax=plt.figure().add_subplot(111)
#     for z0,theta0 in zip(z[:2],theta[:2]):
#         plt.scatter(z0,(theta0)**0.5)
#     for z0,b0,M0 in zip(z,b,M):
#         i=np.argsort(z0)
#         plt.plot(z0[i],(M0@b0)[i])

b=np.array([[-14.43278425],
 [ -1.91759892]])
if b is None:
    sim=sims[0]
    A=model_free(frames[sim]['MethylLib_md'][0],nz=1,fixz=-14)[1][0]
    theta=np.arccos(np.sqrt(((1-A)-1/4)*4/3))*180/np.pi
    z=model_free(frames[sim]['MethylHop_md'][0],nz=1,fixA=8/9)[0][0]
    
    M=np.concatenate(([np.ones(z.shape)],[z])).T
    b=np.linalg.lstsq(M,np.atleast_2d(theta).T**0.5)[0]

if __name__=='__main__':
    ax=plt.figure().add_subplot(111)
    plt.scatter(z,theta**0.5)
    plt.plot(z,M@b)
    

def tc2A(z):
    """
    Solves for the librational amplitude as a function of methyl log-correlation
    time

    Parameters
    ----------
    z : float/array
        Log correlation time.

    Returns
    -------
    A
        Ampltidue of motion

    """
    z=np.array(z)
    M=np.concatenate((np.atleast_2d(np.ones(z.shape)),np.atleast_2d(z))).T
    theta=(M@b)**2
    return 1-1/4*(3*np.cos(theta[:,0]*np.pi/180)**2+1)
    
def solve_met(data,include=None):
    """
    Uses the first two detector and an assumption of a linear relationship between
    the log-correlation time of methyl rotation and the square-root of the
    angle desribing methyl libration (two-site hop)

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    data=copy(data)
    data.source.project=None
    out=copy(data)
    R0=copy(data.R)
    
    include=np.arange(2) if include is None else np.array(include)
    
    max_iter=10
    counter=0
    z=None
    while counter<max_iter:
        counter+=1
        print('counter:{0}'.format(counter))
        z0=z
        z=model_free(data,nz=1,fixA=8/9,include=include)[0][0]
        A=tc2A(z)
        if z0 is not None:
            if np.max(np.abs(z0-z))<.05:
                break
        data.R[:,1:]=(R0[:,1:].T/(1-A)).T
        
    R=(linear_ex(data.sens.z,data.sens.rhoz,z)*(1-A)).T
    R[:,0]+=A
    out.R=R
    
    
    
    return z,A,out