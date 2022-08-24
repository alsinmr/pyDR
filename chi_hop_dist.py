#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 12:04:22 2022

@author: albertsmith
"""


import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')
import pyDR
import os
import numpy as np
from scipy.optimize import least_squares
from copy import copy
from pyDR.Fitting.fit import model_free

proj=pyDR.Project('proj_md')
sub=proj['.+Chi+.Hop']['HETs_MET_4pw']+proj['OuterHop']['HETs_MET_4pw']

def pop_from_file(filename:str='MET_4pw_chi_probabilities'):
    """
    Loads the chi1/chi2 populations from file (found in chi1_2_populations folder)

    Parameters
    ----------
    filename : str, optional
        DESCRIPTION. The default is 'MET_4pw_chi_probabilities'

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """
    directory='/Users/albertsmith/Documents/Dynamics/HETs_Methyl_Loquet/chi1_2_populations'
    if filename is None:filename='MET_4pw_chi_probabilities'
    if not(os.path.exists(filename)):filename=os.path.join(directory,filename)
    if not(os.path.exists(filename)):filename+='.txt'
    
    out=dict()
    with open(filename,'r') as f:
        for line in f:
            key=int(line.strip().split('_')[1].split(';')[0])
            out[key]=dict()
            v=line.strip().split('[')[1].split(']')[0].split(',')
            out[key]['chi1']=np.array([float(v0) for v0 in v])
            if 'Chi2' in line:
                v=line.strip().split('Chi2')[1].split('[')[1].split(']')[0].split(',')
                out[key]['chi2']=np.array([float(v0) for v0 in v])
    return out

def rho_dist(sens,z,A,sigma):
    """
    Calculates the detector response corresponding to a Gaussian distribution with
    center z, amplitude A (integral), and standard deviation sigma.

    Parameters
    ----------
    sens : type
        Sensitivity object for detectors 
    z : float
        Center (log-correlation time).
    A : float
        Amplitude of decay (integral of distribution).
    sigma : float
        standard deviation of distribution (on log scale).

    Returns
    -------
    array
        Detector responses corresponding to distribution
    """
    dz=sens.z[1]-sens.z[0]
    dist=np.exp(-(sens.z-z)**2/(2*sigma**2))
    dist*=A/dist.sum()
    dist[-1]+=(1-A)
    return (sens.rhoz*dist).sum(1)

def fit2dist(data,max_sigma=5):
    """
    Fits a data object to a normal distribution of log-correlation times.

    Parameters
    ----------
    data : TYPE
        pyDR data object.

    Returns
    -------
    z,A,sigma

    """
    z,A,sigma,rhoc=list(),list(),list(),list()
    for rho in data.R:
        def fun(x):
            return rho_dist(data.sens,x[0],x[1],x[2])-rho
        z0=data.sens.info['z0'][np.argmax(rho[:-1])]
        out=least_squares(fun,[z0,1-rho[-1],0.005+max_sigma/2],bounds=([-14,0,0.01],[-3,1,max_sigma]))['x']
        z.append(out[0])
        A.append(out[1])
        sigma.append(out[2])
        rhoc.append(rho_dist(data.sens,z[-1],A[-1],sigma[-1]))
    rhoc=np.array(rhoc)
    out=copy(data)
    out.R=rhoc
    return np.array(z),np.array(A),np.array(sigma),out


if __name__=='__main__':
    sub.close_fig('all')
    for k,data in enumerate(sub):
        sub.current_plot=k+1
        sub[k].plot(style='bar')
        *_,out=model_free(data,nz=1)
        sub.plot(data=out,color='black')
        *_,out=model_free(data,nz=2)
        sub.plot(data=out,color='black',linestyle='--')
        *_,out=fit2dist(data)
        sub.plot(data=out,color='black',linestyle=':')
        for a in sub.plot_obj.ax[:-1]:a.set_ylim([0,.8])
    