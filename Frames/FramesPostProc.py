#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of pyDIFRATE

pyDIFRATE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyDIFRATE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyDIFRATE.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de


Created on Mon Sep 20 13:56:00 2021

@author: albertsmith
"""

from pyDR.MDtools import vft
import numpy as np

def moving_avg(t,v,sigma) -> np.array:
    """
    Calculates the moving average (Gaussian weighting) of a set of vectors, given
    in a 3xNresxNtpts array (number of residues, number of time points).
    
    Note that the result of moving_avg is NOT normalized. If this is  required,
    then use vft.norm

    Parameters
    ----------
    t : np.array
        time axis (1D array)
    v : np.array
        3xNresxNtpts array.
    sigma : float
        Standard deviation of the Gaussian weighting (use same units as time axis).

    Returns
    -------
    np.array
        Time averaged vector.

    """
    
    nsteps=np.ceil((sigma/np.diff(t).min())*2).astype(int)  #Cut off the average after 2*sigma
    return np.moveaxis([(np.exp(-(t0-t[np.max([0,k-nsteps]):k+nsteps+1])**2/(2*sigma**2))*\
       v[:,:,np.max([0,k-nsteps]):k+nsteps+1]).sum(-1) for k,t0 in enumerate(t)],0,-1)

def AvgGauss(vecs,fr_ind,sigma=0.05):
    """
    Takes a moving average of the frame direction, in order to remove librational
    motion (which tends to be correlated). Moving average is defined by a weighted
    Gaussian, defined in ns.
    """
    if sigma==0:return #Do nothing if sigma is 0
    t=vecs['t']
    
    if np.ndim(vecs['v'][fr_ind])==4:
        vecs['v'][fr_ind]=np.array([moving_avg(t,v,sigma) for v in vecs['v'][fr_ind]])
    else:
        vecs['v'][fr_ind]=moving_avg(t,vecs['v'][fr_ind],sigma)

    

def AvgHop(vecs,fr_ind,vr,sigma=0.05):
    """
    Removes short traverses from hopping motion of a trajectory. sigma determines
    where to cut off short traverses (averaging performed with a Gaussian 
    distribution, default is 50  ps).
    
    Note- needs to be run before any averaging is applied to the reference frame!
    """
    if sigma==0 and vecs['v'][fr_ind].shape[0]<3:
        return #Do nothing if sigma is 0 and only 2 vectors defined
    t=vecs['t']
    

    v12s,v23s,v34s=[moving_avg(t,v,sigma) for v in vecs['v'][fr_ind]] if sigma>0 else vecs['v'][fr_ind]
    v12s,v23s,v34s=[vft.norm(v) for v in [v12s,v23s,v34s]]
    sc=vft.getFrame(v23s,v34s)
    v12s=np.moveaxis(vft.R(v12s,*vft.pass2act(*sc)),-1,0)

    # print(np.array([(vr0*v12s).sum(1) for vr0 in vr])[:,0,:].T)
    i=np.argmax([(vr0*v12s).sum(1) for vr0 in vr],axis=0)
    # print(i.shape)
    
    v12s=vr[i,:,np.arange(i.shape[1])].T
    # from matplotlib import pyplot as plt
    # ax=plt.figure().add_subplot(111)
    # ax.scatter(v12s[0,6,:],v12s[1,6,:])
    # ax.scatter(vr[:,0,6],vr[:,1,6])
    v12s=vft.R(v12s,*sc)
    # sc=vft.getFrame(*vecs['v'][1])
    # v12s=vft.R(v12s,*vft.pass2act(*sc))
    # ax.scatter(v12s[0,6,:],v12s[1,6,:])
    vecs['v'][fr_ind]=np.array([v12s,v23s])