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

from pyDIFRATE.Struct import vf_tools as vft
import numpy as np


def moving_avg(t,v,sigma):
    """
    Moving average of a vector direction. Note that the output is NOT normalized,
    but the direction is correct
    """
    nsteps=np.ceil((sigma/np.diff(t).min())*2).astype(int)  #Cut off the average after 2*sigma
    return np.moveaxis([(np.exp(-(t0-t[np.max([0,k-nsteps]):k+nsteps+1])**2/(2*sigma**2))*\
       v[:,:,np.max([0,k-nsteps]):k+nsteps+1]).sum(-1) for k,t0 in enumerate(t)],0,-1)

def AvgGauss(vecs,fr_ind,sigma=50):
    """
    Takes a moving average of the frame direction, in order to remove librational
    motion (which tends to be correlated). Moving average is defined by a weighted
    Gaussian, defined in the units of the trajectory (usually ps, default here
    is 50).
    """
    if sigma==0:return #Do nothing if sigma is 0
    t=vecs['t']
    
    if np.ndim(vecs['v'][fr_ind])==4:
        vecs['v'][fr_ind]=np.array([moving_avg(t,v,sigma) for v in vecs['v'][fr_ind]])
    else:
        vecs['v'][fr_ind]=moving_avg(t,vecs['v'][fr_ind],sigma)

    

def AvgHop(vecs,fr_ind,vr,sigma=50):
    """
    Removes short traverses from hopping motion of a trajectory. sigma determines
    where to cut off short traverses (averaging performed with a Gaussian 
    distribution, default is 50  ps, note that if trajectory uses a different unit,
    then this number will need to be adjusted).
    
    Note- needs to be run before any averaging is applied to the reference frame!
    """
    if sigma==0:return #Do nothing if sigma is 0
    t=vecs['t']
    
    v12s,v23s,v34s=[moving_avg(t,v,sigma) for v in vecs['v'][fr_ind]]
    
    sc=vft.getFrame(v23s,v34s)
    v12s=np.moveaxis(vft.R(v12s,*vft.pass2act(*sc)),-1,0)

    i=np.argmax([(vr0*v12s).sum(1) for vr0 in vr],axis=0)
    
    v12s=vr[i,:,np.arange(i.shape[1])].T
    v12s=vft.R(v12s,*sc)
    vecs['v'][fr_ind]=np.array([v12s,v23s])