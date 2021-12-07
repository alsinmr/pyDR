#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:23:50 2021

@author: albertsmith
"""

import numpy as np
from pyDIFRATE.Struct.vf_tools import R,getFrame
from pyDR import Ct_funs
import matplotlib.pyplot as plt
from time import time

#%% Set up a random tumbling motion
n=int(1e6)          #Number of time points in simulated trajectory (reduce for faster calculations)
dt=5e-3             #Time step (ns)
t=np.arange(n)*dt   #Time axis (10 us)
t[0]=1e-3           #We plot on a log scale. We replace 0 with 1 ps to keep it on the scale (labeling later indicates this)


dr=0.5              #Step size (in degrees) for rotation of the molecule in solution (determines correlation time)

gamma_r=np.random.rand(n)*2*np.pi

vr=[np.array([0,0,1])]  #Start out with vector pointing along z
for k in range(n-1):    #Loop over all time points
    sc=getFrame(vr[-1])     #Get frame of current vector direction
    """Below is the direction of the vector due to a step of dr degrees (see simulation parameters),
    occuring at an angle of gamma_r[k] (given in the frame of the current vector direction)
    """
    vr0=[np.cos(gamma_r[k])*np.sin(dr*np.pi/180),np.sin(gamma_r[k])*np.sin(dr*np.pi/180),np.cos(dr*np.pi/180)]
    "We rotate vr0 into the lab frame frome the frame of the vector direcion, and append this angle"
    vr.append(R(vr0,*sc))
vr=np.array(vr).T #Convert into a numpy array


#%% Now calculate the correlation function
A=[vr[i]*vr[j] for i,j in zip([0,1,2,0,0,1],[0,1,2,1,2,2])]
ct_calc=Ct_funs.Ct_calc(A,weight=[3/2,3/2,3/2,3,3,3],offset=-1/2)

t0=time()
ct_calc.run()
ct=ct_calc.cleanup()
totaltime=time()-t0
print(totaltime)

ct_calc._mode='CtJit'
t0=time()
ct_calc.run()
ct=ct_calc.cleanup()
totaltime=time()-t0
print(totaltime)