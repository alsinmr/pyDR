#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:23:50 2021

@author: albertsmith
"""

import numpy as np
import pyDR
from pyDR.MDtools.Ctcalc import sparse_index
import matplotlib.pyplot as plt
from time import time

pdb='/Volumes/My Book/HETs/HETs_5chain_B.pdb'
xtc='/Volumes/My Book/HETs/MDSimulation/HETs_5chain_MET_4pw_cb10_2micros.xtc'

molsys=pyDR.MolSys(pdb,xtc,step=10)

sel=pyDR.MolSelect(molsys)

sel.select_bond(Nuc='ivla1l',segids='B')


index=sparse_index(len(sel.traj),n=-1)

v=np.zeros([len(index),*sel.v.shape])
for k,_ in enumerate(sel.traj[index]):
    v[k]=sel.v
v=pyDR.MDtools.vft.norm(v.T)



fig=plt.figure('Test ct calcs')
fig.clear()
ax=fig.add_subplot(111)
for mode in ['f','a']:
    t0=time()
    ctc=pyDR.MDtools.Ctcalc(length=1,mode=mode,index=index)
    
    for k in range(3):
        for j in range(k,3):
            ctc.a=v[k]*v[j]
            for ctc0 in ctc: 
                ctc0.b=v[k]*v[j]
                ctc0.c=3/2 if k==j else 3
                ctc0.add()
            
    ct=[ctc0.Return(offset=-1/2)[0] for ctc0 in ctc]
    print(time()-t0)
    ax.plot(ctc.t,ct[0][0])
    

ctc=pyDR.MDtools.Ctcalc(length=1,mode=mode,index=index,noCt=False)
for k in range(3):
    for j in range(k,3):
        ctc.a=v[k]*v[j]
        ctc.b=v[k]*v[j]
        ctc.c=3/2 if k==j else 3
        ctc.add()
ct,S2=ctc.Return(offset=-1/2)