#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:55:28 2022

@author: albertsmith
"""


import pyDR
from matplotlib.pyplot import get_cmap


nmr=pyDR.Sens.SolnNMR(Type=['R2','R1','NOE'],v0=600,Nuc='15N',tM=5e-9,zeta_rot=3,eta_rot=1)
nmr.vecs=np.array([[0,0,1],[1,0,0],[0,1,0]])


ax=nmr.plot_rhoz(color='black',linestyle=':')[0].axes
for k,nmr0 in enumerate(nmr):
    nmr0.plot_rhoz(color=get_cmap('tab10')(k),ax=ax)
    
r=pyDR.Sens.Detector(nmr)
r.r_auto(3)

ax=r.plot_rhoz(color='black',linestyle=':')[0].axes
for k,r0 in enumerate(r):
    r0.plot_rhoz(color=get_cmap('tab10')(k),ax=ax)