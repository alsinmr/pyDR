#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:44:11 2022

@author: albertsmith
"""

import pyDR
import matplotlib.pyplot as plt


#%% Solid state
ax=plt.figure().add_subplot(111)

nmr=pyDR.Sens.NMR()

nmr.new_exper(Type='NOE',Nuc='15N',v0=[400,700,1000])
nmr.new_exper(Type='R1',Nuc='15N',v0=[400,700,1000])
nmr.new_exper(Type='R1p',Nuc='15N',v0=700,vr=60,v1=[5,35,50])

nmr.plot_Rz(index=[1,4,7],norm=True,ax=ax)
cmap=plt.get_cmap('tab10')
for k in range(3):
    ax.plot(nmr.z,nmr.rhoz[3*k:3*(k+1)].T/nmr.rhoz[3*k:3*(k+1)].max(1),color=cmap(k))


#%% Solution state
ax=plt.figure().add_subplot(111)

nmr=pyDR.Sens.NMR()

nmr.new_exper(Type='NOE',Nuc='15N',v0=[400,700,1000])
nmr.new_exper(Type='R1',Nuc='15N',v0=[400,700,1000])
nmr.new_exper(Type='R2',Nuc='15N',v0=700)

nmr.plot_Rz(index=[1,4,6],norm=True,ax=ax)
cmap=plt.get_cmap('tab10')
for k in range(3):
    ax.plot(nmr.z,nmr.rhoz[3*k:3*(k+1)].T/nmr.rhoz[3*k:3*(k+1)].max(1),color=cmap(k))
    
#%% NERDD
ax=plt.figure().add_subplot(111)

nmr=pyDR.Sens.NMR(tc=np.logspace(-8.5,-0.5,200),Type='R1p',Nuc='15N',v0=600,v1=np.linspace(10,42,12),vr=44)

nmr.plot_Rz(color='green',ax=ax)
ax.set_xlim([])
