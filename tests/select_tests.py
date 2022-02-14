#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:18:36 2022

@author: albertsmith
"""

import pyDR
import numpy as np
from pyDR.MolSys import MolSelector
from pyDR.Ct_fast import Ct00,Ct00slow,Ct00jit
from time import time


pdb='/Users/albertsmith/MDSimulations/HETs/HETs_5chain_B.pdb'
xtc='/Users/albertsmith/MDSimulations/HETs/HETs_5chain_MET_4pw_cb10_2micros.xtc'

MS=pyDR.MolSys(pdb,xtc,tf=100000)

select=MolSelector(MS)

select.select_bond(Nuc='15N',segids='B')

v=np.zeros([3,select.sel1.__len__(),select.trajectory.__len__()])


for k,_ in enumerate(select.trajectory):
    v[:,:,k]=select.v.T
    if not(np.mod(k,10000)):
        print(k)

t0=time()
ct=Ct00(v[:,:10,:50000])
print(time()-t0)

t0=time()
x,y,z=v[:,:10,:50000]
ct=Ct00jit(x,y,z)
print(time()-t0)

