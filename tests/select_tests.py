#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:18:36 2022

@author: albertsmith
"""

import pyDR
import numpy as np
from pyDR.MolSys import MolSelector
from pyDR.Ct_fast import Ct00
from time import time


pdb='/Users/albertsmith/MDSimulations/HETs/HETs_5chain_B.pdb'
xtc='/Users/albertsmith/MDSimulations/HETs/HETs_5chain_MET_4pw_cb10_2micros.xtc'

MS=pyDR.MolSys(pdb,xtc)

select=MolSelector(MS)

select.select_bond(Nuc='15N',segids='B')

v=np.zeros([3,select.sel1.__len__(),select.trajectory.__len__()])


for k,_ in enumerate(select.trajectory):
    v[:,:,k]=select.v.T
    if not(np.mod(k,10000)):
        print(k)

t0=time()
ct=Ct00(v)
print(time()-t0)