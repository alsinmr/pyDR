#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:05:22 2024

@author: albertsmith
"""

import pyDR
import numpy as np
from pyDR.PCA.PCAclean import PCA

select=pyDR.MolSelect(topo='/Users/albertsmith/Documents/Dynamics/MDsims/HETs/backboneB.pdb',
                      traj_files='/Users/albertsmith/Documents/Dynamics/MDsims/HETs/backboneB.xtc',step=1,tf=10000)

pca=PCA(select)
pca.select_bond('15N',resids=np.concatenate([np.arange(225,248),np.arange(261,281)]))
# pca.select_bond('15N')
pca.select_atoms('all')




ired=pyDR.md2iRED(select).iRED2data()
ired.detect.r_no_opt(10)
noopt=ired.fit()
noopt.detect.r_auto(6)
fit=noopt.fit().modes2bonds()
