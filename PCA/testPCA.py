#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:05:22 2024

@author: albertsmith
"""

import pyDR
import numpy as np
from pyDR.PCA.PCAclean import PCA

# select=pyDR.MolSelect(topo='/Users/albertsmith/Documents/Dynamics/MDsims/HETs/backboneB.pdb',
#                       traj_files='/Users/albertsmith/Documents/Dynamics/MDsims/HETs/backboneB.xtc',step=1,tf=10000)


select=pyDR.MolSelect(topo='/Users/albertsmith/Documents/Dynamics/MDsims.nosync/Y1/prot.pdb',
                      traj_files='/Users/albertsmith/Documents/Dynamics/MDsims.nosync/Y1/apo1.xtc',
                      step=1,t0=5900,tf=19150)

select=pyDR.MolSelect(topo='/Volumes/My Book/Y1/apo/prot.pdb',
                      traj_files='/Volumes/My Book/Y1/apo/apo1.xtc',
                      step=1,t0=5900,tf=19150)

select=pyDR.MolSelect(topo='/Volumes/My Book/GHSR/WT_apo.pdb',
                      traj_files='/Volumes/My Book/GHSR/WT-apo_run1_0.1ns_just_protein.xtc',
                      step=1,tf=100000)


pca=PCA(select)
pca.select_bond('15N')
# pca.select_bond('15N')
pca.select_atoms('name N C CA CB O HN')




ired=pyDR.md2iRED(select).iRED2data()
ired.detect.r_no_opt(10)
noopt=ired.fit()
noopt.detect.r_auto(6)
fit=noopt.fit().modes2bonds()
