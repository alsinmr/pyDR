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
                      traj_files='/Users/albertsmith/Documents/Dynamics/MDsims/HETs/backboneB.xtc',step=100)

pca=PCA(select)
pca.select_bond('15N',resids=np.concatenate([np.arange(225,248),np.arange(261,281)]))
pca.select_atoms('name N C CA')

