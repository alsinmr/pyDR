#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:18:23 2024

@author: albertsmith
"""

import pyDR

select=pyDR.MolSelect(topo='/Users/albertsmith/Documents/GitHub.nosync/pyDR/examples/HETs15N/backboneB.pdb',
                      traj_files='/Users/albertsmith/Documents/GitHub.nosync/pyDR/examples/HETs15N/backboneB.xtc',
                      step=100)


from pyDR.PCA.PCAclean import PCA

pca=PCA(select)

pca.select_bond('15N')
pca.select_atoms('name N C CA')

