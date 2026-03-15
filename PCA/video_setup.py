#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:38:46 2025

@author: albertsmith
"""


import os
import pyDR
import numpy as np
from time import time



topo='/Volumes/My Book/Y1/apo/prot.pdb'
traj='/Volumes/My Book/Y1/apo/apo3_whole.xtc'
sel=pyDR.MolSelect(topo,traj)


sel.traj.step=1
pca=pyDR.PCA.PCA(sel)

pca.select_atoms('name N C CA')
pca.select_bond('N')

m=pca.Movie


m.xtc_log_swp(nframes=900)

index=m.BondRef.R[:,-1]>.8
o=m.options
o.remove_event('DetFader')
o.DetFader(index=index,sc=3)

m.play_xtc()
