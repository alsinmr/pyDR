#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:01:46 2021

@author: albertsmith
"""

from pyDR.FRET import FRETefcy
from pyDR import MolSys


topo='/Volumes/My Book/Y1/eq7_1.gro'
traj='/Volumes/My Book/Y1/y1_apo_run1.xtc'
mol=MolSys(topo,traj,t0=0,tf=50000,step=100)
fret=FRETefcy(mol)


