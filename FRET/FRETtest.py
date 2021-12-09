#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:01:46 2021

@author: albertsmith
"""

import pyDR
from pyDR.FRET import FRETefcy
from pyDR import MolSys


topo='/Volumes/My Book/Y1/eq7_1.gro'
traj='/Volumes/My Book/Y1/y1_apo_run1.xtc'
mol=MolSys(topo,traj,t0=0,tf=50000,step=1)

fret=FRETefcy(mol)

sel1,sel2=mol.get_pair(Nuc='CH3l1',filter_str='resname ILE')
sel1,sel2,sel3,sel4=sel1[:15],sel2[:15],sel1[15:],sel2[15:]
fret.set_dipole(Type='bond',sel1=sel1,sel2=sel2,sel3=sel3,sel4=sel4)

fret.load()