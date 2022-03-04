#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:11:13 2022

@author: albertsmith
"""

import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')
import pyDR

proj=pyDR.Project.Project('/Users/albertsmith/Documents/Dynamics/HETs_Methyl_Loquet/pyDR_proc/project',create=True)
proj.append_data('HETs_13C.txt')
proj[0].del_data_pt(range(-5,0))

molsys=pyDR.MolSys('HETs_3chain.pdb','/Volumes/My Book/HETs/MDSimulation/HETs_MET_4pw.xtc',tf=100000,step=10)
select=pyDR.MolSelect(molsys)

resids=list()
for lbl in proj[0].label:
    resids.append([int(l[:3]) for l in lbl.split(',')])
    
sel1=list()
sel2=list()
for r in resids:
    for k,r0 in enumerate(r):
        select.select_bond(Nuc='ivla',resids=r0,segids='B')
        if select.sel1.residues.resnames[0]=='ILE':select.select_bond(Nuc='ivlal',resids=r0,segids='B')
        if k==0:
            sel1.append(select.sel1)
            sel2.append(select.sel2)
        else:
            sel1[-1]+=select.sel1
            sel2[-1]+=select.sel2
            
select.sel1=sel1
select.sel2=sel2
proj[0].select=select

sel1=select.sel1[0]
sel2=select.sel2[0]
for s1,s2 in zip(select.sel1[1:],select.sel2[1:]):
    sel1+=s1
    sel2+=s2
select=pyDR.MolSelect(molsys)
select.sel1=sel1
select.sel2=sel2

fr_obj=pyDR.Frames.FrameObj(select)
fr_obj.tensor_frame(sel1=1,sel2=2)

fr_obj.load_frames()

for f in fr_obj.frames2data():proj.append_data(f)

#%% Now do some processing
proj[0].detect.r_auto(5)
proj[0].detect.inclS2()
proj[1].detect.r_target(proj[0].detect.rhoz,n=10)
proj.fit()

proj[2:].plot(style='bar')