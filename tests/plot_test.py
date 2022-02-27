#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of Frames Theory Archive (FTA).

FTA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FTA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FTA.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de

Created on Tue Jul 13 13:42:37 2021

@author: albertsmith
"""

import numpy as np
import pyDR as DR
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import least_squares

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)


"This loads the MD trajectory into pyDIFRATE"
tf=5000
# molsys=DR.MolSys('/Users/albertsmith/Documents/GitHub/Frames_Theory_archive/HETs_ILE254.pdb',
#                  '/Users/albertsmith/Documents/GitHub/Frames_Theory_archive/HETs_ILE254.xtc',
#                  tf=tf)
# molsys=DR.MolSys('/Users/albertsmith/Documents/GitHub.nosync/Frames_Theory_archive/HETs_ILE254.pdb',
#               '/Users/albertsmith/Documents/GitHub.nosync/Frames_Theory_archive/HETs_ILE254.xtc',
#               tf=tf)
molsys=DR.MolSys('/Users/albertsmith/MDSimulations/HETs/HETs_5chain_B1.pdb',
                 '/Users/albertsmith/MDSimulations/HETs/HETs_5chain_MET_4pw_cb10_2micros.xtc',tf=tf)
select=DR.MolSelect(molsys)

#%% Define the frames

"We store the frames in a list, and then later load them into the frame analysis"
frames=list()
"""
'Type': Name of the function used for generating the frame 
(see pyDIFRATE/Struct/frames.py, pyDIFRATE/Struct/special_frames.py)
'Nuc': Shortcut for selecting a particular pair of nuclei. 
ivlal selects one group on every Ile, Val, Leu, Ala
'resids': List of residues to include (not required here – only one residue in trajectory)
'segids': List of segments to include (not required here – only one segment in trajectory)
'sigma': Time constant for post-processing (averaging of frame direction with Gaussian)
'n_bonds': Number of bonds away from methyl C–C group to define frame
"""

"Frames without post-process smoothing"
frames.append({'Type':'methylCC','Nuc':'ivlal','sigma':0,'segids':'B'})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':1,'sigma':0,'segids':'B'})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':2,'sigma':0,'segids':'B'})

"Frames with post-process smoothing"
frames.append({'Type':'hops_3site','Nuc':'ivlal','sigma':5,'segids':'B'})
frames.append({'Type':'methylCC','Nuc':'ivlal','sigma':5,'segids':'B'})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':1,'sigma':50,'segids':'B'})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':1,'sigma':50,'segids':'B'})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':2,'sigma':50,'segids':'B'})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':2,'sigma':50,'segids':'B'})

        

#%% Analyze with just one frame
select.select_bond(Nuc='ivlal',segids='B')   #Select 1 methyl group from all isoleucine, valine, leucine, and alanine residues
"""We could also specify the residue, but the provided trajectory just has one residue
A selection command populates mol.sel1 and mol.sel2 with atom groups, where sel1 and sel2 
then define a list of bonds
"""
fr_obj=DR.Frames.FrameObj(select)  #This creates a frame object based on the above molecule object
fr_obj.tensor_frame(sel1=1,sel2=2) #Here we specify to use the same bonds that were selected above in mol.select_atoms
#1,2 means we use for the first atom selection 1 in mol and for the second atom the second selection in mol

for f in frames:fr_obj.new_frame(**f) #This defines all frames in the frame object 
#(arguments were previously stored in the frames list)
fr_obj.load_frames(n=-1)  #This sweeps through the trajectory and collects the directions of all frames at each time point
fr_obj.post_process()   #This applies post processing to the frames where it is defined (i.e. sigma!=0)

"""
For each calculation, we only include some of the 9 frames that were defined above.

1) 3 rotational frames without post procesing (frames 0-2)
2) 3 rotational frames with averaging (frames 4,6,8)
3) 6 frames (rotational+hopping frames, frames 3-8)
"""
include=np.zeros([3,9],dtype=bool)
include[0][:3]=True    #Only methylCC,side_chain_chi frames without post processing
include[1][[4,6,8]]=True  #Only methylCC,side_chain_chi frames with post processing
include[2][3:]=True #All frames with post processing

t=np.arange(tf>>1)*.005     #Only plot the first half of the correlation function, where noise is lower

data=fr_obj.frames2data(include=include[-1],mode='full')


#%% Now use the project
import pyDR
from pyDR.Project import Project
proj=Project('/Users/albertsmith/Documents/Dynamics/test_project.nosync',create=True)

proj.append_data('HETs_13C.txt')
proj[0].select=pyDR.MolSelect(molsys)
resids=np.array([int(lbl[:3]) for lbl in proj[0].label])
proj[0].select.select_bond(Nuc='ivlal1',resids=resids,segids='B')
proj[0].detect.r_auto(5)
proj[0].detect.inclS2()
proj['NMR'].fit()

for d in data[:2]:proj.append_data(d)
proj['Frames'][0].detect.r_no_opt(10)
proj['Frames'].fit(bounds=False)
proj['Frames']['no_opt'][0].detect.r_target(proj[0].detect.rhoz,n=10)
proj['Frames']['no_opt'].fit()

dp=proj['NMR']['proc'][0].plot(style='bar',errorbars=True)
dp.append_data(proj['Frames']['proc'][0])

proj['Frames']['no_opt'][0].detect.r_auto(5)
proj['Frames']['no_opt'].fit()


dp=proj['Direct'][3].plot(errorbars=False,style='bar',index=np.arange(0,69,3),mode='b_in_a')
for k in range(1,2):
    dp.append_data(proj['Product'][3],style='bar',errorbars=True)
