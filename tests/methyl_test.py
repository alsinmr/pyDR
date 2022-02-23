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
tf=2000
molsys=DR.MolSys('/Users/albertsmith/Documents/GitHub/Frames_Theory_archive/HETs_ILE254.pdb',
                '/Users/albertsmith/Documents/GitHub/Frames_Theory_archive/HETs_ILE254.xtc',
                tf=tf)
#molsys=DR.MolSys('/Users/albertsmith/Documents/GitHub.nosync/Frames_Theory_archive/HETs_ILE254.pdb',
#              '/Users/albertsmith/Documents/GitHub.nosync/Frames_Theory_archive/HETs_ILE254.xtc',
#              tf=tf)
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
frames.append({'Type':'methylCC','Nuc':'ivlal','sigma':0})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':1,'sigma':0})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':2,'sigma':0})

"Frames with post-process smoothing"
frames.append({'Type':'hops_3site','Nuc':'ivlal','sigma':5})
frames.append({'Type':'methylCC','Nuc':'ivlal','sigma':5})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':1,'sigma':50})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':1,'sigma':50})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':2,'sigma':50})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':2,'sigma':50})


#%% Create a figure for plotting
titles=[['Methyl rot.',r'$\chi_2$ rot.',r'$\chi_1$ rot.',r'C$\alpha$-C$\beta$ motion','Total'],
        ['Methyl rot.',r'$\chi_2$ rot.',r'$\chi_1$ rot.',r'C$\alpha$-C$\beta$ motion','Total'],
        ['Methyl lib.','Methyl hop.',r'$\chi_2$ lib.',r'$\chi_2$ hop.',
          r'$\chi_1$ lib.',r'$\chi_1$ hop.',r'C$\alpha$-C$\beta$ motion','Total']]
fig=plt.figure('Methyl Dynamics')
fig.clear()
fig.set_size_inches([10.27,  9.03])
ax=[]
ax.append([fig.add_subplot(5,3,k+1) for k in range(0,15,3)])
ax.append([fig.add_subplot(5,3,k+1) for k in range(1,15,3)])
ax.append([fig.add_subplot(5,6,k+1) for k in [4,5,10,11,16,17,22]])
ax[-1].append(fig.add_subplot(5,3,15))

for a0,t0 in zip(ax,titles):
    for a,t in zip(a0,t0):
        a.set_title(t)
        

#%% Analyze with just one frame
select.select_bond(Nuc='ivlal')   #Select 1 methyl group from all isoleucine, valine, leucine, and alanine residues
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

for inc,ax0 in zip(include,ax):
    out=fr_obj.frames2ct(include=inc,mode='full')
    
    for ct,a in zip(out['ct_finF'],ax0):
        a.cla()
        a.plot(t,ct.mean(0)[:tf>>1])
        a.set_ylim([0,1.05])
        S2=ct.mean(0)[tf>>2:tf>>1].mean()
        b=np.argwhere(ct.mean(0)-S2<0)[0,0]
#        tc0=np.max([.001,((ct.mean(0)[:b]-S2)/(1-S2)).sum()*.005])
        tc0=t[np.argmin(np.abs((ct.mean(0)[:b]-S2)/(1-S2)-np.exp(-1)))]
        fun=lambda x:(((x[0]+(1-x[0])*np.exp(-t[:b]/x[1]))-ct.mean(0)[:b])**2).sum()
        S2,tc=least_squares(fun,[S2,tc0]).x
        a.plot(t,S2+(1-S2)*np.exp(-t/tc),color='grey',linestyle=':')
        a.set_xlim([0,np.min([10*tc,fr_obj.t[tf>>1]])])
    ax0[-1].semilogx(out['t'][:tf>>1],out['ct'].mean(0)[:tf>>1])
    ax0[-1].semilogx(out['t'][:tf>>1],out['ct_prod'].mean(0)[:tf>>1])
    ax0[-1].set_ylim([0,.5])
    
fig.set_size_inches([180/25.4,220/25.4])

plt.show()

data=fr_obj.frames2data(include=inc,mode='full')


#%% Now use the project
import pyDR
from pyDR.Project import Project
proj=Project('/Users/albertsmith/Documents/Dynamics/test_project.nosync',create=True)


for d in data:proj.append_data(d)
proj[0].detect.r_no_opt(10)
proj.fit()
proj['no_opt'][0].detect.r_auto(5)
proj['no_opt'].fit()


