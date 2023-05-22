#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:21:54 2023

@author: albertsmith
"""

import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
import pyDR    #This adds some functionality to the matplotlib axes



id0=np.load('indexfile.npy',allow_pickle=True).item()

#%% Rearrange the index
index=dict()

for key,value in id0.items():
    if np.any(['_A' in k for k in value.keys()]):
        index[key+'_A']={'MET':value[key+'_A'],'chi1':value['chi1']}
        index[key+'_B']={'MET':value[key+'_B'],'chi1':value['chi1']}    #For ILE, this is the delta carbon
        if 'chi2' in value.keys():
            index[key+'_A']['chi2']=value['chi2']
            if 'LEU' in key:
                index[key+'_B']['chi2']=value['chi2']
    else:
        index[key]={'MET':value[key]}
        if 'chi1' in value.keys():
            index[key]['chi1']=value['chi1']
        if 'chi2' in value.keys():
            index[key]['chi2']=value['chi2']
            

#%% Load data
met=np.mod(np.load('dih_MET_4pw_10μs.npy'),360)


#%% Determine state of each angle (0, 1, or 2)
ref0=np.mod(met,120).mean(1)
ref_met=np.concatenate(([ref0],[ref0+120],[ref0+240]),axis=0)

state_met=np.argmin([np.abs(met.T-ref) for ref in ref_met],axis=0)

#%% Determine hop locations
hop_met=np.diff(state_met,axis=0).astype(bool)



#%% Evaluate mean density around a methyl group

key='ILE_219_A'
value=index[key]

n=5000

uni=mda.Universe('/Volumes/My Book/HETs/HETs_3chain.pdb','/Volumes/My Book/HETs/MDSimulation/HETs_MET_4pw.xtc')

counter=0
fig,ax=plt.subplots(12,4)
fig.set_size_inches([9.5,10.5])
ax=ax.flatten()

for key,value in index.items():
    if 'ILE' in key:
        if 'A' in key:
            name=['HG21','HG22','HG23']        
        else:
            name=['HD1','HD2','HD3']
    elif 'LEU' in 'key':
        if 'A' in key:
            name=['HD11','HD12','HD13']
        else:
            name=['HD21','HD22','HD23']
    elif 'VAL' in 'key':
        if 'A' in key:
            name=['HG11','HG12','HG13']
        else:
            name=['HG21','HG22','HG23']
    
    sel0=uni.select_atoms(f'segid B and resid {key[4:7]} and name {name[0]} {name[1]} {name[2]}')
    
    i=np.zeros(uni.atoms.positions.shape[0],dtype=bool)
    for s0 in sel0:
        i+=np.sqrt(((uni.atoms.positions-s0.position)**2).sum(1))<8
        
    sel=uni.atoms[i]-sel0.residues.atoms
    
    D=np.ones(len(uni.trajectory[::n]))*100
    
    for k,_ in enumerate(uni.trajectory[::n]):
        for s0 in sel0:
            d=np.sqrt(((s0.position-sel.positions)**2).sum(1))
            D[k]=min(d.min(),D[k])
 
    ax0=ax[counter]
    counter+=1
    ax0.plot(t[:len(D)],Rmet[value['MET']][:len(D)]/Rmet[value['MET']].mean())
    ax0.plot(t[:len(D)],D/D.mean())
        