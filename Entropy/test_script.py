# -*- coding: utf-8 -*-



import pyDR
import os

from pyDR.Entropy.StateCounter import StateCounter

proj=pyDR.Project()


if os.path.exists('/Volumes/My Book/'):
    md_dir='/Volumes/My Book/Y1/apo'
    
    topo=os.path.join(md_dir,'prot.pdb')
    traj=os.path.join(md_dir,'apo1.xtc')
else:
    md_dir='/Users/albertsmith/Documents/Dynamics/HETs_Methyl/MDsim/'
    
    topo=os.path.join(md_dir,'HETs_3chain.pdb')
    traj=os.path.join(md_dir,'HETs_MET_4pw.xtc')

select=pyDR.MolSelect(topo,traj,project=proj)
select.select_bond('15N')
select.traj.step=1

SC=StateCounter(select)

SC.plotCC()