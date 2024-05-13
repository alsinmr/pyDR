# -*- coding: utf-8 -*-



import pyDR

import pyDR
import os

from pyDR.Entropy.StateCounter import StateCounter

proj=pyDR.Project()

md_dir='/Volumes/My Book/Y1/apo'

topo=os.path.join(md_dir,'prot.pdb')
traj=os.path.join(md_dir,f'apo1.xtc')

select=pyDR.MolSelect(topo,traj,project=proj)
select.select_bond('15N')

SC=StateCounter(select)

