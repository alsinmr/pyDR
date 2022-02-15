#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:34:36 2021

@author: albertsmith
"""
import numpy as np
import copy
from MDAnalysis import Universe
from pyDR.misc.ProgressBar import ProgressBar
from pyDR.Selection import select_tools as selt
from pyDR.MDtools.vft import pbc_corr
from pyDR import Defaults
dtype=Defaults['dtype']

class MolSys():
    """
    Object for storage of the molecule or MD trajectory
    """
    def __init__(self,topo=None,traj=None,t0=0,tf=-1,step=1,dt=None):
        self._uni=list()
        self._traj=list()
        self.cur_molecule=None #Allow multiple trajectories to be stored, select which one accessed with this index
        if topo:self.add_molecule(topo,traj,t0=t0,tf=tf,step=step,dt=dt)       
        
    
    def add_molecule(self,topo,traj=None,t0=0,tf=-1,step=1,dt=None,**kwargs):
        """Adds a molecule to the MolSys object
        """
        self._uni.append(Universe(topo,traj))
        self._traj.append(Trajectory(self._uni[-1].trajectory,t0=t0,tf=tf,step=step,dt=dt) \
                          if self._uni[-1].trajectory else None)
        self.cur_molecule=len(self._uni)-1
    
    def __getitem__(self,index):
        """
        Get a specific molecule from MolSys
        """
        out=copy.copy(self)
        assert index<len(self._uni),"Index exceeds number of molecules in MolSys"
        out.cur_molecule=index
        return out
    
    @property
    def uni(self):
        "Return the universe object corresponding to this element of the MolSys"
        assert len(self._uni)>0,'No molecules loaded'
        return self._uni[self.cur_molecule]
    
    @property
    def traj(self):
        return self._traj[self.cur_molecule]
    
    
    def __setattr__(self,name,value):
        if name=='cur_molecule':
            assert value is None or value<len(self._uni),'{} molecules are currently loaded'.format(len(self._uni))
        super().__setattr__(name,value)
        


class Trajectory():
    def __init__(self,traj,t0=0,tf=-1,step=1,dt=None):
        self.t0=t0
        self.__tf=len(traj)
        self.tf=tf
        self.step=step
        self.__dt=dt if dt else traj.dt
        self.traj=traj
        self.ProgressBar=False
            
    @property
    def dt(self):
        return self.__dt*self.step
    
    def __setattr__(self,name,value):
        "Make sure t0, tf, step are integers"
        if name in ['t0','tf','step']:
            value=int(value)
        if name=='tf':
            assert value<=self.__tf,"tf must be less than or equal to the original trajectory length ({} frames)".format(self.__tf)
            value%=self.__tf #Take care of negative indices
        super().__setattr__(name,value)
        
    def __getitem__(self,index):
        if np.array(index).dtype=='bool':
            i=np.zeros(self.tf-self.t0,dtype=bool)
            i[::self.step]=True
            index=np.concatenate((np.zeros(self.t0,dtype=bool),i,np.zeros(self.__tf-self.tf,dtype=bool)))
            return self.traj[index]
        elif isinstance(index,slice):
            stop=index.stop if index.stop else len(self)
            assert stop<=self.__len__(),'stop index must be less than or equal to the truncated trajectory length'
            start=index.start if index.start else 0
            
            stop=stop if stop==len(self) else stop%len(self)
            step=index.step if index.step else 1
            
            def iterate():
                for k in range(start,stop,step):
                    if self.ProgressBar:ProgressBar((k+1-start)*step,stop-step,'Loading:','',0,40)
                    yield self.traj[k]
            return iterate()
        elif hasattr(index,'__iter__'):
            def iterate():
                for m,k in enumerate(index):
                    if self.ProgressBar and hasattr(index,'__len__'):ProgressBar(m+1,len(index),'Loading:','',0,40)
                    yield self.traj[k]
            return iterate()
        else:
            assert index<self.__len__(),"index must be less than the truncated trajectory length"
            index%=self.__len__() #Take care of negative indices
            return self.traj[self.t0+index*self.step]
    
    def __len__(self):
        return int((self.tf-self.t0)/self.step)
    
    def __iter__(self):
        return self[:]
    
class MolSelect():
    def __init__(self,molsys):
        super().__setattr__('MolSys',molsys)
        self.__mol_no=molsys.cur_molecule
        self.sel0=None
        self.sel1=None
        self._repr_sel=None
    

    def __setattr__(self,name,value):
        """
        Use to set special behaviors for certain attributes
        """
        if name in ['MolSys']:
            print('Warning: Changing {} is not allowed'.format(name))
            return
        super().__setattr__(name,value)
        
    def select_bond(self,Nuc=None,resids=None,segids=None,filter_str=None):
        """
        Select a bond according to 'Nuc' keywords
            '15N','N','N15': select the H-N in the protein backbone
            'CO','13CO','CO13': select the CO in the protein backbone
            'CA','13CA','CA13': select the CA in the protein backbone
            'ivl','ivla','ch3': Select methyl groups 
                (ivl: ILE,VAL,LEU, ivla: ILE,LEU,VAL,ALA, ch3: all methyl groups)
                (e.g. ivl1: only take one bond per methyl group)
                (e.g. ivlr,ivll: Only take the left or right methyl group)
        """
        self.sel0,self.sel1=selt.protein_defaults(Nuc=Nuc,mol=self,resids=resids,segids=segids,filter_str=filter_str)
        
        if Nuc.lower() in ['15n','n15','n','co','13co','co13','ca','13ca','ca13']:
            self._repr_sel=list()
            for s in self.sel0:
                self._repr_sel.append(s.residue.atoms.select_atoms('name H HN N CA'))
                resi=s.residue.resindex-1
                if resi>=0 and self.uni.residues[resi].segid==s.segid:
                    self._repr_sel[-1]+=self.uni.residues[resi].atoms.select_atoms('name C CA')
        elif Nuc.lower()[:3] in ['ivl','ch3'] and '1' in Nuc:
            self._repr_sel=list()
            for s in self.sel0:
                self._repr_sel.append(s+s.residue.atoms.select_atoms('name H* and around 1.4 name {}'.format(s.name)))

    @property
    def repr_sel(self):
        if self.sel0 is None:
            print('Warning: No selection currently set')
            return 
        if self._repr_sel is None:
            if self.sel1 is not None:
                self._repr_sel=[s0+s1 for s0,s1 in zip(self.sel0,self.sel1)]
            else:
                self._repr_sel=[s0 for s0 in self.sel0]
        return self._repr_sel
    
    def set_selection(i,sel=None,resids=None,segids=None,fitler_str=None):
        """
        Define sel1 and sel2 separately. The first argument is 0 or 1, depending on
        which selection is to be set. Following arguments are:
            sel: Atom group, string
        """
        pass
    
    @property
    def box(self):
        return self.MolSys.uni.dimensions[:3]
    
    @property
    def uni(self):
        return self.MolSys.uni
    
    @property
    def traj(self):
        return self.MolSys.traj
    
    @property
    def pos(self):
        assert self.sel0 is not None,"First, perform a selection"
        if hasattr(self.sel0[0],'__len__'):
            return np.array([s.positions.mean(0) for s in self.sel0],dtype=dtype)
        else:
            return self.sel0.positions.astype(dtype)
    
    @property
    def v(self):
        assert self.sel0 is not None and self.sel1 is not None,'vec requires both sel1 and sel2 to be defined'
        if hasattr(self.sel0[0],'__len__') and hasattr(self.sel1[0],'__len__'):
            out=np.array([s0.positions.mean(0)-s1.positions.mean(0) for s0,s1 in zip(self.sel0,self.sel1)])
        elif hasattr(self.sel0[0],'__len__'):
            out=np.array([s0.positions.mean(0)-s1.position for s0,s1 in zip(self.sel0,self.sel1)])
        elif hasattr(self.sel1[0],'__len__'):
            out=np.array([s0.position-s1.positions.mean(0) for s0,s1 in zip(self.sel0,self.sel1)])
        else:
            out=self.sel0.positions-self.sel1.positions
        return pbc_corr(out.astype(dtype),self.box)
        
    def __len__(self):
        if self.sel0 is None:return 0
        return len(self.sel0)
        