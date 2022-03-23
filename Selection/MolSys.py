#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:34:36 2021

@author: albertsmith
"""
import numpy as np
from MDAnalysis import Universe,AtomGroup
from pyDR.misc.ProgressBar import ProgressBar
from pyDR.Selection import select_tools as selt
from pyDR.MDtools.vft import pbc_corr
from pyDR import Defaults
from copy import copy
import os
dtype=Defaults['dtype']

class MolSys():
    """
    Object for storage of the molecule or MD trajectory
    """
    def __init__(self,topo=None,traj_files=None,t0=0,tf=None,step=1,dt=None):
        if traj_files is not None and not(isinstance(traj_files,list) and len(traj_files)==0):
            if isinstance(traj_files,list):
                traj_files=[os.path.abspath(tf) for tf in traj_files]
            else:
                traj_files=os.path.abspath(traj_files)
            self._uni=Universe(os.path.abspath(topo),traj_files)
        elif topo is not None:
            self._uni=Universe(topo)
        else:
            self._uni=None
            self._traj=None
            return
        self._traj=Trajectory(self.uni.trajectory,t0=t0,tf=tf,step=step,dt=dt) \
            if hasattr(self.uni,'trajectory') else None     
    
    @property
    def uni(self):
        "Return the universe object corresponding to this element of the MolSys"
        return self._uni
    
    @property
    def traj(self):
        return self._traj
    @property
    def topo(self):
        return self.uni.filename
        


class Trajectory():
    def __init__(self,mda_traj,t0=0,tf=None,step=1,dt=None):
        self.t0=t0
        self.__tf=len(mda_traj)
        self.tf=tf if tf is not None else self.__tf
        self.step=step
        self.__dt=dt if dt else mda_traj.dt
        self.mda_traj=mda_traj
        self.ProgressBar=False
            
    @property
    def dt(self):
        return self.__dt*self.step
    
    @property
    def time(self):
        return self.mda_traj.time
    
    def __setattr__(self,name,value):
        "Make sure t0, tf, step are integers"
        if name in ['t0','tf','step']:
            value=int(value)
        if name=='tf':
            assert value<=self.__tf,"tf must be less than or equal to the original trajectory length ({} frames)".format(self.__tf)
            value=(value-1)%self.__tf+1 #Take care of negative indices
        super().__setattr__(name,value)
        
    def __getitem__(self,index):
        if np.array(index).dtype=='bool':
            i=np.zeros(self.tf-self.t0,dtype=bool)
            i[::self.step]=True
            index=np.concatenate((np.zeros(self.t0,dtype=bool),i,np.zeros(self.__tf-self.tf,dtype=bool)))
            return self.mda_traj[index]
        elif isinstance(index,slice):
            stop=index.stop if index.stop else len(self)
            assert stop<=self.__len__(),'stop index must be less than or equal to the truncated trajectory length'
            start=index.start if index.start else 0
            
            stop=stop if stop==len(self) else stop%len(self)
            step=index.step if index.step else 1
            
            def iterate():
                for k in range(start,stop,step):
                    if self.ProgressBar:ProgressBar((k+1-start)*step,stop-step,'Loading:','',0,40)
                    yield self.mda_traj[k]
            return iterate()
        elif hasattr(index,'__iter__'):
            def iterate():
                for m,k in enumerate(index):
                    if self.ProgressBar and hasattr(index,'__len__'):ProgressBar(m+1,len(index),'Loading:','',0,40)
                    yield self.mda_traj[k]
            return iterate()
        else:
            assert index<self.__len__(),"index must be less than the truncated trajectory length"
            index%=self.__len__() #Take care of negative indices
            return self.mda_traj[self.t0+index*self.step]
    
    def __len__(self):
        return int((self.tf-self.t0)/self.step)
    
    def __iter__(self):
        return self[:]
    
    @property
    def files(self):
        if hasattr(self.mda_traj,'filenames'):
            return self.mda_traj.filenames
        else:
            return [self.mda_traj.filename] #Always return a list
    
class MolSelect():
    def __init__(self,molsys):
        super().__setattr__('molsys',molsys)
        self.sel1=None
        self.sel2=None
        self._repr_sel=None
        self._label=None
    

    def __setattr__(self,name,value):
        """
        Use to set special behaviors for certain attributes
        """
        if name in ['MolSys']:
            print('Warning: Changing {} is not allowed'.format(name))
            return
        
        if name=='label':
            super().__setattr__('_label',value)
            return
        
        if name=='repr_sel':
            if self.sel1 is not None and len(self.sel1)!=len(value):
                print('Warning: length of sel1 and repr_sel are not equal. This will cause errors in ChimeraX')
        if name in ['sel1','sel2','repr_sel'] and value is not None:
            if name=='repr_sel':name='_repr_sel'
            if isinstance(value,AtomGroup):
                super().__setattr__(name,value)
            elif isinstance(value,str):
                super().__setattr__(name,self.uni.select_atoms(value))
            else:
                out=np.zeros(len(value),dtype=object)
                for k,v in enumerate(value):out[k]=v
                super().__setattr__(name,out)
            return
        
        super().__setattr__(name,value)
        
    def __copy__(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        for f in ['sel1','sel2','_repr_sel','_label']:
            setattr(out,f,copy(getattr(self,f)))
        return out
        
        
    def select_bond(self,Nuc=None,resids=None,segids=None,filter_str=None,label=None):
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
        self.sel1,self.sel2=selt.protein_defaults(Nuc=Nuc,mol=self,resids=resids,segids=segids,filter_str=filter_str)
        
        repr_sel=list()        
        if Nuc.lower() in ['15n','n15','n','co','13co','co13','ca','13ca','ca13']:
            for s in self.sel1:
                repr_sel.append(s.residue.atoms.select_atoms('name H HN N CA'))
                resi=s.residue.resindex-1
                if resi>=0 and self.uni.residues[resi].segid==s.segid:
                    repr_sel[-1]+=self.uni.residues[resi].atoms.select_atoms('name C CA')
        elif Nuc.lower()[:3] in ['ivl','ch3'] and '1' in Nuc:
            for s in self.sel1:
                repr_sel.append(s+s.residue.atoms.select_atoms('name H* and around 1.4 name {}'.format(s.name)))
        else:
            for s1,s2 in zip(self.sel1,self.sel2):
                repr_sel.append(s1+s2)
        self.repr_sel=repr_sel
            
        self.set_label(label)

    @property
    def repr_sel(self):
        if self.sel1 is None:
            print('Warning: No selection currently set')
            return 
        
        if self._repr_sel is not None and len(self.sel1)!=len(self._repr_sel):
            self._repr_sel=None
        
        if self._repr_sel is None:
            if self.sel2 is not None:
                self.repr_sel=[s0+s1 for s0,s1 in zip(self.sel1,self.sel2)]
            else:
                self.repr_sel=[s0 for s0 in self.sel1]
        return self._repr_sel
    
    def set_selection(self,i,sel=None,resids=None,segids=None,fitler_str=None):
        """
        Define sel2 and sel2 separately. The first argument is 0 or 1, depending on
        which selection is to be set. Following arguments are:
            sel: Atom group, string
        """
        pass
    
    def del_sel(self,index) -> None:
        """
        Delete a bond or bonds from a list (argument is one or a list/array of
            integers)

        Parameters
        ----------
        index : int or int array (list or numpy array)
            Indices of the selections to be removed.

        Returns
        -------
        None
            DESCRIPTION.

        """
        if hasattr(index,'__len__'):
            index=np.sort(index)[::-1]
            for i in index:self.del_sel(i)
            return
        
        for f in ['sel1','sel2','_repr_sel','label']:
            v=getattr(self,f)
            if v is not None and len(v):
                setattr(self,f,np.delete(v,index))
                    
    
    @property
    def box(self):
        return self.molsys.uni.dimensions[:3]
    
    @property
    def uni(self):
        return self.molsys.uni
    
    @property
    def traj(self):
        return self.molsys.traj
    
    @property
    def pos(self):
        assert self.sel1 is not None,"First, perform a selection"
        if hasattr(self.sel1[0],'__len__'):
            return np.array([s.positions.mean(0) for s in self.sel1],dtype=dtype)
        else:
            return self.sel1.positions.astype(dtype)
    
    @property
    def v(self):
        assert self.sel1 is not None and self.sel2 is not None,'vec requires both sel2 and sel2 to be defined'
        if hasattr(self.sel1[0],'__len__') and hasattr(self.sel2[0],'__len__'):
            out=np.array([s0.positions.mean(0)-s1.positions.mean(0) for s0,s1 in zip(self.sel1,self.sel2)])
        elif hasattr(self.sel1[0],'__len__'):
            out=np.array([s0.positions.mean(0)-s1.position for s0,s1 in zip(self.sel1,self.sel2)])
        elif hasattr(self.sel2[0],'__len__'):
            out=np.array([s0.position-s1.positions.mean(0) for s0,s1 in zip(self.sel1,self.sel2)])
        else:
            out=self.sel1.positions-self.sel2.positions
        return pbc_corr(out.astype(dtype),self.box)
        
    def __len__(self):
        if self.sel1 is None:return 0
        return len(self.sel1)
    
    @property
    def label(self):
        if self._label is None or len(self._label)!=len(self):
            self.set_label()
        return self._label
    
    def set_label(self,label=None):
        "We attempt to generate a unique label for this selection, while having labels of minimum length"
        if label is None:
            "An attempt to generate a unique label under various conditions"
            if hasattr(self.sel1[0],'__len__'):
                label=list()
                for s in self.sel1:     #See if each atom group corresponds to a single residue
                    if np.unique(s.resids).__len__()==1:
                        label.append(s.resids[0])
                    else:               #If more than residue, then
                        break
                else:
                    label=np.array(label)
                    if np.unique(label).size==label.size:   #Keep this label if all labels are unique
                        self.label=np.array(label)
                        return
                
                label=list()
                for s in self.sel1:     #See if each atom group corresponds to a single segment
                    if np.unique(s.segids).__len__()==1:
                        label.append(s.segids[0])
                    else:               #If more than residue, then
                        break
                else:
                    
                    label=np.array(label)
                    if np.unique(label).size==label.size:
                        self.label=np.array(label)
                        return
                self.label=np.arange(self.sel1.__len__())
                
            else:
                count,cont=0,True
                while cont:
                    if count==0: #One bond per residue- just take the residue number
                        label=self.sel1.resids  
                    elif count==1: #Multiple segments with same residue numbers
                        label=np.array(['{0}_{1}'.format(s.segid,s.resid) for s in self.sel1])
                    elif count==2: #Same segment, but multiple bonds on the same residue (include names)
                        label=np.array(['{0}_{1}_{2}'.format(s1.resid,s1.name,s2.name) for s1,s2 in zip(self.sel1,self.sel2)])
                    elif count==3: #Multiple bonds per residue, and multiple segments
                        label=np.array(['{0}_{1}_{2}_{3}'.format(s1.segid,s1.resid,s1.name,s2.name) \
                                        for s1,s2 in zip(self.sel1,self.sel2)])
                    "We give up after this"
                    count+=1
                    if np.unique(label).size==label.size or count==4:
                        cont=False
                    
                self.label=label
        
    def copy(self):
        return copy.copy(self)
        
        
    def compare(self,sel,mode='auto'):
        """
        Returns indices such that
        in12: self.sel1[in12],self.sel2[in12] is in sel.sel1,sel.sel2
        in21: sel.sel1[in21],self.sel2[in21] is in self.sel1,self.sel2
        
        Then, self.sel1[in12],self.sel1[in12] and self.sel2[in21],self.sel2[in21]
        are the same set (furthermore, they are in the same order. Selection in
        sel may be resorted to achieve ordering)
        
        Several modes of comparison exist:
            exact:  Requires that both selections are based of the same topology file,
                    and each selection has the same atoms in it
            a0: Requires that segids, resids, and atom names match for all selections
            a1: Requires that resids and atom names match for all selections
            a2: Requires that segids, resids, and atom names match for sel1 
            a3: Requires that resids, and atom names match for sel1             
            auto:Toggles between a0 and exact depending if the topology files match
        """
        if self is sel:return np.arange(len(self)),np.arange(len(self)) #Same selection
        
        assert hasattr(sel,'__class__') and str(self.__class__)==str(sel.__class__),"Comparison only defined for other MolSelect objects"
        assert self.sel1 is not None,"Selection not defined in self"
        assert sel.sel1 is not None,"Selection not defined in sel1"
        assert mode in ['exact','auto','a0','a1','a2','a3'],"mode must be 'auto','exact','a0','a1','a2', or 'a3'"
        
        if self.sel2 is not None and sel.sel2 is None:return np.zeros(0,dtype=int),np.zeros(0,dtype=int)
        if self.sel2 is None and sel.sel2 is not None:return np.zeros(0,dtype=int),np.zeros(0,dtype=int)
        
        if mode=='auto':
            mode='exact' if self.uni.filename==sel.uni.filename else 'a0'
        if self.sel2 is None:
            mode='a2' if mode in ['a0','a2'] else 'a3'
        
        if mode=='exact':
            if self.sel2 is not None:
                id11,id12,id21,id22=[[np.sort(s.ids) if hasattr(s,'ids') else s.id for s in sel] \
                      for sel in [self.sel1,self.sel2,sel.sel1,sel.sel2]]
                id1=[(i0,i1) for i0,i1 in zip(id11,id12)]
                id2=[(i0,i1) for i0,i1 in zip(id21,id22)]
            else:
                id1,id2=[[np.sort(s.ids) if hasattr(s,'ids') else s.id for s in sel] \
                      for sel in [self.sel1,sel.sel1]]
        elif mode in ['a0','a1']:
            id1=list()
            id2=list()
            for id0,sel0 in zip([id1,id2],[self,sel]):
                for s1,s2 in zip(sel0.sel1,sel0.sel2):
                    if hasattr(s1,'segids') and hasattr(s2,'segids'):
                        id0.append((s1.segids,s1.resids,s1.names,s2.segids,s2.resids,s2.names))
                    elif hasattr(s1,'segids'):
                        id0.append((s1.segids,s1.resids,s1.names,s2.segid,s2.resid,s2.name))
                    elif hasattr(s2,'segids'):
                        id0.append((s1.segid,s1.resid,s1.name,s2.segids,s2.resids,s2.names))
                    else:
                        id0.append((s1.segid,s1.resid,s1.name,s2.segid,s2.resid,s2.name))
            if mode=='a1':
                id1,id2=[[(*x[1:3],*x[4:6]) for x in id0] for id0 in [id1,id2]]
        else:
            id1=list()
            id2=list()
            for id0,sel0 in zip([id1,id2],[self,sel]):
                for s1 in sel0.sel1:
                    if hasattr(s1,'segids'):
                        id0.append((s1.segids,s1.resids,s1.names))
                    else:
                        id0.append((s1.segid,s1.resid,s1.name))
            if mode=='a3':
                id1,id2=[[x[1:] for x in id0] for id0 in [id1,id2]]

        in12=list()
        for i in id1:
            in12.append(np.any([np.array_equal(np.sort(i),np.sort(i1)) for i1 in id2]))
        in12=np.argwhere(in12)[:,0]
        
        in21=np.array([(np.argwhere([np.array_equal(np.sort(id1[k]),np.sort(i2)) for i2 in id2])[0,0]) for k in in12])
            
            
        # in21=np.array([np.argwhere([id1[k]==i2 for i2 in id2])[0,0] for k in in12])
                      
        return in12,in21
    
    def __eq__(self,sel):
        if sel is self:return True
        in21=self.compare(sel,mode='auto')[1]
        return len(in21)==len(self) and np.all(in21==np.sort(in21))
                
        
        