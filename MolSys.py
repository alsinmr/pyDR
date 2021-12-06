#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:34:36 2021

@author: albertsmith
"""
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.coordinates.XTC import XTCReader

class MolSys():
    """
    Object for storage of the molecule or MD trajectory
    """
    def __init__(self,topo=None,traj=None,t0=0,tf=-1,step=1,dt=None):
        self._uni=list()
        self.cur_molecule=None #Allow multiple trajectories to be stored, select which one accessed with this index
        if topo:self.add_molecule(topo,traj,t0=t0,tf=tf,step=step,dt=dt)       
        
    
    def add_molecule(self,topo,traj=None,t0=0,tf=-1,step=1,dt=None,**kwargs):
        """Adds a molecule to the MolSys object
        """
        self._uni.append(Universe(topo))
        if traj:self._uni[-1].trajectory=Trajectory(traj,t0=t0,tf=tf,step=step,dt=dt,**kwargs)
        self.cur_molecule=len(self._uni)-1
        
    @property
    def molecule(self):
        assert len(self._uni)>0,'No molecules loaded'
        return self._uni[self.cur_molecule]
    
    @property
    def trajectory(self):
        assert hasattr(self.molecule,'trajectory'),'No trajectory loaded'
        return self.molecule.trajectory
    
    
    def __setattr__(self,name,value):
        if name=='cur_molecule':
            assert value is None or value<len(self._uni),'{} molecules are currently loaded'.format(len(self._uni))
        super().__setattr__(name,value)
  

class Trajectory(XTCReader):
    """
    Trajectory object used to override the slicing behavior of the MDAnalysis
    XTCReader object (allow one to initialize the trajectory while starting or
    ending at different frames, and also allow skipping frames). 
    """

    def __init__(self,filename,t0=0,tf=-1,step=1,dt=None,convert_units=True,sub=None,refresh_offsets=False,**kwargs):
        """
        Initialize the trajectory object. Here, one should  provide the original
        trajectory object (universe.trajectory, which is already initialized) 
        along with optional arguments t0,tf,step, and dt.
        
        t0:     Index of first time point (default 0)
        tf:     Index of last time point (default -1)
        step:   Step size between time points (default 1)
        dt:     Replace dt if stored incorrectly (ns)
                (Note: property dt will return this value multiplied by step)
        
        """
        super().__init__(filename,convert_units,sub,refresh_offsets,**kwargs)
        self.t0=t0
        self.__tf=super().__len__() #Real length of the trajectory
        self.tf=tf
        self.step=step

        self.__dt=dt if dt else super().dt
    
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
            return super().__getitem__(index)
        elif isinstance(index,slice):
            stop=index.stop if index.stop else len(self)
            assert stop<=self.__len__(),'stop index must be less than or equal to the truncated trajectory length'
            start=index.start if index.start else 0
            
            stop=stop if stop==len(self) else stop%len(self)
            step=index.step if index.step else 1
            
#            start=(index.start if index.start else 0)+self.t0
#            stop=index.stop if index.stop is not None and index.stop<=self.__len__() else self.__len__()
#            stop=(stop if stop==self.__len__() else stop%self.__len__())*self.step+self.t0
#            step=self.step*index.step if index.step else self.step
#            sl=slice(start,stop,step)
#            print(sl)
#            return super().__getitem__(sl)
            def iterate():
                for k in range(start,stop,step):
                    yield self[k]
            return iterate
        else:
            assert index<self.__len__(),"index must be less than the truncated trajectory length"
            index%=self.__len__() #Take care of negative indices
            return super().__getitem__(self.t0+index*self.step)
    
    def _apply_limits(self, frame):
        return frame
    
    def __len__(self):
        return int((self.tf-self.t0)/self.step)
    
    def __iter__(self):
        return self[::-1]
    
        
        
      