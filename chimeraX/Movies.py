#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:01:18 2022

@author: albertsmith
"""

import numpy as np
import MDAnalysis as mda
import os
from pyDR import clsDict

class Movies():
    def __init__(self,project) -> None:
        """
        

        Parameters
        ----------
        project : pyDR project
            Project from pyDR.

        Returns
        -------
        None.

        """
        self.project=project
        self.chimera=project.chimera
        self._xtcs=list()
        self._info=None
        if not(os.path.exists(os.path.join(self.project.directory,'movies'))):
            os.mkdir(os.path.join(self.project.directory,'movies'))
        self.update()
    
    @property
    def current(self):
        return self.chimera.current
    
    def __setattr__(self, name, value):
        if name=='current':
            self.chimera.current=value
            return
        super().__setattr__(name, value)
    
    @property
    def xtcs(self):
        return self._xtcs
    
    def update(self):
        #Update the xtc list
        self._xtcs=list()
        for fname in os.listdir(self.directory):
            if fname[-4:]=='.xtc':
                self._xtcs.append(fname)
                if not(os.path.exists(os.path.join(self.directory,fname[:-4]+'.txt'))):
                    print('Warning: Descriptor file for {0} is missing. Cannot play trajectory.'.format(fname))
        
        
        #Update the info field
        flds=['Topo','Trajectory','Spacing','nt','Initial Rate (ns/s)','Final Rate (ns/s)','Frame rate (frames/s)']
        info=clsDict['Info']()
        for f in flds:info.new_parameter(f)
        for file in self.xtcs:
            dct={f:None for f in flds}
            if os.path.exists(os.path.join(self.directory,file[:-4]+'.txt')):
                with open(os.path.join(self.directory,file[:-4]+'.txt'),'r') as f:
                    for line in f:
                        key=line.strip().split(':')[0]
                        value=line.strip().split(':')[1]
                        if key in ['Topo','Spacing']:
                            dct[key]=value.strip()
                        elif key=='Trajectory':
                            dct[key]=value.strip().split('\t')
                        elif key in ['nt','Frame rate (frames/s)']:
                            dct[key]=int(value.strip())
                        else:
                            dct[key]=float(value.strip())
            info.new_exper(**dct)
        self._info=info
    
    @property
    def directory(self):
        return os.path.join(self.project.directory,'movies')

    #%% Play options
    def play_traj(self,i:int=0,ID:int=None):
        if self.current is None:self.current=0
        ID=self.current
        nm=self.chimera.CMX.how_many_models(ID)+1
        cmds=list()
        xtc=os.path.join(self.directory,self.xtcs[i])
        cmds.append("open '{0}' coordset true".format(xtc[:-4]+'.pdb'))
        cmds.append("open '{0}' structureModel #{1}".format(xtc,nm))
        cmds.append("coordset slider #{0}".format(nm))
        print(cmds)
        self.chimera.command_line(cmds,ID=ID)
        
           
    #%% Load trajectory info
    @property
    def info(self):
        return self._info
        
    
    
    #%% Write trajectories
    def write_traj(self,i:int=0,select=None,spacing:str='log',t0:int=0,nt:int=450,nss0:float=0.02,nssf:float=200,fr:int=15):
        """
        

        Parameters
        ----------
        i : int, optional
            Index of the data set from which to create the movie. Data set needs
            to include an MD trajectory. Note that one may first index the project
            to create a subproject with only the desired trajectory and then
            call write_traj without i, since this will just take the first
            data object and its associated trajectory. The default is 0.
        select : selection, optional
            Selection of atoms to write out. Can be a string, in which case it
            is applied to the MDAnalysis universe to filter atoms. May alternatively
            be an MDAnalysis atom group. If not specified, select will undergo
            the following default behavior:
                1) All atoms in the data object's selection are in the same residue,
                    then only that residue will be written out.
                2) Otherwise, all segments in occuring in the data object's 
                    selection are written out.
        spacing : str, optional
            'log' or 'lin' for logarithmic or linear time spacing. The default
            is 'log'.
        t0 : int,optional
            Initial time to start the trajectory from
        nt : int, optional
            DESCRIPTION. The default is 450.
        nss0 : float, optional
            Initial frame rate, given in ns/s. The default is 0.02.
        nssf : float, optional
            Final frame rate, given in ns/s. The default is 200.
        dt : float, optional
            Time step of trajectory, given in ns. The default is 0.005.
        fr : int, optional
            Frame rate for the final movie in frames/s. The default is 15.

        Returns
        -------
        None

        """
        
        atoms=self.get_sel(i=i,select=select)
        traj=self.project[i].select.traj
        dt0=traj._Trajectory__dt/1000
        nt0=traj.mda_traj.__len__()
        
        
        if spacing.lower()=='log':
            index=log_axis(nt0=nt0-t0,nt=nt,nss0=nss0,nssf=nssf,dt=dt0,fr=fr,mode='index')+t0
            Dt=log_axis(nt0=nt0-t0,nt=nt,nss0=nss0,nssf=nssf,dt=dt0,fr=fr,mode='step')
        else:
            spacing='lin'
            index=lin_axis(nt0=nt0-t0,nt=nt,nss=nss0,dt=dt0,fr=fr,mode='index')+t0
            Dt=lin_axis(nt0=nt0-t0,nt=nt,nss=nss0,dt=dt0,fr=fr,mode='step')
            
        
        file0=os.path.split(traj.files[0])[1].rsplit('.',1)[0]
        filename='{0}_{1}_{2}'.format(file0,spacing,len(index))
        write_traj(atoms=atoms,traj=traj.mda_traj,filename=os.path.join(self.directory,filename+'.xtc'),index=index)
        with open(os.path.join(self.directory,filename+'.txt'),'w') as f:
            f.write('Topo: {0}\n'.format(self.project[i].select.molsys.topo))
            f.write('Trajectory: ')
            for file in traj.files:f.write('{0}\t'.format(file))
            f.write('\n')
            f.write('Spacing: {0}\n'.format(spacing))
            f.write('nt: {0}\n'.format(len(index)))
            f.write('Initial Rate (ns/s): {0}\n'.format(Dt[1]*fr))
            f.write('Final Rate (ns/s): {0}\n'.format(Dt[-1]*fr))
            f.write('Frame rate (frames/s): {0}\n'.format(fr))
        
        atoms.write(os.path.join(self.directory,filename+'.pdb'))
        self.update()
            
        
            
    def get_sel(self,i:int=0,select=None) -> mda.AtomGroup:
        """
        Returns the default selection for writing a trajectory. User may provide
        an integer to specify the data object and either a string to filter for
        the desired selection (MDAnalysis formatting) or an atomgroup directly.
        
        If no selection provided, then default behavior is applied

        Parameters
        ----------
        i : int, optional
            Index of the data set from which to create the movie. Data set needs
            to include an MD trajectory. Note that one may first index the project
            to create a subproject with only the desired trajectory and then
            call write_traj without i, since this will just take the first
            data object and its associated trajectory. The default is 0.
        select : selection, optional
            Selection of atoms to write out. Can be a string, in which case it
            is applied to the MDAnalysis universe to filter atoms. May alternatively
            be an MDAnalysis atom group. If not specified, select will undergo
            the following default behavior:
                1) All atoms in the data object's selection are in the same residue,
                    then only that residue will be written out.
                2) Otherwise, all segments in occuring in the data object's 
                    selection are written out.

        Returns
        -------
        mda.AtomGroup.

        """
        
        if select is not None and str(select.__class__)==str(mda.AtomGroup):
            return select

        
        sel0=self.project[i].select
        if sel0.sel1 is None and sel0.sel2 is None:
            return sel0.uni.atoms
        
        if isinstance(select,str):
            if select=='':return sel0
        
        #%% Get all residues/segments
        resids=list()
        segids=list()
        for s in sel0.sel1:
            if hasattr(s,'__len__'):
                resids.extend(s.residues.resids)
                segids.extend(s.segments.segids)
            else:
                resids.append(s.residue.resid)
                segids.append(s.segment.segid)
        if sel0.sel2 is not None:
            for s in sel0.sel2:
                if hasattr(s,'__len__'):
                    resids.extend(s.residues.resids)
                    segids.extend(s.segments.segids)
                else:
                    resids.append(s.residue.resid)
                    segids.append(s.segment.segid)
        if len(np.unique(resids))==1:
            if hasattr(self.sel1[0],'__len__'):
                return sel0.sel1[0].residues[0].atoms
            else:
                return sel0.sel1.residues[0].atoms
        
        i=np.array([s in segids for s in sel0.uni.segments.segids])
        return sel0.uni.segments[i].atoms


def lin_axis(nt0:int,nt:int=450,nss:float=0.02,dt:float=0.005,fr:int=15,mode='time') -> np.array:
    """
    Calculates time axes and corresponding indices for linear-spaced trajectory
    construction. 

    Parameters
    ----------
    nt0 : int
        Time points in the original trajectory.
    nt : int, optional
        DESCRIPTION. The default is 450.
    nss : float, optional
        Frame rate, given in ns/s. The default is 0.02.
    dt : float, optional
        Time step of trajectory, given in ns. The default is 0.005.
    fr : int, optional
        Frame rate for the final movie in frames/s. The default is 15.
    mode : str, optional
        Determine what to return. We can return the step size as a function of
        time ('step'), the time axis itself ('time'), or the index to apply
        to the time axis ('index')

    Returns
    -------
    np.array

    """
    dtpf=nss/fr
    t=np.arange(nt)*dtpf
    Dt=np.concatenate(([0],np.diff(t)))
    if t[-1]>dt*nt0:
        print('Warning: Trajectory not long enough to return desired length and frame rate')
        t=t[t<=dt*nt0]
        Dt=np.concatenate(([0],np.diff(t)))
        print('Only {0} out of {1} requested frames returned'.format(len(t),nt))
    if mode.lower()=='step':return Dt
    if mode.lower()=='time':return t
    return np.round(t/dt).astype(int)
    

def log_axis(nt0:int,nt:int=450,nss0:float=0.02,nssf:float=200,dt:float=0.005,fr:int=15,mode='time') -> np.array:
    """
    Calculates time axes and corresponding indices for log-spaced trajectory
    construction. 

    Parameters
    ----------
    nt0 : int
        Time points in the original trajectory.
    nt : int, optional
        DESCRIPTION. The default is 450.
    nss0 : float, optional
        Initial frame rate, given in ns/s. The default is 0.02.
    nssf : float, optional
        Final frame rate, given in ns/s. The default is 200.
    dt : float, optional
        Time step of trajectory, given in ns. The default is 0.005.
    fr : int, optional
        Frame rate for the final movie in frames/s. The default is 15.
    mode : str, optional
        Determine what to return. We can return the step size as a function of
        time ('step'), the time axis itself ('time'), or the index to apply
        to the time axis ('index')

    Returns
    -------
    np.array

    """
    dtpf0=nss0/fr
    dtpff=nssf/fr
    
    Dt=np.concatenate(([0],np.logspace(np.log10(dtpf0),np.log10(dtpff),nt-1)))
    
    t=np.cumsum(Dt)
    if t[-1]>dt*nt0:
        print('Warning: Trajectory not long enough to return desired step size and frame rate')
        t=t[t<=dt*nt0]
        Dt=np.concatenate(([0],np.diff(t)))
        print('Only {0} out of {1} requested frames returned'.format(len(t),nt))
        print('Final frame rate is {0} instead of {1}'.format(Dt[-1]*fr,nssf))
    
    if mode.lower()=='step':return Dt
    if mode.lower()=='time':return t
    
    t0=np.arange(nt0)*dt
    return np.array([np.argmin(np.abs(t1-t0)) for t1 in t],dtype=int)
    

def write_traj(atoms:mda.AtomGroup,traj,filename:str,index:list=None) -> None:
    """
    Writes out an xtc file from an MDanalysis universe (usually for playback in
    chimeraX). Provide the atom group, trajectory, filename, and index of the 
    frames to write out.


    Parameters
    ----------
    atoms : mda.AtomGroup
        Atom group from MDanalysis.
    traj : MDAnalysis or pyDR trajectory object
        Trajectory object that can be indexed/iterated over.
    filename : str
        Location to write out the traj file.
    index : list, optional
        List-like array of time points to include in the trajectory. 
        The default is None.

    Returns
    -------
    None.

    """
    index=np.array(index,dtype=int) if index is not None else np.arange(len(traj))
    
    with mda.Writer(filename,atoms.n_atoms) as W:
        for _ in traj[index]:
            W.write(atoms)
            
    