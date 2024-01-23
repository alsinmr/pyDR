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
from pyDR.IO import write_file,read_file

class Movies():
    def __init__(self,data=None,molsys=None) -> None:
        """
        

        Parameters
        ----------
        data : pyDR.Data
            Data object from pyDR, where data is stored in a project
            (movies works from within the project folder, so this is required)
        molsys : pyDR.MolSys
            Alternatively, can providing a molsys, although then we may only see
            the molecule without any detector plotting.

        Returns
        -------
        None.

        """
        if data is None or data.select is None or data.select.sel1 is None:
            print("Warning: Selection must be defined to use some movie functions")
        
        if data is not None:
            self.data=data
            self.molsys=None
            self.project=data.source.project
        else:
            assert molsys is not None,"data or molsys must be defined to use movies"
            self.data=None
            self.molsys=molsys
            self.project=molsys.project
        self.CMX=self.project.chimera.CMX
        self.chimera=self.project.chimera
        self._xtcs=list()
        self._info=None
        self._select=None       
        self._settings={} #Save setting that we might need for recording.
        
        if not(os.path.exists(os.path.join(self.project.directory,'movies'))):
            os.mkdir(os.path.join(self.project.directory,'movies'))
        self.update()
    
    @property
    def current(self):
        return self.chimera.current
    
    @property
    def CMXid(self):
        return self.chimera.CMXid
    
    def defaultID(self,ID:int=None)-> int:
        """
        Returns a default ID for accessing a chimera instance, and ensures that
        the returned chimera ID corresponds to an active session. Note that this
        ID should be used for interacting with movies.chimera and the CMXid should
        be used for interacting with movies.CMX

        Parameters
        ----------
        ID : int, optional
            Input ID. The default is None.

        Returns
        -------
        int.
            ID of the chimera instance

        """
        if ID is None:
            if self.current is None:self.current=0
            ID=self.current
        else:
            self.current=ID
        return ID
    
    def __setattr__(self, name, value):
        if name=='current':
            self.chimera.current=value
            return
        super().__setattr__(name, value)
    
    @property
    def xtcs(self):
        self.update()
        return self._xtcs
    
    def update(self):

        #Update the xtc list
        if len(self._xtcs)<sum(['.xtc' in file for file in os.listdir(self.directory)]) or True:                
            self._xtcs=list()
            for fname in os.listdir(self.directory):
                if fname[-4:]=='.xtc':
                    self._xtcs.append(fname)
                    if not(os.path.exists(os.path.join(self.directory,fname[:-4]+'.txt'))):
                        print('Warning: Descriptor file for {0} is missing. Cannot play trajectory.'.format(fname))
        
        
            #Update the info field
            info=clsDict['Info']()
            flds=['Topo','Trajectory','Spacing','No. Frames','Initial Rate (ns/s)',
              'Final Rate (ns/s)','dt (ns)','Frame rate (frames/s)','No. Atoms']
            for f in flds:info.new_parameter(f)
            for file in self._xtcs:
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
                            elif key in ['No. Frames','Frame rate (frames/s)']:
                                dct[key]=int(value.strip())
                            elif key in info.keys:
                                dct[key]=float(value.strip())
                info.new_exper(**dct)
            self._info=info
    
    @property
    def directory(self):
        return os.path.join(self.project.directory,'movies')
    
    @property
    def command_line(self):
        return self.chimera.command_line

    #%% Load trajectory info
    @property
    def info(self):
        self.update()
        return self._info

    @property
    def valid_xtcs(self):
        out=list()
        for k,xtc in enumerate(self.xtcs):
            if self.data.select.molsys.topo==self.info['Topo'][k]:
                out.append(xtc)
        return out
    

    #%% Functions for making selections to show in movie
    def get_sel(self,select=None) -> mda.AtomGroup:
        """
        Returns the default selection for writing a trajectory. User may provide
        a string to filter for the desired selection (MDAnalysis formatting) or 
        an atomgroup directly.
        
        If no selection provided, then default behavior is applied, as defined
        below:
            1) If All atoms in the data object's selection are in the same 
                    residue, then only that residue will be written out.
            2) Otherwise, all segments in occuring in the data object's 
                selection are written out.
                
        Note that after the first run of get_sel, the result is saved and recycled.
        If the default behavior has been overridden via a previous call, set
        select='reset' to restore the defaults.

        Parameters
        ----------
        select : selection, optional
            Selection of atoms to write out. Can be a string, in which case it
            is applied to the MDAnalysis universe to filter atoms. May alternatively
            be an MDAnalysis atom group. To select all atoms in the universe,
            set select=''.
                

        Returns
        -------
        mda.AtomGroup.

        """
        
        if isinstance(select,str) and select=='reset':
            self._select=None
            select=None
        
        if self._select is None:
            "Atom group provided directly"
            if select is not None and str(select.__class__)==str(mda.AtomGroup):
                self._select=select
            elif isinstance(select,str): #Selection string provided
                if select=='': #Select all atoms in universe
                    self._select=self.data.select.sel0.uni.atoms
                else:
                    self._select=self.data.select.uni.select_atoms(select)
            else:
                sel0=self.data.select
                "Get all residues/segments"
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
                        self._select=sel0.sel1[0].residues[0].atoms         
                    else:
                        self._select=sel0.sel1.residues[0].atoms
                else:
                    i=np.array([s in segids for s in sel0.uni.segments.segids])
                    self._select=sel0.uni.segments[i].atoms
        return self._select
            

    def det_x_id(self,i:int=0):
        """
        Get the correct indices for the given xtc/pdb and also filter out any
        data values that do not have corresponding selections in the pdb.

        Parameters
        ----------
        i : int, optional
            Determine which xtc to use. Selected from the list returned by
            movies.valid_xtcs. The default is 0.

        Returns
        -------
        x,ids

        """
        sel0=read_file(os.path.join(self.directory,self.valid_xtcs[i][:-4]+'.sel'))
        sel0._mdmode=True
        ids=list()
        for sel in self.data.select.repr_sel:
            ids.append(list())
            for id0 in sel.ids:
                if id0 in sel0.sel1.ids:
                    ids[-1].append(np.argwhere(sel0.sel1.ids==id0)[0,0])     #This is the location of id0 in the truncated pdb
            ids[-1]=np.array(ids[-1])
        
        x=self.data.R
        x=x[np.array([True if len(id0) else False for id0 in ids],dtype=bool)]
                
        return x,ids

    #%% Play options
    def play_traj(self,i:int=0,ID:int=None):
        ID=self.defaultID(ID)
        CMXid=self.CMXid
        
        nm=self.chimera.CMX.how_many_models(CMXid)+1
        cmds=list()
        if not(len(self.valid_xtcs)):self.write_traj()
        xtc=os.path.join(self.directory,self.valid_xtcs[i])
        cmds.append("open '{0}' coordset true".format(xtc[:-4]+'.pdb'))
        cmds.append("open '{0}' structureModel #{1}".format(xtc,nm))
        cmds.append("~ribbon #{0}".format(nm))
        cmds.append("show #{0}".format(nm))
        cmds.append("coordset slider #{0}".format(nm))
        self._settings=self.info[self.xtcs.index(self.valid_xtcs[i])]
        self.chimera.command_line(cmds,ID=ID)
        
    def det_fader(self,rho_index=None,i:int=0,ID:int=None,scaling=None):
        ID=self.defaultID(ID) if ID is None else ID
        rho_index=np.arange(self.data.sens.rhoz.shape[0]) if rho_index is None else np.array(rho_index,dtype=int)
        self.play_traj(i=i,ID=ID)
        x,ids=self.det_x_id(i)
        x=x[:,rho_index]
        x[x<0]=0
        if scaling is None:scaling=1/x.max()
        x*=scaling
        info=self.info[self.xtcs.index(self.valid_xtcs[i])]
        keys=['Spacing','Initial Rate (ns/s)','Final Rate (ns/s)','dt (ns)','Frame rate (frames/s)','No. Frames']
        Spacing,nss0,nssf,dt,fr,nt=[info[k] for k in keys]
        
        if Spacing=='log':
            tau=log_axis(int(1e12),nt=nt,nss0=nss0,nssf=nssf,dt=dt,fr=fr,mode='step')*fr
        else:
            tau=lin_axis(int(1e12),nt=nt,nss=nss0,dt=dt,fr=fr,mode='step')*fr
        rhoz=self.data.sens.rhoz[rho_index]
        tc=self.data.sens.tc
        
        nm=self.chimera.CMX.how_many_models(self.chimera.CMXid)
        cmds=['set bgColor gray','lighting simple','lighting shadows false','sel #{0}'.format(nm),
              'color sel 82,71,55','style sel ball','size sel stickRadius 0.2',
              'size sel atomRadius 0.8','~sel']
        
        self.chimera.command_line(ID=ID,cmds=cmds)
        self.CMX.add_event(self.CMXid,'DetectorFader',x,ids,tau,rhoz,tc,3)
        
    def timescale_indicator(self):
        """
        Adds a timescale indicator to chimeraX. Applies to the last trajectory
        loaded (currently, multiple timescale indicators are not supported)

        Returns
        -------
        None.

        """
        info=self._settings
        keys=['Spacing','Initial Rate (ns/s)','Final Rate (ns/s)','dt (ns)','Frame rate (frames/s)','No. Frames']
        Spacing,nss0,nssf,dt,fr,nt=[info[k] for k in keys]
        
        if Spacing=='log':
            tau=log_axis(int(1e12),nt=nt,nss0=nss0,nssf=nssf,dt=dt,fr=fr,mode='step')*fr
        else:
            tau=lin_axis(int(1e12),nt=nt,nss=nss0,dt=dt,fr=fr,mode='step')*fr
            
        CMXid=self.CMXid
        
        self.CMX.add_event(self.CMXid,'TimescaleIndicator',tau)
        
    def record(self,filename:str) -> None:
        """
        If the last entry into chimeraX is a trajectory (with tensors, detectors,
        etc.), then this will play that trajectory and record the results into
        the project directory, using the provided filename. 

        Parameters
        ----------
        filename : str
            Name of movie file. Include ending ('.mp4','.avi', etc.) to specify
            file type (default is mp4).

        Returns
        -------
        None
            DESCRIPTION.

        """
        ID=self.defaultID()
        if filename[-4]!='.':filename=os.path.join(self.directory,filename+'.mp4')
        fr=self._settings['Frame rate (frames/s)']
        nt=self._settings['No. Frames']
        nm=self.chimera.CMX.how_many_models(self.chimera.CMXid)
        
        # cmds=['movie stop','movie record supersample 3','coordset #{0}\n'.format(nm),
        #       'movie encode "{0}" framerate {1}'.format(filename,fr)]
        # self.command_line(ID=ID,cmds=cmds)
        
        cxc=os.path.join(self.directory,'temp.cxc')
        with open(cxc,'w') as f:
            f.write('movie record\n')
            f.write('coordset #{}\n'.format(nm))
            f.write('wait {}\n'.format(nt))
            f.write('movie encode "{0}" framerate {1}\n'.format(filename,fr))
        
        self.chimera.command_line(ID=ID,cmds='open "{0}"'.format(cxc))
        return filename
            
           

        
    
    
    #%% Write trajectories
    def write_traj(self,select=None,spacing:str='log',t0:int=0,nt:int=450,nss0:float=0.005,nssf:float=500,fr:int=15):
        """
        

        Parameters
        ----------
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
            Initial frame rate, given in ns/s. The default is 0.005.
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
        
        atoms=self.get_sel(select=select)
        traj=self.data.select.traj
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
            f.write('Topo: {0}\n'.format(self.data.select.molsys.topo))
            f.write('Trajectory: ')
            for file in traj.files:f.write('{0}\t'.format(file))
            f.write('\n')
            f.write('Spacing: {0}\n'.format(spacing))
            f.write('nt: {0}\n'.format(len(index)))
            f.write('Initial Rate (ns/s): {0}\n'.format(Dt[1]*fr if len(Dt)>1 else 0))
            f.write('Final Rate (ns/s): {0}\n'.format(Dt[-1]*fr))
            f.write('dt (ns): {0}\n'.format(dt0))
            f.write('Frame rate (frames/s): {0}\n'.format(fr))
            f.write('No. Atoms: {0}\n'.format(len(atoms)))
            f.write('No. Frames: {0}\n'.format(filename.split('_')[-1].split('.')[0]))
        
        atoms.write(os.path.join(self.directory,filename+'.pdb'))
        sel=clsDict['MolSelect'](self.data.select.molsys)
        sel.sel1=atoms
        sel.sel2=atoms #Pretty hack-y should fix the need to do this
        write_file(os.path.join(self.directory,filename+'.sel'),sel,overwrite=True)
        self.update()
            
        
        


def lin_axis(nt0:int,nt:int=450,nss:float=0.02,dt:float=0.001,fr:int=15,mode='time') -> np.array:
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
    

def log_axis(nt0:int,nt:int=450,nss0:float=0.005,nssf:float=500,dt:float=0.005,fr:int=15,mode='time') -> np.array:
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
            
    