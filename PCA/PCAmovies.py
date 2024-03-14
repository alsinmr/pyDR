#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:57:46 2024

@author: albertsmith
"""

import numpy as np
from ..chimeraX.MovieTools import lin_axis,log_axis,timescale_swp
from MDAnalysis import Writer
from copy import copy
import os

class PCAmovies():
    def __init__(self,pca):
        """
        Creates trajectories based on principal component analysis.
        
        Provide the PCA object and optionally a data object with fits of the
        principal component (results of fitting pca.PCA2data())

        Parameters
        ----------
        pca : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.pca=pca
        self._n=None
        self._PCARef=None
        self._BondRef=None
        self._zrange=None
        self._timescale=None
        self._xtc=None
        self._select=None
        self._options=Options(self)
        self._thread=None
    
    
    @property
    def n(self):
        """
        Default number of detectors used to analyze a trajectory. Can also be
        set by the user

        Returns
        -------
        int

        """
        if self._n is None:
            return self.pca.n
    
    @n.setter
    def n(self,n):
        self._n=n
    
    @property
    def PCARef(self):
        """
        Processed PCA detector analysis. 

        Returns
        -------
        data
            data object containing a detector analysis of the PCA.

        """
        if self._PCARef is None:
            return self.pca.Data.PCARef
            
        return self._PCARef

    
    @PCARef.setter
    def data(self,data):
        self._PCARef=data
        self._zrange=None
    
    @property
    def BondRef(self):
        """
        Data object containing analysis of the bond motion (no PCA). 

        Returns
        -------
        None.

        """
        if self._BondRef is None:
            return self.pca.Data.BondRef
        return self._BondRef
    
    @BondRef.setter
    def BondRef(self,data):
        self._BondRef=data
    
    @property
    def sens(self):
        """
        Reference to the data's sensitivity object

        Returns
        -------
        sens

        """
        return self.PCARef.sens
    
    @property
    def select(self):
        """
        Selection object that only has atoms included in the pca

        Returns
        -------
        MolSelect

        """
        if self._select is None:
            self._select=self.pca.select.new_molsys_from_sel(self.pca.atoms)
        return self._select
    
    @property
    def molsys(self):
        """
        Molsys with only atoms included in the pca

        Returns
        -------
        MolSys

        """
        return self.select.molsys
    
    @property
    def mdlnums(self):
        return self.molsys.movie.mdlnums
    
    def command_line(self,cmds):
        self.molsys.movie.command_line(cmds)
    
    @property
    def directory(self):
        """
        Reference to the molsys default directory

        Returns
        -------
        str

        """
        return self.molsys.directory
    
    @property
    def options(self):
        return self._options
    
    @property
    def zrange(self):
        """
        Default zrange for the given data object

        Returns
        -------
        tuple
        2 elements specifying the z-range for this data object

        """
        if self._zrange is None:
            rhoz=self.sens.rhoz[0]
            
            if rhoz[0]/rhoz.max()>0.9:
                i=np.argmin(np.abs(rhoz/rhoz.max()-0.75))
                zrange=[self.sens.z[i]]
            else:
                zrange=[self.sens.info['z0'][0]]
                
            rhoz=self.sens.rhoz[-1]
            if rhoz[-1]/rhoz.max()>0.9:
                i=np.argmin(np.abs(rhoz/rhoz.max()-0.75))
                zrange.append(self.sens.z[i])
            else:
                zrange.append(self.sens.info['z0'][-1])
                
            self._zrange=zrange
        return self._zrange
    
    @zrange.setter
    def zrange(self,zrange):
        assert len(zrange)==2,'zrange must have two elements'
        self._zrange=zrange
    
    
    def setup_swp(self,zrange=None,nframes:int=300,framerate:int=15):
        """
        Prepares time axes required for sweeping the timescale. zrange may
        be provided, but otherwise it will be calculated from the stored
        sensitivity object. nframes defines the number of frames used in the
        sweep and framerate defines how many frames that will be displayed per
        second

        Parameters
        ----------
        zrange : tuple/list, optional
            Two elements given the range of correlation times (as log values)
            The default is None, which will use internal values
        nframes : int, optional
            Number of frames in trajectory. The default is 150.
        framerate : int, optional
            Framerate for movie. The default is 15.

        Returns
        -------
        dict with keys
            tscale_swp : sampled timescales
            t: time axis for the trajectory
            index: frame index for the trajectory

        """
        out={}
        if zrange is None:zrange=self.zrange
        out['tscale_swp']=timescale_swp(zrange=zrange,nframes=nframes)     
        out['t'],out['index']=log_axis(traj=self.pca.traj,zrange=zrange,nframes=nframes,framerate=framerate)
        
        return out

    # def weight(self,timescale):
    #     """
    #     Returns a weighting of the principal components for a given timescale
    #     or timescales.
        
    #     This is determined from the stored sensitity object and the relative
    #     amplitudes of the detectors at the given timescale. 

    #     Parameters
    #     ----------
    #     timescale : float,array
    #         timescale or timescales given on a log-scale. The default is None.

    #     Returns
    #     -------
    #     np.array

    #     """
    #     if timescale is None:
    #         timescale=timescale_swp(self.zrange)
    #     timescale=np.atleast_1d(timescale)
        
    #     i=np.array([np.argmin(np.abs(ts-self.sens.z)) for ts in timescale],dtype=int)
    #     wt0=self.sens.rhoz[:,i]
    #     # wt0/=wt0.sum(0)
        
    #     return self.data.R@wt0
    
    #%% xtc writers
    def xtc_from_weight(self,wt,rho_index:int,filename:str='temp.xtc',
                        nframes:int=150,framerate:int=15,pc_scaling:float=1):
        """
        General function for calculating the xtc from a weighting of principal components

        Parameters
        ----------
        wt : TYPE
            DESCRIPTION.
        rho_index : int
            DESCRIPTION.
        filename : str, optional
            DESCRIPTION. The default is 'temp.xtc'.
        nframes : int, optional
            DESCRIPTION. The default is 150.
        framerate : int, optional
            DESCRIPTION. The default is 15.
        pc_scaling : float, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        
        
        timescale=self.PCARef.sens.info['z0'][rho_index]
        step=np.round(10**timescale*1e12/self.PCARef.source.select.traj.dt/framerate).astype(int)
        if step==0:step=1
        if step*nframes>len(self.pca.traj):
            step=np.round(len(self.pca.traj)/nframes).astype(int)
            
        Delta=np.concatenate([np.zeros([wt.sum(),1]),
                              np.diff(self.pca.PCamp[wt,:step*nframes:step],axis=1)],axis=1)
        pos=self.pca.pos[0]+pc_scaling*np.cumsum(self.pca.PCxyz[:,:,wt]@Delta,axis=-1).T
        
        
        ag=copy(self.pca.atoms)
        with Writer(filename, n_atoms=len(ag)) as w:
            for pos0 in pos:
                # ag.positions=self.pca.mean+((wt[:,k]*self.pca.PCamp[:,i[k]])*self.pca.PCxyz).sum(-1).T
                ag.positions=pos0
                w.write(ag)
                
        self._xtc=filename
        
        
        return self
    
    def xtc_from_PCamp(self,PCamp,rho_index:int,filename:str='temp.xtc',nframes:int=150,
                       framerate:int=15,pc_scaling:float=1):
        
        
        timescale=self.PCARef.sens.info['z0'][rho_index]
        step=np.round(10**timescale*1e12/self.PCARef.source.select.traj.dt/framerate/3).astype(int)
        # step=1
        if step==0:step=1
        if step*nframes>PCamp.shape[1]:
            step=np.round(PCamp.shape[1]/nframes).astype(int)
        
        pos=self.pca.mean+pc_scaling*(self.pca.PCxyz@PCamp[:,:nframes*step:step]).T
        
        ag=copy(self.pca.atoms)
        with Writer(filename, n_atoms=len(ag)) as w:
            for pos0 in pos:
                # ag.positions=self.pca.mean+((wt[:,k]*self.pca.PCamp[:,i[k]])*self.pca.PCxyz).sum(-1).T
                ag.positions=pos0
                w.write(ag)
                
        self._xtc=filename
        
        self.options.TimescaleIndicator(tau=np.ones(nframes)*step*self.pca.select.traj.dt/1e3)
        
        return self
                                      
                                      
        
    
    
    #%% Display modes
    def xtc_rho(self,rho_index:int,frac:float=0.75,filename:str='temp.xtc',
                      nframes:int=150,framerate:int=15,pc_scaling:float=1):
        """
        

        Parameters
        ----------
        rho_index : int
            DESCRIPTION.
        frac : float, optional
            DESCRIPTION. The default is 0.75.
        filename : str, optional
            DESCRIPTION. The default is 'temp.xtc'.
        nframes : int, optional
            DESCRIPTION. The default is 150.
        framerate : int, optional
            DESCRIPTION. The default is 15.
        pc_scaling : float, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        
        wt=self.pca.Weighting.rho_spec(rho_index=rho_index,PCAfit=self.PCARef,frac=frac)
        
        self.xtc_from_weight(wt=wt,rho_index=rho_index,filename=filename,
                        nframes=nframes,framerate=framerate,pc_scaling=pc_scaling)
        
        return self
    
    def xtc_bond(self,index:int,rho_index:int,frac:float=0.75,filename:str='temp.xtc',
                 nframes:int=150,framerate:int=15,pc_scaling:float=1):
        """
        

        Parameters
        ----------
        index : int
            DESCRIPTION.
        rho_index : int, optional
            DESCRIPTION. The default is None.
        frac : float, optional
            DESCRIPTION. The default is 0.75.
        filename : str, optional
            DESCRIPTION. The default is 'temp.xtc'.
        nframes : int, optional
            DESCRIPTION. The default is 300.
        framerate : int, optional
            DESCRIPTION. The default is 15.

        Returns
        -------
        None.

        """


        
        wt=self.pca.Weighting.bond(index=index,rho_index=rho_index,PCAfit=self.PCARef,frac=frac)
        
        self.xtc_from_weight(wt=wt,rho_index=rho_index,filename=filename,
                        nframes=nframes,framerate=framerate,pc_scaling=pc_scaling)

        return self
    
    def xtc_impulse(self,index:int,rho_index:int,frac:float=0.75,filename:str='temp.xtc',
                 nframes:int=150,framerate:int=15,pc_scaling:float=1):
        PCamp=self.pca.Impulse.PCamp_bond(index)[:,0:]
        wt=self.pca.Weighting.bond(index=index,rho_index=rho_index)
        
        self.xtc_from_PCamp(PCamp=(PCamp.T*wt).T, rho_index=rho_index,filename=filename,
                            nframes=nframes,framerate=framerate,pc_scaling=pc_scaling)
        
        self.options.commands=['~ribbon #{0}','show #{0}']
        
        return self
    
    def xtc_noweight(self,rho_index:int=None,mode_index:int=None,filename:str='temp.xtc',
                     nframes:int=150,framerate:int=15):
        """
        Produces an unweighted xtc (all modes), for comparison to weighted xtcs

        Parameters
        ----------
        rho_index : int, optional
            DESCRIPTION. The default is None.
        mode_index : int, optional
            DESCRIPTION. The default is None.
        filename : str, optional
            DESCRIPTION. The default is 'temp.xtc'.
        nframes : int, optional
            DESCRIPTION. The default is 150.
        framerate : int, optional
            DESCRIPTION. The default is 15.

        Returns
        -------
        None.

        """
        
        if mode_index is not None:
            rho_index=np.argmax(self.PCARef.R[mode_index])
        
        assert rho_index is not None,"rho_index or mode_index must be specified"
        timescale=self.sens.info['z0'][rho_index]
        
        step=np.round(10**timescale*1e12/self.PCARef.source.select.traj.dt/framerate).astype(int)
        if step==0:step=1
        if step*nframes>len(self.pca.traj):
            step=np.round(len(self.pca.traj)/nframes).astype(int)
            
        ag=copy(self.pca.atoms)
        with Writer(filename, n_atoms=len(ag)) as w:
            for pos0 in self.pca.pos[::step]:
                ag.positions=pos0
                w.write(ag)
                
        self._xtc=filename
        
        return self

            
        
    def xtc_mode(self,mode_index,filename:str='temp.xtc',nframes:int=150,
                 framerate:int=15,pc_scaling:float=1):
        """
        Displays a specific mode

        Parameters
        ----------
        mode_index : TYPE
            DESCRIPTION.
        filename : str, optional
            DESCRIPTION. The default is 'temp.xtc'.
        nframes : int, optional
            DESCRIPTION. The default is 150.
        framerate : int, optional
            DESCRIPTION. The default is 15.

        Returns
        -------
        None.

        """
        
        rho_index=np.argmax(self.PCARef.R[mode_index])
        timescale=self.sens.info['z0'][rho_index]
        step=np.round(10**timescale*1e12/self.PCARef.source.select.traj.dt/framerate).astype(int)
        if step==0:step=1
        if step*nframes>len(self.pca.traj):
            step=np.round(len(self.pca.traj)/nframes).astype(int)
            
        Delta=np.concatenate([np.zeros(1),
                  np.diff(self.pca.PCamp[mode_index,:step*nframes:step],axis=0)],axis=0)
        pos=self.pca.pos[0]+pc_scaling*np.cumsum(np.atleast_3d(self.pca.PCxyz[:,:,mode_index])@\
                                      np.atleast_2d(Delta),axis=-1).T
        ag=copy(self.pca.atoms)
        with Writer(filename, n_atoms=len(ag)) as w:
            for pos0 in pos:
                # ag.positions=self.pca.mean+((wt[:,k]*self.pca.PCamp[:,i[k]])*self.pca.PCxyz).sum(-1).T
                ag.positions=pos0
                w.write(ag)
                
        self._xtc=filename
        
        return self

        
    
    def xtc_log_swp(self,filename:str='temp.xtc',zrange=None,nframes:int=300,framerate:int=15):
        """
        Create an xtc file for a log sweep of the timescales. Setup is the output
        of setup_swp, which can be omitted if default values are desired.
        
        filename is the output filename of the xtc. If filetype (eg. .xtc) is
        omitted, then .xtc will be appended. It will be put into the molsys
        temporary folder by default

        Parameters
        ----------
        filename : str, optional
            DESCRIPTION. The default is 'temp.xtc'.
        setup : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        self

        """
        self.options.clear()
        
        setup=self.setup_swp(zrange=zrange,nframes=nframes,framerate=framerate)
        self.setup=setup
        
        wt=self.pca.Weighting.timescale(setup['tscale_swp'])
        # wt=(wt.T/wt.mean(-1)).T
        i=setup['index']
        
        ag=copy(self.pca.atoms)
        
        filename=os.path.join(self.directory,filename)
        
        pos0=pos=copy(self.pca.pos[0])
        DelPC=np.concatenate([np.zeros([self.pca.PCamp.shape[0],1]),
                              np.diff(self.pca.PCamp[:,i],axis=-1)],axis=-1)*wt
        
        PCcorr=self.pca.PCamp[:,i[-1]]-self.pca.PCamp[:,0]-DelPC.sum(-1)
        
        
        PCamp=np.cumsum(DelPC,axis=-1)+np.atleast_2d(PCcorr).T@np.atleast_2d(i)/i[-1]
        
        pos=self.pca.pos[0]+(self.pca.PCxyz@PCamp).T
        # pos=self.pca.pos[0].T+np.array([(self.pca.PCxyz*a).sum(-1) for a in PCamp.T])
        
        
        
        # pos=self.pca.pos[0]+(np.cumsum(DelPC,axis=-1)*self.pca.PCxyz).sum(-1).T+\
        #     +(np.atleast_2d(PCcorr).T@np.atleast_2d(i)/i[-1])
        
        # pos=self.pca.pos[0]+np.array([])
        
        
        
        
        with Writer(filename, n_atoms=len(ag)) as w:
            for pos0 in pos:
                # ag.positions=self.pca.mean+((wt[:,k]*self.pca.PCamp[:,i[k]])*self.pca.PCxyz).sum(-1).T
                ag.positions=pos0
                w.write(ag)
                
        self._xtc=filename
        self.options.DetFader().TimescaleIndicator()
        
        return self
        
    def play_xtc(self):
        """
        Plays the last xtc created. Will also create the xtc if none is found
        saved

        Returns
        -------
        self

        """
        if self._xtc is None:self.xtc_log_swp()
        self.molsys.new_trajectory(self._xtc)
        self.molsys.movie()
        self.options()
        self.command_line(f'coordset #{self.mdlnums[0]}')
        
        return self
    
    @property
    def thread(self):
        return self._thread
    
    @thread.setter
    def thread(self):
        if self._thread is not None:self.thread.stop()
        
    
    def bond_interactive(self,rho_index=None,frac:float=0.75,
                 nframes:int=150,framerate:int=15,scaling:float=1,pc_scaling:float=1):
        self.molsys.movie()
            
        
        if rho_index is None:rho_index=np.arange(self.pca.Data.PCARef.ne)
        rho_index=np.atleast_1d(rho_index)
        out={'mdl_num':self.molsys.movie.mdlnums[1],
             'ids':[rs.ids for rs in self.select.repr_sel],
             'xtc_type':'xtc_bond',
             'rho_index':rho_index,
             'file':os.path.join(self.molsys.directory,'xtc_temp')}
        
        CMXid=self.molsys.movie.CMXid
        CMX=self.molsys.movie.CMX
        CMX.add_event(CMXid,'PCAtraj',out)
        
        
        if self.thread is not None:self.thread.stop()
        
        mv=self.molsys.movie
        cmds=[f'~ribbon #{mv.mdlnums[0]}',f'show #{mv.mdlnums[0]}',
              f'color #{mv.mdlnums[0]} tan',f'style #{mv.mdlnums[0]} ball']
        self.molsys.movie.command_line(cmds)
        
        self._thread=BondWait(self,frac=frac,nframes=nframes,framerate=framerate,
                                pc_scaling=pc_scaling)
        self._thread.start()
        
        self.options.Detectors(rho_index=rho_index,scaling=scaling)
        
        
        return self
        
from threading import Thread
from time import sleep
class BondListener(Thread):
    def __init__(self,movie,**kwargs):
        super().__init__()
        self.PCAmovie=movie
        self.MSmovie=movie.molsys.movie
        self.conn=movie.molsys.movie.CMX.conn[movie.molsys.movie.CMXid]
        self.kwargs=kwargs
        self.cont=True
    def run(self):
        out=self.conn.recv()
        if out[0]=='xtc_request':
            if out[3]==-1:
                self.PCAmovie.xtc_rho(rho_index=out[2],**self.kwargs).play_xtc()
            else:
                self.PCAmovie.xtc_bond(index=out[3],rho_index=out[2],**self.kwargs).play_xtc()
            self.MSmovie.command_line(f'coordset #{self.MSmovie.mdlnums[0]} 1,')
        if self.cont:self.run()
    def stop(self):
        self.cont=False
        
class BondWait(Thread):
    def __init__(self,movie,**kwargs):
        super().__init__()
        self.PCAmovie=movie
        self.MSmovie=movie.molsys.movie
        self.conn=movie.molsys.movie.CMX.conn[movie.molsys.movie.CMXid]
        self.kwargs=kwargs
        self.cont=True
        self.file=os.path.join(self.PCAmovie.molsys.directory,'xtc_temp')
    def run(self):
        while self.cont:
            if os.path.exists(self.file):
                with open(self.file,'r') as f:
                    out=f.readline().strip().split()
                    if out[0]=='xtc_request':
                        if out[3]==-1:
                            self.PCAmovie.xtc_rho(rho_index=int(out[2]),**self.kwargs).play_xtc()
                        else:
                            self.PCAmovie.xtc_bond(index=int(out[3]),rho_index=int(out[2]),
                                                   **self.kwargs).play_xtc()
                        self.MSmovie.command_line(f'coordset #{self.MSmovie.mdlnums[0]} 1,')
                os.remove(self.file)
                if 'Detectors' in self.PCAmovie.options.dict:
                    self.PCAmovie.options.dict.pop('Detectors')
            sleep(.2)
    def stop(self):
        self.cont=False
            
        
    
    
    
class Options():
    def __init__(self,pca_movie):
        self.dict={}
        self.pca_movie=pca_movie
        self._commands=[]
    
    @property
    def commands(self):
        return self._commands
    @commands.setter
    def commands(self,value):
        if isinstance(value,str):
            value=[value]
        assert isinstance(value,list),'Commands must be a list'
        self._commands=value
    
    def __call__(self):
        for f in self.dict.values():f()
        self.run_commands()
    
    def clear(self):
        for k in self.dict:
            self.remove_event(k)
        self.dict={}
        return self
    
    @property
    def CMX(self):
        return self.pca_movie.molsys.movie.CMX
    
    @property
    def CMXid(self):
        return self.pca_movie.molsys.movie.CMXid
    
    def add_event(self,name,*args):
        self.CMX.add_event(self.CMXid,name,*args)
        
    def remove_event(self,name):
        self.CMX.remove_event(self.CMXid,name)
        
    def run_commands(self):
        for cmd in self.commands:
            self.CMX.command_line(self.CMXid,cmd.format(self.pca_movie.molsys.movie.mdlnums[0]))
    
    def TimescaleIndicator(self,tau=None,remove:bool=False):
        """
        Add a timescale indicator to the trajectory

        Parameters
        ----------
        tau : np.array
            Timescales for each frame of the trajectory.
        remove : bool
            Remove the timescale indicator

        Returns
        -------
        None.

        """
        
        if remove:
            if 'TimescaleIndicator' in self.dict:self.dict.pop('TimescaleIndicator')
            self.remove_event('TimescaleIndicator')
            return
        if tau is None:tau=10**self.pca_movie.setup['tscale_swp']*1e9
        if tau is not None:
            self.dict['TimescaleIndicator']=\
                lambda tau=tau:self.add_event('TimescaleIndicator', tau,
                                              self.pca_movie.molsys.movie.mdlnums[1])
        return self
    
    def Detectors(self,index=None,rho_index:int=None,scaling:float=10,remove:bool=False):
        """
        

        Returns
        -------
        None.

        """
        
        R=self.pca_movie.BondRef.R
        ids=[rs.ids for rs in self.pca_movie.select.repr_sel]
        
        out=dict(R=R*scaling,rho_index=rho_index,ids=ids)
        self.dict['Detectors']=lambda out=out:self.add_event('Detectors',out)
        return self
    
    
    def DetFader(self,tau=None,index=None,rho_index=None,remove:bool=False):
        """
        Add a Detector Fader to the trajectory

        Parameters
        ----------
        tau : TYPE, optional
            DESCRIPTION. The default is None.
        index : TYPE, optional
            DESCRIPTION. The default is None.
        rho_index : TYPE, optional
            DESCRIPTION. The default is None.
        remove : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        self

        """
        
        print('check1')
        if remove:
            if 'DetFader' in self.dict:self.dict.pop('DetFader')
            self.remove_event('DetFader')
            return
        
        if tau is None:tau=10**self.pca_movie.setup['tscale_swp']
        
        data=self.pca_movie.BondRef
        rhoz=data.sens.rhoz
        
        if index is None:index=np.ones(data.R.shape[0],dtype=bool)
        x=data.R[index]
        
            
        if rho_index is None:
            rho_index=np.ones(data.R.shape[1],dtype=bool)
            if rhoz[-1][-1]>0.9:rho_index[-1]=False

        
        x=x[:,rho_index]
        rhoz=rhoz[rho_index]
        
        tc=data.sens.tc
        ids=[a.ids for a in self.pca_movie.select.repr_sel[index]]
        
            
        
        x/=x.max()
        
        
        def fun(x=x,ids=ids,tau=tau,rhoz=rhoz):
            mn0,mn=self.pca_movie.molsys.movie.mdlnums
            self.add_event('DetectorFader',mn,x,ids,tau*1e9,rhoz,tc,5)
            self.CMX.command_line(self.CMXid,f'color #{mn0} tan')
            self.CMX.command_line(self.CMXid,f'~ribbon #{mn0}')
            self.CMX.command_line(self.CMXid,f'show #{mn0}')
            self.CMX.command_line(self.CMXid,f'style #{mn0} ball')
            
        self.dict['DetFader']=fun
            
        return self
        
        