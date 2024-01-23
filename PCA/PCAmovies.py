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
        self._data=None
        self._direct_data=None
        self._zrange=None
        self._timescale=None
        self._xtc=None
        self._select=None
        self._options=Options(self)
    
    
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
    def data(self):
        """
        Processed PCA detector analysis. Can be slow at the first call

        Returns
        -------
        data
            data object containing a detector analysis of the PCA.

        """
        if self._data is None:
            d=self.pca.PCA2data()
            d.detect.r_no_opt(self.n)
            noo=d.fit()
            noo.detect.r_auto(self.n)
            self._data=noo.fit()
            
        return self._data

    
    @data.setter
    def data(self,data):
        self._data=data
        self._zrange=None
    
    @property
    def direct_data(self):
        """
        Data object containing direct analysis of the bond motion (no PCA). 

        Returns
        -------
        None.

        """
        if self._direct_data is None:
            d=self.pca.direct2data()
            d.detect.r_no_opt(self.n)
            noo=d.fit()
            noo.detect.r_auto(self.n)
            self._direct_data=noo.fit()
        return self._direct_data
    
    @direct_data.setter
    def direct_data(self,data):
        self._direct_data=data
    
    @property
    def sens(self):
        """
        Reference to the data's sensitivity object

        Returns
        -------
        sens

        """
        return self.data.sens
    
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

    def weight(self,timescale):
        """
        Returns a weighting of the principal components for a given timescale
        or timescales.
        
        This is determined from the stored sensitity object and the relative
        amplitudes of the detectors at the given timescale. 

        Parameters
        ----------
        timescale : float,array
            timescale or timescales given on a log-scale. The default is None.

        Returns
        -------
        np.array

        """
        if timescale is None:
            timescale=timescale_swp(self.zrange)
        timescale=np.atleast_1d(timescale)
        
        i=np.array([np.argmin(np.abs(ts-self.sens.z)) for ts in timescale],dtype=int)
        wt0=self.sens.rhoz[:,i]
        # wt0/=wt0.sum(0)
        
        return self.data.R@wt0
    
    # def xtc_log_swp2(self,filename:str='temp.xtc',zrange=None,nframes:int=300,framerate:int=15):
    #     """
    #     Create an xtc file for a log sweep of the timescales. Setup is the output
    #     of setup_swp, which can be omitted if default values are desired.
        
    #     filename is the output filename of the xtc. If filetype (eg. .xtc) is
    #     omitted, then .xtc will be appended. It will be put into the molsys
    #     temporary folder by default

    #     Parameters
    #     ----------
    #     filename : str, optional
    #         DESCRIPTION. The default is 'temp.xtc'.
    #     setup : TYPE, optional
    #         DESCRIPTION. The default is None.

    #     Returns
    #     -------
    #     self

    #     """
        
    #     setup=self.setup_swp(zrange=zrange,nframes=nframes,framerate=framerate)
        
    #     wt=self.weight(setup['tscale_swp'])
    #     i=setup['index']
        
    #     ag=copy(self.pca.atoms)
        
    #     filename=os.path.join(self.directory,filename)
        
    #     pos0=pos=copy(self.pca.pos[0])
    #     DelPC=np.concatenate([np.zeros([self.pca.PCamp.shape[0],1]),
    #                           np.diff(self.pca.PCamp[:,i],axis=-1)],axis=-1)*wt
        
        
        
    #     with Writer(filename, n_atoms=len(ag)) as w:
    #         for k,_ in enumerate(self.pca.traj[i]):
    #             # ag.positions=self.pca.mean+((wt[:,k]*self.pca.PCamp[:,i[k]])*self.pca.PCxyz).sum(-1).T
    #             pos+=(DelPC[:,k]*self.pca.PCxyz).sum(-1).T
    #             ag.positions=pos
    #             w.write(ag)
                
    #     self._xtc=filename
        
    #     return self
    
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
        
        wt=self.weight(setup['tscale_swp'])
        # wt=(wt.T/wt.mean(-1)).T
        i=setup['index']
        
        ag=copy(self.pca.atoms)
        
        filename=os.path.join(self.directory,filename)
        
        pos0=pos=copy(self.pca.pos[0])
        DelPC=np.concatenate([np.zeros([self.pca.PCamp.shape[0],1]),
                              np.diff(self.pca.PCamp[:,i],axis=-1)],axis=-1)
        
        PCcorr=self.pca.PCamp[:,i[-1]]-self.pca.PCamp[:,0]-DelPC.sum(-1)
        
        
        PCamp=np.cumsum(DelPC,axis=-1)+np.atleast_2d(PCcorr).T@np.atleast_2d(i)/i[-1]
        
        pos=self.pca.pos[0].T+np.array([(self.pca.PCxyz*a).sum(-1) for a in PCamp.T])
        
        # pos=self.pca.pos[0]+(np.cumsum(DelPC,axis=-1)*self.pca.PCxyz).sum(-1).T+\
        #     +(np.atleast_2d(PCcorr).T@np.atleast_2d(i)/i[-1])
        
        # pos=self.pca.pos[0]+np.array([])
        
        
        
        
        with Writer(filename, n_atoms=len(ag)) as w:
            for pos0 in pos:
                # ag.positions=self.pca.mean+((wt[:,k]*self.pca.PCamp[:,i[k]])*self.pca.PCxyz).sum(-1).T
                ag.positions=pos0.T
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
        
        return self
    
class Options():
    def __init__(self,pca_movie):
        self.dict={}
        self.pca_movie=pca_movie
        
    def __call__(self):
        for f in self.dict.values():f()
    
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
        if tau is None:tau=10**self.pca_movie.setup['tscale_swp']*1e9
        if remove:
            if __name__ in self.dict:self.dict.pop(__name__)
            return
        if tau is not None:
            self.dict['TimescaleIndicator']=lambda tau=tau:self.add_event('TimescaleIndicator', tau)
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
        
        if tau is None:tau=10**self.pca_movie.setup['tscale_swp']
        
        if remove:
            if __name__ in self.dict:self.dict.pop(__name__)
            return
        
        data=self.pca_movie.direct_data
        rhoz=data.sens.rhoz
        
        if index is None:index=np.ones(data.R.shape[0],dtype=bool)
        x=data.R[index]
        
            
        if rho_index is None:
            rho_index=np.ones(data.R.shape[1],dtype=bool)
            if rhoz[-1][-1]>0.9:rho_index[-1]=False
            print(rho_index)
        
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
        
        