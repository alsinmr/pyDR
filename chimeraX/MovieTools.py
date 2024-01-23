#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:33:06 2024

@author: albertsmith
"""

import numpy as np

def timescale_swp(zrange=[-10,-6],nframes:int=150):
    """
    Returns a sweep of the timescale defined by zrange. zrange is given
    by two values on a log scale, e.g. default [-10 -6] runs from 100 ps to 1
    μs. 
    
    timescale = np.logspace(zrange[0],zrange[1],nframes)
    
    Parameters
    ----------
    zrange : TYPE, optional
        DESCRIPTION. The default is [-10,-6].
    nframes : TYPE
        DESCRIPTION.
    framerate : TYPE
        DESCRIPTION.

    Returns
    -------
    np.array
        1D array of the timescales on logscale

    """
    return np.linspace(zrange[0],zrange[1],nframes)

def log_axis(traj,zrange=[-10,-6],nframes:int=150,framerate:int=15):
    """
    Returns a time axis to cover timescales defined by zrange. zrange is given
    by two values on a log scale, e.g. default [-10 -6] runs from 100 ps to 1
    μs. This means that at the input framerate (default 15 ns), 1 second of
    movie will cover 100 ps of trajectory at the beginning of the axis and
    1 second of movie will cover 1 μs of trajectory at the end of the axis.
    
    timestep beginning: 100 ps/15 fps = 6.67 ps/frame
    timestep end:       1 μs/15 fps = 66.67 ns / frame
    
    target timestep:
    np.logspace(zrange[0]-np.log10(framerate),zrange[1]-np.log10(framerate),nframes-1)
    
    Note that if the initial timestep is too short for the stepsize of the 
    trajectory, then the initial timestep will just be the trajectory's 
    timestep, although this timestep will be retained up until it is shorter
    than the target timestep.
    
    If the trajectory is not long enough to support the requested range of
    timescales, then timesteps at the end of the trajectory will be shortened
    until the time axis fits.
    
    traj is usually the Trajectory object contained in MolSys and MolSelect 
    objects. However, it may also be a tuple (dt,nt) with the timestep in
    picoseconds and number of points in the trajectory.
    
    The returned time axis is rounded to match dt in the time axis. An index
    of timepoints for the original axis is also returned

    Parameters
    ----------
    traj : Trajectory object / tuple
        Either a trajectory object, or tuple with dt (trajectory timestep) and
        nt (number of frames in trajectory)
    zrange : TYPE, optional
        DESCRIPTION. The default is [-9,-5].
    nframes : TYPE, optional
        DESCRIPTION. The default is 150.
    framerate : TYPE, optional
        DESCRIPTION. The default is 15.

    Returns
    -------
    tuple
    t: numpy float array (nanoseconds), index: numpy int array

    """
    
    dt,nt=traj if len(traj)==2 else traj.dt,len(traj)
    dt*=1e-12
    total=dt*(nt-1)
    
    Dt=np.logspace(zrange[0]-np.log10(framerate),zrange[1]-np.log10(framerate),nframes-1)
    Dt[Dt<dt]=dt
    
    t=np.concatenate([[0],np.cumsum(Dt)])
    
    tt=t[:-1]+Dt*np.arange(len(t)-1,0,-1)
    i=tt>total
    
    Dt[tt>total]=Dt[np.argmax(i)-1]
    
    t=np.concatenate([[0],np.cumsum(Dt)])
    i=np.round(t/dt).astype(int)
    t=i*dt*1e9
    
    return t,i

def lin_axis(traj,z:float=-9,nframes:int=75,framerate:int=15):
    """
    Returns a time axis around the timescale defined by z, e.g. with the default
    -9, one second of movie will cover 1 nanosecond at the input framerate.
    
    If the input z value cannot be matched due to the timestep or trajectory
    length, then this will be adjusted

    Parameters
    ----------
    traj : TYPE
        DESCRIPTION.
    z : TYPE, optional
        DESCRIPTION. The default is -9.
    nframes : TYPE, optional
        DESCRIPTION. The default is 75.
    framerate : TYPE, optional
        DESCRIPTION. The default is 15.

    Returns
    -------
    tuple
    t: numpy float array (nanoseconds), index: numpy int array

    """
    dt,nt=traj if len(traj)==2 else traj.dt,len(traj)
    dt*=1e-12
    total=dt*(nt-1)
    
    Dt=10**z
    
    if Dt<dt:
        print(f'Requested timescale ({10**z*1e9:.1f} ns) is too short for a trajectory with dt={dt*1e9:.1f} ns')
        Dt=dt
    elif Dt*nframes>total:
        print(f'Requested timescale ({10**z*1e9:.1f} ns) is too long for a trajectory with length={total*1e9:.1f} ns')
        Dt=total/nframes
    
    t=np.arange(nframes)*Dt
    i=np.round(t/dt).astype(int)
    t=i*dt*1e9
    
    return t,i
    