#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of pyDIFRATE

pyDIFRATE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyDIFRATE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyDIFRATE.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de


Created on Thu Feb  6 10:45:12 2020

@author: albertsmith
"""

"""
This module is meant for containing special-purpose frames. Usually, these will
only work on a specific type of system, and may be more complex types of functions
(for example, peptide_plane is still a standard frame, although it only works
for proteins, because it is relatively simple)
    
    1) The first argument must be "molecule", where this refers to the molecule
    object of pyDIFRATE
    2) The output of this function must be another function.
    3) The returned function should not require any input arguments. It should 
    only depend on the current time point in the MD trajectory (therefore, 
    calling this function will return different results as one advances through
    the trajectory).
    4) The output of the sub-function should be one or two vectors (if the
    frame is defined by just a bond direction, for example, then one vector. If
    it is defined by some 3D object, say the peptide plane, then two vectors 
    should be returned)
    5) Each vector returned should be a numpy array, with dimensions 3xN. The
    rows corresponds to directions x,y,z. The vectors do not need to be normalized
    
    6) Be careful of factors like periodic boundary conditions, etc. In case of
    user frames and in the built-in definitions (frames.py) having the same name,
    user frames will be given priority.
    7) The outer function must have at least one required argument aside from
    molecule. By default, calling molecule.new_frame(Type) just returns a list
    of input arguments.
    
    
    Ex.
        def user_frame(molecule,arguments...):
            some_setup
            sel1,sel2,...=molecule_selections (use select_tools for convenience)
            ...
            
            def sub()
                ...
                v1,v2=some_calculations
                ...
                box=molecule.box (periodic boundary conditions)
                v1=vft.pbc_corr(v1,box)
                v2=vft.pbc_corr(v2,box)
                
                return v1,v2
            return sub
            
"""



import numpy as np
from pyDR.MDtools import vft
from pyDR.Selection import select_tools as selt
from copy import copy

def hop_setup(uni,sel1,sel2,sel3,sel4,ntest=1000):
    """
    Function that determines where the energy minima for a set of bonds can be
    found. Use for chi_hop and hop_3site.
    """
    v12,v23,v34=list(),list(),list()
    box=uni.dimensions
    traj=uni.trajectory
    step=np.floor(traj.n_frames/ntest).astype(int)
    
    for _ in traj[::step]:
        v12.append(vft.pbc_corr((sel1.positions-sel2.positions).T,box[:3]))
        v23.append(vft.pbc_corr((sel2.positions-sel3.positions).T,box[:3]))
        v34.append(vft.pbc_corr((sel3.positions-sel4.positions).T,box[:3]))
    
    traj[0] #Sometimes, leaving the trajectory at the end can create other errors...
    
    v12,v23,v34=[np.moveaxis(np.array(v),0,-1) for v in [v12,v23,v34]]

    v12a=vft.applyFrame(v12,nuZ_F=v23,nuXZ_F=v34) #Rotate so that 23 is on z-axis, 34 in XY-plane
    
    v0z=vft.norm(np.array([np.sqrt(v12a[0]**2+v12a[1]**2).mean(axis=-1),\
                           np.zeros(v12a.shape[1]),v12a[2].mean(axis=-1)]))    #Mean projection onto xz
    
    v12a[2]=0               #Project v12 onto xy-plane
    v12a=vft.norm(v12a)
    i=np.logical_and(v12a[0]<.5,v12a[1]>0)          #For bonds not between -60 and 60 degrees
    v12a[:,i]=vft.Rz(v12a[:,i],-.5,-np.sqrt(3)/2)    #we rotate +/- 120 degrees to align them all
    i=np.logical_and(v12a[0]<.5,v12a[1]<=0)
    v12a[:,i]=vft.Rz(v12a[:,i],-.5,np.sqrt(3)/2)
    
    v0xy=vft.norm(v12a.mean(-1))    #This is the average direction of v12a (in xy-plane)
    theta=np.arctan2(v0xy[1],v0xy[0])
    """The direction of the frame follows sel2-sel3, sel3-sel4, but sel1-sel2 
    is forced to align with a vector in vr"""
    vr=np.array([vft.Rz(v0z,k+theta) for k in [0,2*np.pi/3,4*np.pi/3]])  #Reference vectors (separated by 120 degrees)
    "axis=1 of vr is x,y,z"    
    
    return vr

def chi_hop(molecule,n_bonds=1,Nuc=None,resids=None,segids=None,filter_str=None,ntest=1000,sigma=0):
    """
    Determines contributions to motion due to 120 degree hops across three sites
    for some bond within a side chain. Motion of the frame will be the three site
    hoping plus any outer motion (could be removed with additional frames), and
    motion within the frame will be all rotation around the bond excluding 
    hopping.
    
    One provides the same arguments as side_chain_chi, where we specify the
    nucleus of interest (ch3,ivl,ivla,ivlr,ivll, etc.), plus any other desired
    filters. We also provide n_bonds, which will determine how many bonds away
    from the methyl group (only methyl currently implemented) we want to observe
    the motion (usually 1 or 2). 
    """

    "First we get the selections, and simultaneously determine the frame_index"    
    if Nuc is None:
        Nuc='ch3'
    selC,_=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)  
    selC=selC[::3]    #Above line returns 3 copies of each carbon. Just take 1 copy
    frame_index=list()
    sel1,sel2,sel3,sel4=None,None,None,None
    k=0
    for s in selC:
        chain=selt.get_chain(s,s.residue.atoms)[2+n_bonds:6+n_bonds]
        if len(chain)==4:
            frame_index.extend([k,k,k])
            k+=1
            if sel1 is None:
                sel1,sel2,sel3,sel4=chain[0:1],chain[1:2],chain[2:3],chain[3:4]
            else:
                sel1=sel1+chain[0]
                sel2=sel2+chain[1]
                sel3=sel3+chain[2]
                sel4=sel4+chain[3]
        else:
            frame_index.extend([np.nan,np.nan,np.nan])
    frame_index=np.array(frame_index)
    
    "Next, we sample the trajectory to get an estimate of the energy minima of the hopping"
    #Note that we are assuming that minima are always separated by 120 degrees
    
    vr=hop_setup(molecule.uni,sel1,sel2,sel3,sel4,ntest)
    
    box=molecule.box
    if sigma!=0:
        def sub():
            return [vft.pbc_corr((s1.positions-s2.positions).T,box[:3]) \
                 for s1,s2 in zip([sel1,sel2,sel3],[sel2,sel3,sel4])]
        return sub,frame_index,{'PPfun':'AvgHop','vr':vr,'sigma':sigma}
    else:
            
        def sub():
            v12s,v23s,v34s=[vft.pbc_corr((s1.positions-s2.positions).T,box[:3]) \
                         for s1,s2 in zip([sel1,sel2,sel3],[sel2,sel3,sel4])]
            v12s=vft.norm(v12s)
            sc=vft.getFrame(v23s,v34s)
            v12s=vft.R(v12s,*vft.pass2act(*sc))    #Into frame defined by v23,v34
            # print((v12s*vr).sum(axis=1)[:,0])
            i=np.argmax((v12s*vr).sum(axis=1),axis=0)   #Index of best fit to reference vectors (product is cosine, which has max at nearest value)
            v12s=vr[i,:,np.arange(v12s.shape[1])] #Replace v12 with one of the three reference vectors
            return vft.R(v12s.T,*sc),v23s  #Rotate back into original frame        
            
        return sub,frame_index

def hops_3site(molecule,sel1=None,sel2=None,sel3=None,sel4=None,\
               Nuc=None,resids=None,segids=None,filter_str=None,ntest=1000,sigma=0):
    """
    Determines contributions to motion due to 120 degree hops across three sites. 
    Motion within this frame will be all motion not involving a hop itself. Motion
    of the frame will be three site hoping plus any outer motion (ideally removed
    with a methylCC frame)
    
    sel1 and sel2 determine the bond of the interaction (sel2 should be the 
    carbon). sel2 and sel3 determine the rotation axis, and sel3/sel4 keep the
    axis aligned.
    
    sel1-sel4 may all be automatically determined if instead providing some of
    the usual selection options (Nuc, resids, segids, filter_str)
    
    First step is to use sel2/sel3 as a z-axis and project the bond onto the x/y
    plane for a series of time points. We then rotate around z to find an 
    orientation that best explains the sel1/sel2 projection as a 3 site hop.
    
    Second step is to project sel1/sel2 vector onto the z-axis and determine the
    angle of the bond relative to the z-axis.
    
    Then, this frame will only return vectors that match this angle to the z-axis
    and are defined by the 3-site hop.
    
    Setup requires a sampling of the trajectory. We use 1000 points by default
    (ntest). This frame will take more time than most to set up because of this
    setup.
    
    hops_3site(molecule,sel1=None,sel2=None,sel3=None,sel4=None,ntest=1000)
    """
    
    
    if sel1:sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
    if sel2:sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)     
    if sel3:sel3=selt.sel_simple(molecule,sel3,resids,segids,filter_str)
    if sel4:sel4=selt.sel_simple(molecule,sel4,resids,segids,filter_str)
    
    if not(sel1) and not(sel2):sel2,sel1=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)

    if 'H' in sel2[0].name:sel1,sel2=sel2,sel1

    "Get all atoms in the residues included in the initial selection"
    uni=molecule.uni
    resids=np.unique(np.concatenate([sel1.resids,sel2.resids]))
    sel0=uni.residues[np.isin(uni.residues.resids,resids)].atoms
    
    if not(sel3):
        sel3=selt.find_bonded(sel2,sel0,exclude=sel1,n=1,sort='cchain',d=1.65)[0]
    if not(sel4):
        sel4=selt.find_bonded(sel3,sel0,exclude=sel2,n=1,sort='cchain',d=1.65)[0]
        
    vr=hop_setup(molecule.uni,sel1,sel2,sel3,sel4,ntest)
    
    box=molecule.box
    if sigma!=0:
        def sub():
            return [vft.pbc_corr((s1.positions-s2.positions).T,box) \
                         for s1,s2 in zip([sel1,sel2,sel3],[sel2,sel3,sel4])]
        return sub,None,{'PPfun':'AvgHop','vr':vr,'sigma':sigma}
    else:
        def sub():
            v12s,v23s,v34s=[vft.pbc_corr((s1.positions-s2.positions).T,box) \
                         for s1,s2 in zip([sel1,sel2,sel3],[sel2,sel3,sel4])]
            v12s=vft.norm(v12s)
            sc=vft.getFrame(v23s,v34s)
            v12s=vft.R(v12s,*vft.pass2act(*sc))    #Into frame defined by v23,v34
            i=np.argmax((v12s*vr).sum(axis=1),axis=0)   #Index of best fit to reference vectors (product is cosine, which has max at nearest value)
            v12s=vr[i,:,np.arange(v12s.shape[1])] #Replace v12 with one of the three reference vectors
            return vft.R(v12s.T,*sc),v23s  #Rotate back into original frame
        return sub
    

    
def membrane_grid(molecule,grid_pts,sigma=25,sel0=None,sel='type P',resids=None,segids=None,filter_str=None):
    """
    Calculates motion of the membrane normal, defined by a grid of points spread about
    the simulation. For each grid point, a normal vector is returned. The grid
    is spread uniformly around some initial selection (sel0 is a single atom!)
    in the xy dimensions (currently, if z is not approximately the membrane 
    normal, this function will fail).
    
    The membrane normal is defined by a set of atoms (determined with some 
    combination of the arguments sel, resids, segids, filter_str, with sel_simple)
    
    At each grid point, atoms in the selection will be fit to a plane. However,
    the positions will be weighted depending on how far they are away from that
    grid point in the xy dimensions. Weighting is performed with a normal 
    distribution. sigma, by default, has a width approximately equal to the 
    grid spacing (if x and y box lengths are different, we have to round off the
    spacing)
    
    The number of points is given by grid_pts. These points will be distributed
    automatically in the xy dimensions, to have approximately the same spacing
    in both dimensions. grid_pts will be changed to be the product of the exact
    number of points used (we will always distribute an odd number of points
    in each dimension, so the reference point is in the center of the grid)
    
    if sel0, defining the reference atom, is omitted, then the center of the
    box will be used. Otherwise, the grid will move around with the reference
    atom
    
    membrane_grid(molecule,grid_pts,sigma,sel0,sel,resids,segids,filter_str)
      
    """

    uni=molecule.uni
    
    X,Y,Z=uni.dimensions[:3]
    nX,nY=1+2*np.round((np.sqrt(grid_pts)-1)/2*np.array([X/Y,Y/X]))
    dX,dY=X/nX,Y/nY
    
    print('{0:.0f} pts in X, {1:.0f} pts in Y, for {2:.0f} total points'.format(nX,nY,nX*nY))
    print('Spacing is {0:.2f} A in X, {1:.2f} A in Y'.format(dX,dY))
    print('Center of grid is found at index {0:.0f}'.format(nX*(nY-1)/2+(nX-1)/2))
    print('sigma = {0:.2f} A'.format(sigma))
    
    
    if sel0 is not None:
        sel0=selt.sel_simple(molecule,sel0)  #Make sure this is an atom group
        if hasattr(sel0,'n_atoms'):
            if sel0.n_atoms!=1:
                print('Only one atom should be selected as the membrane grid reference point')
                print('Setup failed')
                return
            else:
                sel0=sel0[0]    #Make sure we have an atom, not an atom group
        
        tophalf=sel0.position[2]>Z/2    #Which side of the membrane is this?
    else:
        tophalf=True
        
    "Atoms defining the membrance surface"    
    sel=selt.sel_simple(molecule,sel,resids,segids,filter_str)
    
    "Filter for only atoms on the same side of the membrane"
    sel=sel[sel.positions[:,2]>Z/2] if tophalf else sel[sel.positions[:,2]<Z/2]
    
    def grid():
        "Subfunction, calculates the grid"
        X0,Y0=(X/2,Y/2) if sel0 is None else sel0.position[:2]  #Grid at center, or at position of sel0
        Xout=np.transpose([X0+(np.arange(nX)-(nX-1)/2)*dX]).repeat(nY,axis=1).reshape(int(nX*nY))
        Yout=np.array([Y0+(np.arange(nY)-(nY-1)/2)*dY]).repeat(nY,axis=0).reshape(int(nX*nY))
        return Xout,Yout
    
    def sub():
        "Calculate planes for each element in grid"
        X,Y=grid()
        v=list()
        box=molecule.box
        for x,y in zip(X,Y):  
            v0=vft.pbc_corr(np.transpose(sel.positions-[x,y,0]),box)
            d2=v0[0]**2+v0[1]**2
            i=d2>3*sigma
            weight=np.exp(-d2[i]/(2*sigma**2))
            
            v.append(vft.RMSplane(v0[:,i],np.sqrt(weight)))
        v=np.transpose(v)
        return v/np.sign(v[2])
    
    return sub
    
    

def hop(molecule,angle=np.arccos(np.sqrt(1/3)),sel1=None,sel2=None,sel3='auto',Nuc=None,resids=None,segids=None,filter_str=None,step=1):
    """
    Frame to separate hops from small motions.
    Evaluates the orientation of the current frame relative to the previous 
    frame (or several frames earlier, set step for more than one frame back)
    If the change in orientation is less more than the given angle (default 54.74),
    then this motion is registered as a hop, otherwise, the frame remains fixed.
    
    Note that the size of the hop is not fixed, but instead is set so that motion
    within the hop frame is 0 at that time step.
    
    Warning: as implemented, this frame cannot be used with sparse trajectory 
    sampling!
    
    Thus far, I haven't gotten this frame to perform well - A.S., Jan. 2022
    """
    if Nuc is not None:
        sel2,sel1=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel2,sel1=[selt.sel_simple(molecule,s,resids,segids,filter_str) for s in [sel1,sel2]]
        
    if isinstance(sel3,str) and sel3=='auto':
        uni=sel1.universe
        resids=np.unique(sel2.resids)
        sel0=uni.residues[np.isin(uni.residues.resids,resids)].atoms
        sel3=selt.find_bonded(sel2,sel0,exclude=sel1,n=1,sort='cchain')[0]
    elif sel3 is not None:
        sel3=selt.sel_simple(molecule,sel3,resids,segids,filter_str)
    
    molecule.traj[0]
    box=molecule.box
    vZ0=vft.pbc_corr((sel1.positions-sel2.positions).T,box)
    priorZ=[vZ0 for _ in range(step)] #List to keep track of previous frames
   
    vXZ0=vft.pbc_corr((sel3.positions-sel2.positions).T,box) if sel3 else None
    priorXZ=[vXZ0 for _ in range(step)]
    
    threshold=np.cos(angle)

    def sub():                
        box=molecule.box
        vZ=vft.norm(vft.pbc_corr((sel1.positions-sel2.positions).T,box))
        vXZ=vft.pbc_corr((sel3.positions-sel2.positions).T,box) if sel3 else None
              
        "Did a hop occur?"
        hop=(priorZ[0]*vZ).sum(0)<threshold    

        priorZ.append(vZ)
        priorXZ.append(vXZ)      
        
        if np.any(hop):
            scp=vft.getFrame(priorZ[-2][:,hop],priorXZ[-2][:,hop] if sel3 else None,return_angles=True)    #Find the frame of the previous orientation
            vZt,vXZt,vZ,vXZ=[vft.R(v[:,hop],*vft.pass2act(*scp)) for v in [vZ0,vXZ0,vZ,vXZ]] #Apply that frame to the current orientation
            sc=vft.getFrame(vZ,vXZ,return_angles=True)
            "Rotate by motion acting on vZ,vXZ, rotate back into lab frame"
            vZt,vXZt=[vft.R(vft.R(v,*sc),*scp) for v in [vZt,vXZt]]
            vZ0[:,hop]=vZt 
            
        priorZ.pop(0)       #Pop out the earliest entry into priors
        priorXZ.pop(0)
        
        
        if sel3 is None:
            return vZ0.copy()
        if np.any(hop):
            vXZ0[:,hop]=vXZt
        return vZ0.copy(),vXZ0.copy()
    return sub


#%% Lipid frames
#The following frames are specifically intended for separating dynamics in lipid
# These are introduced now 
    
def TetraHop(molecule,sel1=None,sel2=None,resids=None,segids=None,filter_str=None):
    return TH(molecule,sel1=sel1,sel2=sel2,resids=resids,segids=segids,filter_str=filter_str)

class TH():
    def __init__(self,molecule,sel1=None,sel2=None,resids=None,segids=None,filter_str=None):
        """
        Attempts to isolate tetrahedral hops by comparing the orientation of the
        bond in the current frame to orientations in the previous frame. If the 
        bond remains closest to its orientation in the prevous frame, then this
        frame remains fixed. If it better matches a different orientation, then
        the orientation of this frame will rotate such that the bond remains
        fixed in this frame.
        
        Warning: This frame cannot be used with sparse trajectory sampling!
    
        Parameters
        ----------
        molecule : TYPE
            DESCRIPTION.
        sel1 : TYPE, optional
            DESCRIPTION. The default is None.
        sel2 : TYPE, optional
            DESCRIPTION. The default is None.
        resids : TYPE, optional
            DESCRIPTION. The default is None.
        segids : TYPE, optional
            DESCRIPTION. The default is None.
        filter_str : TYPE, optional
            DESCRIPTION. The default is None.
    
        Returns
        -------
        None.
    
        """
        self.molecule=molecule
        self.sel1,self.sel2=[selt.sel_simple(molecule,s,resids,segids,filter_str) for s in [sel1,sel2]]
        
        self.sel0=selt.find_bonded(self.sel2,self.sel2.residues.atoms,exclude=self.sel1,n=3)
        
        self.reset()
        
    def reset(self):
        self.vm1=None
        self.nuZ=np.zeros([3,len(self.sel1)])
        self.nuZ[2]=1
        self.nuXZ=np.zeros([3,len(self.sel1)])
        self.nuXZ[0]=1
        
        self.frame=-1
        self.traj[0]
        self.record_state()
        return self
        
    @property
    def traj(self):
        return self.molecule.traj
        
        
    @property
    def vZ(self):
        """
        Current Z-direction of the bonds

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return vft.norm(vft.pbc_corr((self.sel1.positions-self.sel2.positions).T,self.molecule.box))
    
    @property
    def vXZ(self):
        """
        Current XZ-directions of the bonds

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return vft.norm(vft.pbc_corr((self.sel0[0].positions-self.sel2.positions).T,self.molecule.box))
    
    @property
    def v(self):
        """
        List of directions of tetrahedral bonds

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        out=[self.vZ]
        for sel0 in self.sel0:
            out.append(vft.norm(vft.pbc_corr((sel0.positions-self.sel2.positions).T,self.molecule.box)))
        return out
    
    #%% Here, we take care of the frame definition
    
    
    def record_state(self):
        if self.frame==self.traj.frame:return
        
        self.frame=self.traj.frame
        self.vm1=self.v
        self.vZm1=self.vZ
        self.vXZm1=self.vXZ
        return self
        
    @property
    def overlap(self):
        """
        Overlap of the current Z direction of the bond with the bonds in the
        previous frame. If biggest hop is not with the first bond, then a 
        hop has occured

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if self.vm1 is None:
            self.vm1=self.v
        vZ=self.vZ
        return np.array([(vm1*vZ).sum(0) for vm1 in self.vm1])
        
        
    
    @property
    def hop(self):
        # if self.vm1 is None:
        #     return np.zeros(len(self.sel1),dtype=bool)
        # else:
        return np.logical_not(np.all(self.overlap[0]>=self.overlap[1:],axis=0))
        
    def __call__(self):
        if self.traj.frame==0:self.reset()
        hop=self.hop  #Did a hop occur?
        

        if np.any(hop):
            nuZ,nuXZ=vft.applyFrame(self.nuZ[:,hop],self.nuXZ[:,hop],
                                    nuZ_F=self.vZm1[:,hop],nuXZ_F=self.vXZm1[:,hop])
            sc=vft.getFrame(self.vZ[:,hop],self.vXZ[:,hop])
            
            self.nuZ[:,hop],self.nuXZ[:,hop]=vft.R(nuZ,*sc),vft.R(nuXZ,*sc)
        
            
        self.record_state()
        
        return copy(self.nuZ),copy(self.nuXZ)
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    