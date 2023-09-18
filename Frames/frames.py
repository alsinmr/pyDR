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


Created on Thu Feb  6 10:43:33 2020

@author: albertsmith
"""

import numpy as np
from pyDR.MDtools import vft
from pyDR.Selection import select_tools as selt

"We import frames from other files as well"
from .special_frames import *    # analysis:ignore
from .user_frames import *       # analysis:ignore
#%% Frames
"""
Here, we define various functions that define the frames of different motions
in an MD trajectory. Each function should return another function that will
produce one or two vectors defining the frame (without arguments). Those vectors
should have X,Y,Z as the first dimension (for example, such that we can apply
X,Y,Z=v). Note this is the transpose of the outputs of MDanalysis positions
"""    

def peptide_plane(molecule,resids=None,segids=None,filter_str=None,full=True,sigma=0):
    """
    Aligns the peptide plane motion. Two options exist, full=True performs an
    RMS alignment of the N,H,CA of the given residue and C',O,CA of the previous
    residue. 
    full=False uses only the positions of the N of the given residue and C',O
    of the previous.
    
    The former is notably slower, but also performs better when separating
    librational motion
    """
    "Peptide plane motion, defined by C,N,O positions"
    if full:
        "Get selections" 
        selCA,selH,selN,selCm1,selOm1,selCAm1=selt.peptide_plane(molecule,resids,segids,filter_str)
        
        "Get universe, reset time"
        molecule.traj[0]
        
        "Define function to calculate the vectors defining the plane"
        def vfun():
            v=list()
            for CA,H,N,Cm1,Om1,CAm1 in zip(selCA,selH,selN,selCm1,selOm1,selCAm1):
                v0=np.array([CA.position-N.position,
                            H.position-N.position,
                            N.position-Cm1.position,
                            Cm1.position-Om1.position,
                            Cm1.position-CAm1.position])
                box=molecule.box
                v.append(vft.pbc_corr(v0.T,box))
            return v
        
        "Get the reference vectors (at t=0)"
        vref=vfun()        
        
        def sub():
            R=list()
            vecs=vfun()
            R=[vft.RMSalign(vr,v) for v,vr in zip(vecs,vref)]
            return vft.R2vec(R)
        return sub,None,{'PPfun':'AvgGauss','sigma':sigma}
    else:
        "Peptide plane motion, defined by C,N,O positions"
        selN,selC,selO=selt.peptide_plane(molecule,resids,segids,filter_str,full)
        def sub():
            box=molecule.box
            v1=selO.positions-selC.positions
            v2=selN.positions-selC.positions
            v1=vft.pbc_corr(v1.T,box)
            v2=vft.pbc_corr(v2.T,box)
        
            return v1,v2
        return sub,None,{'PPfun':'AvgGauss','sigma':sigma}
    
def bond(molecule,sel1=None,sel2=None,sel3=None,Nuc=None,resids=None,segids=None,filter_str=None):
    """Bond defines the frame. 
    sel1/sel2   :   Defines the z-axis of the frame (the bond itself). Follows 
                    the argument rules of sel_simple (sel2 should usually be
                    the heteroatom) 
    Nuc         :   Automatically sets sel1 and sel2 for a given nucleus definition
    sel3        :   sel2 and sel3 will define the xz-plane of the bond frame. 
                    This is optional: however, if this frame is the PAS of the
                    bond responsible for relaxation, then frames may not 
                    function correctly if this is not provided. By default, sel3
                    is set to None and is omitted. However, if called from within
                    molecule.tensor_frame, then default is changed to sel3='auto'
    resids, segids, filter_str apply additional filters to sel1, sel2, and sel3
    if defined.
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
    
    uni=molecule.uni
    
    if sel3 is None:
        def sub():
            box=molecule.box
            v=sel1.positions-sel2.positions
            v=vft.pbc_corr(v.T,box)
            return v
    else:
        def sub():
            box=molecule.box
            vZ=sel1.positions-sel2.positions
            vXZ=sel3.positions-sel2.positions
            vZ=vft.pbc_corr(vZ.T,box)
            vXZ=vft.pbc_corr(vXZ.T,box)
            return vZ,vXZ
    return sub

def LabXY(molecule,sel1=None,sel2=None,Nuc=None,resids=None,segids=None,filter_str=None):
    """Motion projected to the XY-plane of the Lab frame. Use only for systems
    that remain aligned along z
    """
    if Nuc is not None:
        sel1,sel2=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
        sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)

    def sub():
        box=molecule.box
        v=sel1.positions-sel2.positions
        v=vft.pbc_corr(v.T,box)
        v[2]=0
        return v
    return sub

def LabZ(molecule,sel1=None,sel2=None,Nuc=None,resids=None,segids=None,filter_str=None):
    """Motion projected to the Z-axis of the Lab frame. Use only for systems
    that remain aligned along z
    """
    if Nuc is not None:
        sel1,sel2=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
        sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)

    def sub():
        box=molecule.box
        v=sel1.positions-sel2.positions
        v=vft.pbc_corr(v.T,box)
        v[:2]=0
        return v
    return sub

def bond_rotate(molecule,sel1=None,sel2=None,sel3=None,Nuc=None,resids=None,segids=None,filter_str=None):
    """
    Rotation around a given bond, defined by sel1 and sel2. Has a very similar
    effect to simply using bond with the same sel1 and sel2. However, an addition
    selection is created to a third atom. Then, the vector between sel1 and
    sel2 defines the rotation axis. However, rotation around this axis caused
    by more distant motions is removed, because a third selection (sel3) is
    used with sel2 to create a second vector, which then remains in the xz plane
    
    (if only sel1 and sel2 are specified for rotation, then some rotation further
    up a carbon chain, for example, may not move the vector between sel1 and sel2,
    but does cause rotation of the inner bonds- in most cases it is not clear if
    this is happening, but becomes particularly apparent when rotation appears
    on double bonds, where rotation should be highly restricted)
    
    sel3 may be defined, but is not required. If it is not provided, a third 
    atom will be found that is bound to sel2 (this frame won't work if sel2 is
    not bound to any other atom). 
    """
    
    if Nuc is not None:
        sel1,sel2=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
        sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)
        
    if sel3 is not None:
        sel3=selt.sel_simple(molecule,sel3,resids,segids,filter_str)
    else:
        resids=np.unique(sel1.resids)
        i=np.isin(sel1.universe.residues.resids,resids)    #Filter for atoms in the same residues
        sel0=sel1.universe.residues[i].atoms
        sel3=selt.find_bonded(sel2,sel0,sel1,n=1,sort='cchain')[0]
        


    def sub():
        box=molecule.box
        v1=sel1.positions-sel2.positions
        v2=sel2.positions-sel3.positions
        v1=vft.pbc_corr(v1.T,box)
        v2=vft.pbc_corr(v2.T,box)
        return v1,v2
    return sub

def superimpose(molecule,sel=None,resids=None,segids=None,filter_str=None,sigma=0):
    """
    Superimposes a selection of atoms to a reference frame (the first frame)
    
    Note that we may have multiple selections. In this case, then at least some
    of the arguments will be lists or higher dimensional. For this purpose, the
    sel_lists function is used (in select_tools.py)
    
    f=superimpose(molecule,sel=None,resids,None,segids=None,filter_str=None)
    
    f() returns vectors representing the rotation matrix
    """
    
    sel=selt.sel_lists(molecule,sel,resids,segids,filter_str)    
    uni=molecule.uni
    "Calculate the reference vectors"
    uni.trajectory.rewind()
    molecule.traj[0]
    vref=list()
    i0=list()
    for s in sel:
        vr=s.positions
        i0.append(vft.sort_by_dist(vr))
        vref.append(np.diff(vr[i0[-1]],axis=0).T)
       
    def sub():
        R=list()
        box=molecule.box
        for s,vr,i in zip(sel,vref,i0):
            v=vft.pbc_corr(np.diff(s.positions[i],axis=0).T,box)   #Calculate vectors, periodic boundary correction
            R.append(vft.RMSalign(vr,v))    #Get alignment to reference vector
            
        return vft.R2vec(R)     #This converts R back into two vectors
    
    #Construct the frame index (can be overridden by the user)
    if molecule.sel1 is not None:
        frame_index=[]
        for resid in molecule.sel1.resids:
            for k,sel0 in enumerate(sel):
                if resid in sel0.resids:
                    frame_index.append(k)
                    break
            else:
                frame_index.append(np.nan)
        return sub,np.array(frame_index),{'PPfun':'AvgGauss','sigma':sigma}
    
    return sub,None,{'PPfun':'AvgGauss','sigma':sigma}
            

def chain_rotate(molecule,sel=None,Nuc=None,resids=None,segids=None,filter_str=None):
    """
    Creates a frame for which a chain of atoms (usually carbons) is aligned
    such that the vector formed by the previous and next heteroatom (not 1H)
    are aligned along z.
    
    Note that the frame is selected with a single initial selection, and the
    function automatically searches for the surrounding atoms. In case a methyl
    carbon is included, the rotation is defined by the carbon itself and its
    nearest neighbor, instead of the surrounding two atoms (which would then
    have to include a methyl proton)
    """

    uni=molecule.uni

    "Get the initial selection"
    if Nuc is not None:
        sel,_=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel=selt.sel_simple(molecule,sel,resids,segids,filter_str)
    
    "Get all atoms in the residues included in the initial selection"
    resids=np.unique(sel.resids)
    sel0=uni.residues[np.isin(uni.residues.resids,resids)].atoms
    
    "Get bonded"
    sel1,sel2=selt.find_bonded(sel,sel0=sel0,n=2,sort='cchain')
    
    "Replace 1H with the original selection"
    i=sel2.types=='H'
    
    sel20=sel2
    sel2=uni.atoms[:0]
    for s2,s,i0 in zip(sel20,sel,i):
        if i0:
            sel2+=s
        else:
            sel2+=s2
            
    
    def sub():
        box=molecule.box
        v=sel2.positions-sel1.positions
        v=vft.pbc_corr(v.T,box)
        return v
    return sub

    
def methylCC(molecule,Nuc=None,resids=None,segids=None,filter_str=None,sigma=0):
    """
    Superimposes the C-X bond attached to a methyl carbon, and can separate
    methyl rotation from re-orientation of the overall methyl group
    
    Note- we only return one copy of the C–C bond, so a frame index is necessary
    """            
    
    if Nuc is None:
        Nuc='ch3'
    selC1,_=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)  
    selC1=selC1[::3]    #Above line returns 3 copies of each carbon. Just take 1 copy     
    
    resids=molecule.uni.residues.resids
    sel0=molecule.uni.residues[np.isin(resids,selC1.resids)].atoms
    selC2=selt.find_bonded(selC1,sel0,n=1,sort='cchain')[0]
    selC3=selt.find_bonded(selC2,sel0,exclude=selC1,n=1,sort='cchain')[0]
#    
#    selC2=sum([sel0.select_atoms('not name H* and around 1.6 atom {0} {1} {2}'\
#                                 .format(s.segid,s.resid,s.name)) for s in selC1])
    
    def sub():
        box=molecule.box
        v1,v2=selC1.positions-selC2.positions,selC2.positions-selC3.positions
        v1,v2=[vft.pbc_corr(v.T,box) for v in [v1,v2]]
        return v1,v2
    frame_index=np.arange(len(selC1)).repeat(3)
    return sub,frame_index,{'PPfun':'AvgGauss','sigma':sigma}

def side_chain_chi(molecule,n_bonds=1,Nuc=None,resids=None,segids=None,filter_str=None,sigma=0):
    """
    Returns a frame that accounts for motion arounda given bond in the side chain,
    where we are interested in the total methyl dynamics. Ideally, the product of
    all side chain rotations plus the backbone motion and methyl rotation yields
    the total motion. One should provide the same selection arguments as used for
    the methylCC frame, plus one additional argument, n_bonds, which determines
    how many bonds away from the methyl group we define the frame. 
    
    Note that, due to different side chain lengths, some frames defined this way
    will not be defined, because n_bonds is too large. For example, side_chain_chi
    will never return a frame for an alanine group, and valine will only yield a
    frame for n_bonds=1. This should not cause an error, but rather will result
    in np.nan found in the returned frame index.
    """
    
    if Nuc is None:
        Nuc='ch3'
    selC,_=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)  
    selC=selC[::3]    #Above line returns 3 copies of each carbon. Just take 1 copy
    
    frame_index=list()
    sel1,sel2,sel3=None,None,None
    k=0
    for s in selC:
        chain=selt.get_chain(s,s.residue.atoms)[3+n_bonds:6+n_bonds]
        if len(chain)==3:
            frame_index.extend([k,k,k])
            k+=1
            if sel1 is None:
                sel1,sel2,sel3=chain[0:1],chain[1:2],chain[2:3]
            else:
                sel1=sel1+chain[0]
                sel2=sel2+chain[1]
                sel3=sel3+chain[2]
        else:
            frame_index.extend([np.nan,np.nan,np.nan])
    frame_index=np.array(frame_index)
    def sub():
        box=molecule.box
        vZ=sel1.positions-sel2.positions
        vXZ=sel2.positions-sel3.positions
        vZ=vft.pbc_corr(vZ.T,box)
        vXZ=vft.pbc_corr(vXZ.T,box)
        return vZ,vXZ
    
    return sub,frame_index,{'PPfun':'AvgGauss','sigma':sigma}

def librations(molecule,sel1=None,sel2=None,Nuc=None,resids=None,segids=None,filter_str=None,full=True,sigma=0):
    """
    Defines a frame for which librations are visible. That is, for a given bond,
    defined by sel1 and sel2, we search for other atoms bound to the 
    heteroatom (by distance). The reference frame is then defined by the 
    heteroatom and the bonded atoms, leaving primarily librational
    motion of the bond. We preferentially select the other two atoms for larger
    masses, but they may also be protons (for example, a methyl H–C bond will 
    be referenced to the next carbon but also another one of the protons of 
    the methyl group)
    
    In case the heteroatom only has two bound partners, the second atom in the
    bond will also be used for alignment, reducing the effect motion
    (not very common in biomolecules)
    
    librations(sel1,sel2,Nuc,resids,segids,filter_str)
    """
    if Nuc is not None:
        sel1,sel2=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
        sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)
        
    if sel1.masses.sum()<sel2.masses.sum():
        sel1,sel2=sel2,sel1 #Make sel1 the heteroatom
    
    resids=np.unique(sel1.resids)
    i=np.isin(sel1.universe.residues.resids,resids)    #Filter for atoms in the same residues
    sel0=sel1.universe.residues[i].atoms
    if full:
        "Slightly improved performance if we align all 4 bonds to the carbon"
        "Note, "
        sel2,sel3,sel4,sel5=selt.find_bonded(sel1,sel0,n=4,sort='mass')
        
        def vfun():
            v=list()
            for v1,v2,v3,v4,v5 in zip(sel1,sel2,sel3,sel4,sel5):
                v0=np.array([v2.position-v1.position,
                            v3.position-v1.position,
                            v4.position-v1.position,
                            v5.position-v1.position])
                box=molecule.box
                v.append(vft.pbc_corr(v0.T,box))
            return v
        
        molecule.traj[0]
        
        vref=vfun()
        
        def sub():
            R=list()
            vecs=vfun()
            R=[vft.RMSalign(vr,v) for v,vr in zip(vecs,vref)]
            return vft.R2vec(R)
    else:
        sel2,sel3=selt.find_bonded(sel1,sel0,n=2,sort='mass')
        
        def sub():
            box=molecule.box
            v1=sel2.positions-sel1.positions
            v2=sel1.positions-sel3.positions
            v1=vft.pbc_corr(v1.T,box)
            v2=vft.pbc_corr(v2.T,box)
            return v1,v2
    return sub,None,{'PPfun':'AvgGauss','sigma':sigma}
    
def librations0(molecule,sel1=None,sel2=None,Nuc=None,resids=None,segids=None,filter_str=None):
    """
    Defines a frame for which librations are visible. That is, for a given bond,
    defined by sel1 and sel2, we search for two other atoms bound to the 
    heteroatom (by distance). The reference frame is then defined by the 
    heteroatom and the additional two atoms, leaving primarily librational
    motion of the bond. We preferentially select the other two atoms for larger
    masses, but they may also be protons (for example, a methyl H–C bond will 
    be referenced to the next carbon but also another one of the protons of 
    the methyl group)
    
    In case the heteroatom only has two bound partners, the second atom in the
    bond will also be used for alignment, reducing the effect motion
    (not very common in biomolecules)
    
    librations(sel1,sel2,Nuc,resids,segids,filter_str)
    """
    if Nuc is not None:
        sel1,sel2=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
        sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)
        
    if sel1.masses.sum()<sel2.masses.sum():
        sel1,sel2=sel2,sel1 #Make sel1 the heteroatom
    
    resids=np.unique(sel1.resids)
    i=np.isin(sel1.universe.residues.resids,resids)    #Filter for atoms in the same residues
    sel0=sel1.universe.residues[i].atoms
    sel2,sel3,sel4,sel5=selt.find_bonded(sel1,sel0,n=4,sort='mass')
    
    def vfun():
        v=list()
        for v1,v2,v3,v4,v5 in zip(sel1,sel2,sel3,sel4,sel5):
            v0=np.array([v2.position-v1.position,
                        v3.position-v1.position,
                        v4.position-v1.position,
                        v5.position-v1.position])
            box=molecule.box
            v.append(vft.pbc_corr(v0.T,box))
        return v
    
    molecule.traj[0]
    
    vref=vfun()
    
    def sub():
        R=list()
        vecs=vfun()
        R=[vft.RMSalign(vr,v) for v,vr in zip(vecs,vref)]
        return vft.R2vec(R)
        
    return sub        

def MOIz(molecule,sel,resids=None,segids=None,filter_str=None):
    """
    Defines a frame for which the moment of inertia of a set of atoms remains
    aligned along the z-axis. Note, atomic mass is NOT considered, all atoms
    have equal influence.
    
    MOIz(sel,resids,segids,filter_str)
    
    """
    
    sel=selt.sel_lists(molecule,sel,resids,segids,filter_str)    
    molecule.traj[0]
    
    box=molecule.box
    
    for k,s in enumerate(sel):
        vr=s.positions
        i0=vft.sort_by_dist(vr)
        sel[k]=sel[k][i0]
        
    vref=list()
    for s in sel:
        v0=vft.pbc_pos(s.positions.T,box)
        vref.append(vft.principle_axis_MOI(v0)[:,0])
        
    
    def sub():
        v=list()
        box=molecule.box
        for s,vr in zip(sel,vref):
            v0=vft.pbc_pos(s.positions.T,box)
            v1=vft.principle_axis_MOI(v0)[:,0]
            v.append(v1*np.sign(np.dot(v1,vr)))  #This keeps the MOI vector from doing 180 degree flips
        return np.array(v).T
       
    return sub

def MOIxy(molecule,sel,sel1=None,sel2=None,Nuc=None,index=None,resids=None,segids=None,filter_str=None):
    """
    Separates out rotation within the moment of inertia frame (should be used in
    conjunction with MOIz). That is, we identify rotational motion, where the z-axis
    is the direction of the Moment of Inertia vector. 
    
    The user must provide one or more selections to define the moment of inertia 
    (sel). The user must also provide the selections to which the MOI is applied
    (sel1 and sel2, or Nuc). Additional filters will be used as normal, applied 
    to all selections (resids,segids,filter_str). In case multiple MOI selections
    are provided (in a list), the user must provide an index, to specifify which
    bond goes with which MOI selection. This should usually be the same variable
    as provided for the frame_index when using MOIz (and one will usually not
    use a frame_index when setting up this frame)
    
    MOIxy(sel,sel1=None,sel2=None,Nuc=None,index=None,resids=None,segids=None,filter_str=None)
    """
    
    sel=selt.sel_lists(molecule,sel,resids,segids,filter_str)
    
    if Nuc is not None:
        sel1,sel2=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
        sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)
    
    molecule.traj[0]
    
    
    for k,s in enumerate(sel):
        vr=s.positions
        i0=vft.sort_by_dist(vr)
        sel[k]=sel[k][i0]
        
    if index is None:
        if len(sel)==1:
            index=np.zeros(sel1.n_atoms,dtype=int)
        elif len(sel)==sel1.n_atoms:
            index=np.arange(sel1.n_atoms,dtype=int)
        else:
            print('index must be defined')
            return
        
    def sub():
        vnorm=list()
        box=molecule.box
        for s in sel:
            v0=vft.pbc_pos(s.positions.T,box)
            vnorm.append(vft.principle_axis_MOI(v0)[:,0])
        vnorm=np.array(vnorm)
        
        #Pre-allocate output vector, to point along z
        v1=np.zeros([3,sel1.n_atoms])
        v1[2]=1
        v2=v1.copy()
        
        v0=vft.pbc_corr((sel1.positions-sel2.positions).T,box)
        for k,vn in enumerate(vnorm):
            v1[:,k==index]=vft.projXY(v0[:,k==index],vn)
            v2[:,k==index]=np.array([vn]).T.repeat((k==index).sum(),axis=1)
        
        return v1,v2
    
    return sub

def MOIbeta(molecule,sel,sel1=None,sel2=None,Nuc=None,index=None,resids=None,segids=None,filter_str=None):
    """
    Separates out rotation within the moment of inertia frame (should be used in
    conjunction with MOIz). That is, we identify rotational motion, where the z-axis
    is the direction of the Moment of Inertia vector. 
    
    The user must provide one or more selections to define the moment of inertia 
    (sel). The user must also provide the selections to which the MOI is applied
    (sel1 and sel2, or Nuc). Additional filters will be used as normal, applied 
    to all selections (resids,segids,filter_str). In case multiple MOI selections
    are provided (in a list), the user must provide an index, to specifify which
    bond goes with which MOI selection. This should usually be the same variable
    as provided for the frame_index when using MOIz (and one will usually not
    use a frame_index when setting up this frame)
    
    MOIxy(sel,sel1=None,sel2=None,Nuc=None,index=None,resids=None,segids=None,filter_str=None)
    """
    
    sel=selt.sel_lists(molecule,sel,resids,segids,filter_str)
    
    if Nuc is not None:
        sel1,sel2=selt.protein_defaults(Nuc,molecule,resids,segids,filter_str)
    else:
        sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
        sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)
    
    molecule.traj[0]
    
    
    sel=selt.sel_lists(molecule,sel,resids,segids,filter_str)    
    
    box=molecule.box
    
    for k,s in enumerate(sel):
        vr=s.positions
        i0=vft.sort_by_dist(vr)
        sel[k]=sel[k][i0]
        
    vref=list()
    for s in sel:
        v0=vft.pbc_pos(s.positions.T,box)
        vref.append(vft.principle_axis_MOI(v0)[:,0])
        
    
    def MOIsub():
        v=list()
        box=molecule.box
        for s,vr in zip(sel,vref):
            v0=vft.pbc_pos(s.positions.T,box)
            v1=vft.principle_axis_MOI(v0)[:,0]
            v.append(v1*np.sign(np.dot(v1,vr)))
        return np.array(v).T
    
#    for k,s in enumerate(sel):
#        vr=s.positions
#        i0=vft.sort_by_dist(vr)
#        sel[k]=sel[k][i0]
        
    if index is None:
        if len(sel)==1:
            index=np.zeros(sel1.n_atoms,dtype=int)
        elif len(sel)==sel1.n_atoms:
            index=np.arange(sel1.n_atoms,dtype=int)
        else:
            print('index must be defined')
            return 

#    vref=list()
#    box=molecule.box
#    for s in sel:
#        v0=vft.pbc_pos(s.positions.T,box)
#        vref.append(vft.principle_axis_MOI(v0)[:,0])

    def sub():
        vnorm=MOIsub()
        #Pre-allocate output vector, to point along z
        vZ=np.zeros([3,sel1.n_atoms])
        vZ[2]=1     #If a bond not in index, then vZ just on Z
#        vXZ=np.zeros([3,sel1.n_atoms])
#        vXZ[0]=1    #If a bond not in index, then vXZ just along x
        
        sc=np.array(vft.getFrame(vnorm)).T
        v00=vft.norm(vft.pbc_corr((sel1.positions-sel2.positions).T,box))
        
        for k,(vn,sc0) in enumerate(zip(vnorm.T,sc)):
            v0=v00[:,k==index]
            cb=v0[0]*vn[0]+v0[1]*vn[1]+v0[2]*vn[2]  #Angle between MOI and bond
            cb[cb>1]=1.0
            sb=np.sqrt(1-cb**2)
            v0=np.concatenate(([sb],[np.zeros(sb.shape)],[cb]),axis=0)
            "Here, we keep the vector fixed in the xz plane of the MOI frame"
            vZ[:,k==index]=vft.R(v0,*sc0)
#            vZ[:,k==index]=v0
#            vXZ[:,k==index]=np.atleast_2d(vn).T.repeat(v0.shape[1],1)
        
        return vZ
    
    return sub


def aromatic_plane(molecule,resids=None,segids=None,filter_str:str=None,sigma:float=0)->tuple:
    """
    Frame which returns a vector perpendicular to the plane of an aromatic amino
    acid (F,H,W,Y) and a second vector pointing from the center of ring to the
    CB.

    Parameters
    ----------
    molecule : TYPE
        Selection object.
    resids : TYPE, optional
        List of residues for which we should return aromatic planes. 
        The default is None.
    segids : TYPE, optional
        List of segments for which we should return aromatic planes.
        The default is None.
    filter_str : str, optional
        string which filters the selection using MDAnalysis format. 
        The default is None.
    sigma : float, optional
        Parameter to determine Gaussian moving average in post processing. 
        The default is 0 (no post processing).

    Returns
    -------
    tuple
        Frame function, frame_index (np.array), dict (Post Processing info)

    """
    
    
    sel0=selt.aromatic_plane(molecule,resids=resids,segids=segids,filter_str=filter_str)
    
    sel=list()
    selCB=list()
    for k,s in enumerate(sel0):
        if len(s):
            sel.append(s)
            selCB.append(s[s.names=='CB'])
            
    #Make a frame index
    if molecule.sel1 is not None:
        mdmode=molecule._mdmode
        molecule._mdmode=True
        frame_index=np.ones(len(molecule.sel1))*np.nan
        sel1=molecule.sel1
        for k,s in enumerate(sel):
            frame_index[np.logical_and(s.resids[0]==sel1.resids,s.segids[0]==sel1.segids)]=k
        molecule._mdmode=mdmode
    else:
        frame_index=np.repeat(np.arange(len(sel0)),2) #Just a best guess if no other information available
        
        
    
    def sub():
        vZ=np.zeros([len(sel),3])
        vXZ=np.zeros([len(sel),3])
        for k,(s,sCB) in enumerate(zip(sel,selCB)):
            vZ[k]=vft.RMSplane(s.positions.T)
            vXZ[k]=(s-sCB).positions.mean(0)-sCB.positions
        return vZ.T,vXZ.T
    
    return sub,frame_index,{'PPfun':'AvgGauss','sigma':sigma}

