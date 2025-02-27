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
Use this file to write your own frame definitions. Follow the formating found in
frames.py. A few critical points:
    
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
            uni=molecule.mda_object
            
            def sub()
                ...
                v1,v2=some_calculations
                ...
                box=uni.dimensions[:3] (periodic boundary conditions)
                v1=vft.pbc_corr(v1,box)
                v2=vft.pbc_corr(v2,box)
                
                return v1,v2
            return sub
            
"""



import numpy as np
from pyDR.MDtools import vft
from pyDR.Selection import select_tools as selt

def bondXY(molecule,sel1=1,sel2=2,resids=None,segids=None,filter_str:str=None,sigma:float=0):
    """
    

    Parameters
    ----------
    molecule : TYPE
        Selection object.
    sel1 : TYPE, optional
        First atom defining the bond. The default is 1.
    sel2 : TYPE, optional
        Second atom defining the bond. The default is 2.
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
    None.

    """
    
    if not(hasattr(sel1,'__len__')):
        sel1=(sel1+sel1)[:1]
    if not(hasattr(sel2,'__len__')):
        sel2=(sel2+sel2)[:1]
    
    sel1=selt.sel_simple(molecule,sel1,resids,segids,filter_str)
    sel2=selt.sel_simple(molecule,sel2,resids,segids,filter_str)
    
    

    
    if len(sel1)==1:
        frame_index=np.zeros(len(molecule.sel1),dtype=int)
    elif len(sel1)==len(molecule.sel1):
        frame_index=np.arange(len(sel1))
    else:
        frame_index=None
        
    def sub():
        box=molecule.box
        vXZ=sel1.positions-sel2.positions
        vXZ[:,2]=0
        vXZ=vft.pbc_corr(vXZ.T,box)
        vZ=np.zeros(vXZ.shape)
        vZ[2]=1
        return vZ,vXZ
    
    return sub,frame_index,{'PPfun':'AvgGauss','sigma':sigma}
        
        
    
    
    