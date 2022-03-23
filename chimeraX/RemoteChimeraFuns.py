#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:25:34 2022

@author: albertsmith
"""

import numpy as np


def set_color_radius(atoms,x:np.array,color:list,ids:list,sc:float=4):
    """
    Set the color and radius for a group of atoms 

    Parameters
    ----------
    atoms : atom group in chimeraX
        Set of atoms (model.atoms) to be colored.
    x : np.array
        1D array of floats defining the colors and radii.
    color : list
        Color to fade towards (4 elements, 0-255, 4th element is alpha).
    ids : list
        List of indices to determine which atoms to color.
    sc  : float
        Control how much the radius increases for a value of x=1

    Returns
    -------
    None.

    """
    
    """
    We start out with a list of x values, and a list of lists of ids, where the
    outer list is the same length as x. Here, we expand the list of lists into
    a 1d array, and copy x values to get a list the same length as the ids
    """
    x0=list()
    id0=list()
    for i,y in zip(ids,x):
        id0.extend(i)
        x0.extend([y for _ in i])
    x0=np.array(x0)
    id0=np.array(id0)
    
    """
    Now, we find unique elements in ids. Then, for each unique value, we average
    over the occurences of that value to get the new x values
    """
    ids=np.unique(id0)  
    x=np.array([x0[i==id0].mean() for i in ids])    
    "Finally, transfer the results to the atoms"
    atoms[id0].radii=0.8+sc*x0
    atoms[id0].colors=color_calc(x0,colors=[[210,180,140,255],color])



def color_calc(x,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]]):
    """
    Calculates color values for a list of values in x (x ranges from 0 to 1).
    
    These values are linear combinations of reference values provided in colors.
    We provide a list of N colors, and a list of N x0 values (if x0 is not provided,
    it is set to x0=np.linspace(0,1,N). If x is between the 0th and 1st values
    of x0, then the color is somewhere in between the first and second color 
    provided. Default colors are blue at x=0, tan at x=0.5, and red at x=1.
    
    color_calc(x,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]])
    """
    
    colors=np.array(colors,dtype='uint8')
    N=len(colors)
    if x0 is None:x0=np.linspace(0,1,N)
    x=np.array(x)
    if x.min()<x0.min():
        print('Warning: x values less than min(x0) are set to min(x0)')
        x[x<x0.min()]=x0.min()
    if x.max()>x0.max():
        print('Warning: x values greater than max(x0) are set to max(x0)')
        x[x>x0.max()]=x0.max()

    i=np.digitize(x,x0)
    i[i==len(x0)]=len(x0)-1
    clr=(((x-x0[i-1])*colors[i].T+(x0[i]-x)*colors[i-1].T)/(x0[i]-x0[i-1])).T
    return clr.astype('uint8')