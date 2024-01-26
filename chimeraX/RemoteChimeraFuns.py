#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:25:34 2022

@author: albertsmith
"""

import numpy as np
from time import sleep
from matplotlib.pyplot import get_cmap


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
    atoms[ids].radii=(0.8+sc*x).astype('float32')
    atoms[ids].colors=color_calc(x,colors=[[210,180,140,255],color])
    
def set_color_radius_CC(atoms,x:np.array,color:list,ids:list,sc:float=4) -> None:
    """
    Set the color and radius indicating cross-correlated motion for a given
    detector.
    
    Works essentially like set_color_radius, except that we check for what atom
    is currently selected and use this to determine which cross correlations we
    show. Note that if no atom is selected, we take the previous selection, which
    we determine here internally by searching for the largest radius

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
    
    r=list()
    for k,id0 in enumerate(ids):
        a0=atoms[np.array(id0,dtype=int)]
        if np.any(a0.selected) or np.any(a0.bonds.selected):
            x=x[k]
            break
        else:
            r.append(a0.radii.max())
    else:
        k=np.argmax(r)
        x=x[k]
    # x/=x[k]
    x[k]=1
    set_color_radius(atoms,x,color,ids,sc)
    atoms[np.array(ids[k],dtype=int)].colors=color_calc(np.ones(ids[k].shape),colors=[[0,0,0,255],[0,0,0,255]])


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
            

class DetFader():
    def __init__(self,model,x:np.array,ids:list,tau:np.array,rhoz:np.array,tc:np.array,sc:float=3):
        """
        Calculates and sets color and radii for fading through multiple detector
        sensitivities in chimeraX. 
    
        Parameters
        ----------
        model: Model in chimeraX
        x : np.array
            Scaled detector responses (x is normally between 0 and 1).
        ids : list
            List of atom ids used for coloring.
        sc : float, optional
            Scaling of the atom radius. The default is 4.
        tau : np.array
            Correlation times to be swept over
        rhoz : np.array
            Detector sensitivities. Usually nx200 (usually normalized with 'MP' normalization)
        tc : np.array
            Correlation time axis to go with rhoz. Usually 200 elements
    
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
        x=np.array([x0[i==id0].mean(0) for i in ids])  
        
        
        self.model=model
        self.ids=ids
        self.atoms=model.atoms #These are the selection of atoms that will get updated
        
        x[x<0]=0 #No negative responses
        #We pre-calculate the radius and color for each element of the trajectory
        r=list()
        clr=list()
        
        #Colors to use
        clr0=[210,180,140,255] #Tan
        cmap=(np.array([get_cmap("tab10")(i%10) for i in range(rhoz.shape[0])])*255).astype(int).T #default color cycle
        
        
        for t in tau:
            itc=np.argmin(np.abs(tc*1e9-t))  #Index for sensitivity
            r.append(0.8+(x@rhoz[:,itc])*sc)
            clr.append([(c*(x*rhoz[:,itc])).sum(-1)+c0*(1-(x*rhoz[:,itc]).sum(-1)) for c,c0 in zip(cmap,clr0)])
        
        self.r=np.array(r)    
        self.clr=np.array(clr,dtype=int).swapaxes(1,2)
        self.clr[self.clr>255]=255
        self.clr[self.clr<0]=0
        
        self.i=-1
        print('Detector Fader initialized')

    def set_color_radius(self) -> None:
        """
        Update the radius and colors for the ith element of the trajectory

        Parameters
        ----------
        i : int
            Index for the current frame of the trajectory.

        Returns
        -------
        None.

        """
        i=self.model.active_coordset_id-1
        # print(i)
        if i!=self.i:
            # self.atoms[self.ids].radii=self.r[i].astype('float32')
            self.atoms[self.ids].colors=self.clr[i]
            self.atoms[self.ids].radii=self.r[i].astype('float32')
            # r=self.atoms.radii
            # r[self.ids]=self.r[i].astype('float32')
            # self.model.set_ra
            # clrs=self.atoms.colors
            # clrs[self.ids]=self.clr[i]
            # self.model.set_colors(clrs)
            self.i=i
 
        
#%% The following function requires a listener in pyDR (threading)
def xtc_request(atoms,xtc_type:str,rho_index:int,ids:list,cmx,file):
    """
    Requests a particular xtc from pyDR

    Parameters
    ----------
    atoms : Atom group
        Atoms in the relevant model.
    xtc_type : str
        DESCRIPTION.
    rho_index : TYPE
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    cmx : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    for k,id0 in enumerate(ids):
        a0=atoms[np.array(id0,dtype=int)]
        if np.any(a0.selected) or np.any(a0.bonds.selected):
            q=k
            break
    else:
        q=-1
    # cmx.client.send(('xtc_request',xtc_type,rho_index,q))
    with open(file,'w') as f:
        f.write(f'xtc_request {xtc_type} {rho_index} {q}')
    print('xtc_request',xtc_type,rho_index,q)
    # sleep(.5)
    
#%% THe functions below will typically be used outside of event loops.
#cmx (the CMXReceiver instance) should always be the first argument!!
    


def draw_tensors(cmx,A,Aiso=None,pos=None,colors=((1,.39,.39,1),(.39,.39,1,1)),comp='Azz'):
    """
    Draws tensors in chimeraX. A is an Nx5 list of tensors. These may be placed
    along bonds, identified by an atom 

    Parameters
    ----------
    cmx : TYPE
        cmx object
    A : array
        Nx5 (or 5 elements) array containing the tensors to be plotted.
    colors : tuple,optional
        Two colors defining positive and negative parts of tensors. Each of
        the two elements should be a list/tuple with 4 elements on a scale
        from 0 to 1 (R,G,B,alpha).
    atoms : TYPE, optional
        DESCRIPTION. The default is None.
    pos : TYPE, optional
        Nx3 list of positions for the tensors. 
        The default is None, which will space the tensors every 1.5 Angstroms along the x-axis

    Returns
    -------
    None.

    """
    from Surfaces import load_sphere_surface
    A=np.atleast_2d(A).astype(complex)
    assert A.shape[1]==5,"A must be an Nx5 array"
    N=A.shape[0]
    if Aiso is None:Aiso=np.zeros(N)
    
    
    session=cmx.session
    # if pos is None:
    #     pos=np.zeros((N,3),dtype=float)
    #     pos[:,0]=np.arange(N)*1.5
    

    load_sphere_surface(session, A, Aiso,pos,colors,comp)
        
    
  
def draw_surface(cmx,x,y,z,colors:list=(.3,.3,.3,1)):
    """
    Draws a surface in ChimeraX, given axes x,y, and amplitude, z. One may
    also specify the color at each point, with a list with the same length
    as the number of elements in z (or a Nx*Ny*3/4 element vector)

    Parameters
    ----------
    cmx : TYPE
        cmx object.
    x : np.array
        x-axis of surface (Nx points).
    y : np.array
        y-axis of surface (Ny points).
    z : np.array
        Amplitude (Nx*Ny points)
    colors : list, optional
        Nx*Ny list of colors or one color (3-4 elements) The default is (.3,.3,.3,1).

    Returns
    -------
    None.

    """
    
    from Surfaces import load_cart_surface
    
    colors=np.array(colors)
    if colors.ndim>1:
        colors=np.reshape(colors,[z.size,colors.shape[-1]])
    load_cart_surface(cmx.session,x,y,z,colors=colors)




      