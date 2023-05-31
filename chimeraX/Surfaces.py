#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:22:29 2023

@author: albertsmith
"""

"""
Thanks to Tom Goddard at ChimeraX for explaining to me how to implement this!!
https://rbvi.github.io/chimerax-recipes/spherical_harmonics/spherical_harmonics.html 

and also for explaining a bit how threading works in ChimeraX and resolving 
issues where it threading creates problems.

"""

import numpy as np
import sys
import os
path=os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],'MDtools')
sys.path.append(path)
from vft import Spher2pars,d2
# from pyDR.MDtools.vft import Spher2pars,d2

def sphere_triangles(theta_steps=100,phi_steps=50):
    """
    Creates arrays of theta and phi angles for plotting spherical tensors in ChimeraX.
    Also returns the corresponding triangles for creating the surfaces
    """
    
    theta=np.linspace(0,2*np.pi,theta_steps,endpoint=False).repeat(phi_steps)
    phi=np.repeat([np.linspace(0,np.pi,phi_steps,endpoint=True)],theta_steps,axis=0).reshape(theta_steps*phi_steps)
    
    triangles = []
    for t in range(theta_steps):
        for p in range(phi_steps-1):
            i = t*phi_steps + p
            t1 = (t+1)%theta_steps
            i1 = t1*phi_steps + p
            triangles.append((i,i+1,i1+1))
            triangles.append((i,i1+1,i1))
    
    return theta,phi,triangles



def spherical_surface(delta,eta=None,euler=None,pos=None,Aiso=0,sc=2.09,
                      theta_steps = 100,
                      phi_steps = 50,
                      positive_color = (255,100,100,255), # red, green, blue, alpha, 0-255 
                      negative_color = (100,100,255,255),comp='Azz'):
    """
    Function for generating a surface in ChimeraX. delta, eta, and euler angles
    should be provided, as well positions for each tensor (length of all arrays
    should be the same, that is (N,), (N,), (3,N), (3,N) respectively.
    
    Returns arrays with the vertices positions (Nx3), the triangles definitions
    (list of index triples, Nx3), and a list of colors (Nx4)
    
    Component (comp) is by default the zz component. However, one may request
    Axz,Ayz,Ayz, etc. or may request the component by index (-2,-1,0,1,2)
    
    xyz,tri,colors=spherical_surface(delta,eta=None,euler=None,pos=None,
                                     theta_steps=100,phi_steps=50,
                                     positive_color=(255,100,100,255),
                                     negative_color=(100,100,255,255),comp='Azz')
    """
    # Compute vertices and vertex colors
    a,b,triangles=sphere_triangles(theta_steps,phi_steps)
    
    if euler is None:euler=[0,0,0]
    if pos is None:pos=[0,0,0]
    if eta is None:eta=0
    
    # Compute r for each set of angles
    sc=np.sqrt(2/3)*sc
    
    A=[-1/2*delta*eta,0,np.sqrt(3/2)*delta,0,-1/2*delta*eta]   #Components in PAS
    
    #0 component after rotation by a and b
    if isinstance(comp,str):
        if comp=='Azz':
            APAS=np.sqrt(2/3)*np.array([A[mp+2]*d2(b,m=0,mp=mp)*np.exp(1j*mp*a) for mp in range(-2,3)]).sum(axis=0).real
        else:
            if comp=='Ayy':
                weight=[-.5,0,-np.sqrt(1/6),0,-.5]
            elif comp=='Axx':
                weight=[0.5,0,-np.sqrt(1/6),0,0.5]
            elif comp in ['Axy','Ayx']:
                weight=[0.5*1j,0,0,0,-0.5*1j]
            elif comp in ['Axz','Azx']:
                weight=[0,0.5,0,-0.5,0]
            elif comp in ['Ayz','Azy']:
                weight=[0,0.5*1j,0,0.5*1j,0]
            APAS=np.zeros(b.shape,dtype=complex)
            for m,wt in zip(range(-2,3),weight):
                APAS+=wt*np.array([A[mp+2]*d2(b,m=m,mp=mp)*np.exp(1j*mp*a) for mp in range(-2,3)]).sum(axis=0).real
    else:
        APAS=np.array([A[mp+2]*d2(b,m=comp,mp=mp)*np.exp(1j*mp*a) for mp in range(-2,3)]).sum(axis=0).real
    APAS=APAS.astype(float)
    #Coordinates before rotation by alpha, beta, gamma
    x0=(np.cos(a)*np.sin(b)*np.abs(APAS+Aiso))*sc/2
    y0=(np.sin(a)*np.sin(b)*np.abs(APAS+Aiso))*sc/2
    z0=(np.cos(b)*np.abs(APAS+Aiso))*sc/2

    # alpha,beta,gamma=euler
    # alpha,beta,gamma=-alpha,-beta,-gamma    #Added 30.09.21 along with edits to vf_tools>R2euler
    # #Rotate by alpha
    # x1,y1,z1=x0*np.cos(alpha)+y0*np.sin(alpha),-x0*np.sin(alpha)+y0*np.cos(alpha),z0
    # #Rotate by beta
    # x2,y2,z2=x1*np.cos(beta)-z1*np.sin(beta),y1,np.sin(beta)*x1+np.cos(beta)*z1
    # #Rotate by gamma
    # x,y,z=x2*np.cos(gamma)+y2*np.sin(gamma),-x2*np.sin(gamma)+y2*np.cos(gamma),z2
    
    
    alpha,beta,gamma=euler
    # alpha,beta,gamma=-gamma,-beta,-alpha  #If you add this line back, you'll break your tensor_rotation class :-(
    #Rotate by alpha
    x1,y1,z1=x0*np.cos(alpha)-y0*np.sin(alpha),x0*np.sin(alpha)+y0*np.cos(alpha),z0
    #Rotate by beta
    x2,y2,z2=x1*np.cos(beta)+z1*np.sin(beta),y1,-np.sin(beta)*x1+np.cos(beta)*z1
    #Rotate by gamma
    x,y,z=x2*np.cos(gamma)-y2*np.sin(gamma),x2*np.sin(gamma)+y2*np.cos(gamma),z2
    

    x=x+pos[0]
    y=y+pos[1]
    z=z+pos[2]
    
#    xyz=[[x0,y0,z0] for x0,y0,z0 in zip(x,y,z)]
    #Determine colors
    colors=np.zeros([APAS.size,4],np.uint8)
    colors[(APAS+Aiso)>=0]=positive_color
    colors[(APAS+Aiso)<0]=negative_color
    

    # Create numpy arrays
#    xyz = np.array(xyz, np.float32)
    xyz=np.ascontiguousarray(np.array([x,y,z]).T,np.float32)       #ascontiguousarray forces a transpose in memory- not just editing the stride
    colors = np.array(colors, np.uint8)
    tri = np.array(triangles, np.int32)

    return xyz,tri,colors

def load_sphere_surface(session,A,Aiso=None,Pos=None,colors=((1,.39,.39,1),(.39,.39,1,1)),comp='Azz'):
    
    theta_steps,phi_steps=100,50
    pc,nc=[[int(c*255) for c in color0] for color0 in colors]
    A=np.atleast_2d(A)
    if Pos is None:
        Pos=np.zeros([A.shape[0],3])
        Pos[:,0]=np.arange(A.shape[0],dtype=float)*2
    Pos=np.atleast_2d(Pos)

    Delta,Eta,*Euler=Spher2pars(A.T,return_angles=True)  #Convert into parameters for better plotting

    Aiso=np.zeros(A.shape[0]) if Aiso is None else np.atleast_1d(Aiso)

    from chimerax.core.models import Surface
    from chimerax.surface import calculate_vertex_normals,combine_geometry_vntc

    geom=list()
    for k,(delta,eta,euler,pos,Aiso0) in enumerate(zip(Delta,Eta,np.array(Euler).T,Pos,Aiso)):
        xyz,tri,colors=spherical_surface(delta=delta,eta=eta,euler=euler,pos=pos,Aiso=Aiso0,\
                                         theta_steps=theta_steps,\
                                         phi_steps=phi_steps,\
                                         positive_color=pc,\
                                         negative_color=nc,comp=comp)

        norm_vecs=calculate_vertex_normals(xyz,tri)
        
        geom.append((xyz,norm_vecs,tri,colors)) 
    
        
    xyz,norm_vecs,tri,colors=combine_geometry_vntc(geom)    
    s = Surface('surface',session)
    
    s.set_geometry(xyz,norm_vecs,tri)
    s.vertex_colors = colors
    
    session.models.add([s])

    return s

def load_cart_surface(session,x,y,z,colors:list=(.3,.3,.3,1)):
    """
    Creates and loads a cartesian surface, defined by an x and y axis, with 
    amplitude z. z may be 2D, but can also be flattened. In any case, it should
    Nx*Ny elements. Colors can be set to a single color, or be defined for
    every point, z.

    Parameters
    ----------
    session : ChimeraX session
        DESCRIPTION.
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
    
    from scipy.spatial import Delaunay
    from chimerax.core.models import Surface
    from chimerax.surface import calculate_vertex_normals
    
    x,y=[a.flatten() for a in np.meshgrid(x,y)]
    tri=Delaunay(np.array([x,y]).T).vertices
    
    
    
    xyz=np.ascontiguousarray(np.array([x.flatten(),y.flatten(),z.flatten()]).T,np.float32)
    
    if hasattr(colors[0],'__len__'):
        assert len(colors)==xyz.shape[0],'Colors must have the same number of elements as z'
        
        for k,c in enumerate(colors):
            if len(c)==3:
                colors[k]=[*c,1]
    else:
        colors=[colors for _ in range(xyz.shape[0])]
        
    # color=np.array([[(255*np.mean([colors[i0][k] for i0 in i])).astype(np.uint8) for k in range(4)] for i in tri])

    # color=list()
    # for i in tri:
    #     color.append((np.array(colors[i[0]])*255).astype(np.uint8))
    # color=np.array(color,dtype=np.uint8)

    norm_vecs=calculate_vertex_normals(xyz,tri)
    
    s=Surface('surface',session)
    s.set_geometry(xyz,norm_vecs,tri)
    s.vertex_colors=(np.array(colors)*255).astype(np.uint8)
    session.models.add([s])
    
    return s
    