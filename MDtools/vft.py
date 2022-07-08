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



Created on Wed Nov 27 13:21:51 2019

@author: albertsmith
"""

"""
Library of functions to deal with vectors and tensors, used for aligning tensors
and vectors into different frames. We assume all vectors provided are 2D numpy
arrays, with the first dimension being X,Y,Z (we do not deal with time-
dependence in these functions. This is obtained by sweeping over the trajectory.
Frames are processed at each time point separately)
"""


"""
Rotations are 
"""


import numpy as np
from scipy.linalg import svd

#%% Periodic boundary condition check
def pbc_corr(v0,box):
    """
    Corrects for bonds that may be extended across the box. Our assumption is
    that no vector should be longer than half the box. If we find such vectors,
    we will add/subtract the box length in the appropriate dimension(s)
    
    Input should be 3xN vector and 3 element box dimensions
    
    v = pbc_corr(v0,box)
    """
    
    "Copy input, take tranpose for easier calculation"
    v=v0.copy()
    if v.shape[0]==3:
        v=v.T
        tp=True
    else:
        tp=False
    
    
    i=v>box/2
    ib=np.argwhere(i).T[1]
    v[i]=v[i]-box[ib]
    
    i=v<-box/2
    ib=np.argwhere(i).T[1]
    v[i]=v[i]+box[ib]
        
    if tp:
        return v.T
    else:
        return v

#%% Periodic boundary condition for positions
def pbc_pos(v0,box):
    """
    Sometimes, we are required to work with an array of positions instead of
    a pair of positions (allowing easy calculation of a vector and determining
    if the vector wraps around the box). In this case, we take differences 
    between positions, and make sure the differences don't yield a step around
    the box edges. The whole molecule, however, may jump around the box after
    this correction. This shouldn't matter, since all calculations are orientational,
    so the center position is irrelevant.
    
    Input is a 3xN vector and a 3 element box
    """
    
    v=np.concatenate((np.zeros([3,1]),np.diff(v0,axis=1)),axis=1)
    v=pbc_corr(v,box)
    
    return np.cumsum(v,axis=1)+np.atleast_2d(v0[:,0]).T.repeat(v.shape[1],axis=1)
    
    
#%% Vector normalization
def norm(v0):
    """
    Normalizes a vector to a length of one. Input should be a 3xN vector.
    """
    if v0 is None:
        return None
    
#    X,Y,Z=v0
#    length=np.sqrt(X**2+Y**2+Z**2)
#    
#    return v0/length
    v0=np.array(v0)
    return v0/np.sqrt((v0**2).sum(0))

#%% Reverse rotation direction (passive/active)
def pass2act(cA,sA,cB,sB=None,cG=None,sG=None):
    """
    After determining a set of euler angles, we often want to apply them to go
    into the reference frame corresponding to those angles. This requires
    reversing the rotation, performed by this function
    
    -gamma,-beta,-alpha=pass2act(alpha,beta,gamma)
    
    or 
    
    cG,-sG,cB,-sB,cA,-sA=pass2act(cA,sA,cB,sB,cG,sG)
    """
    
    if sB is None:
        return -cB,-sA,-cA
    else:
        return cG,-sG,cB,-sB,cA,-sA

#%% Change sines and cosines to angles
def sc2angles(cA,sA=None,cB=None,sB=None,cG=None,sG=None):
    """
    Converts cosines and sines of angles to the angles themselves. Takes one or
    three cosine/sine pairs. Note, if an odd number of arguments is given (1 or 3),
    we assume that this function has been called using angles instead of cosines
    and sines, and simply return the input.
    """
    if sA is None:
        return cA
    elif cB is None:
        return np.mod(np.arctan2(sA,cA),2*np.pi)
    elif sB is None:
        return np.array([cA,sA,cB])
    else:
        return np.mod(np.array([np.arctan2(sA,cA),np.arctan2(sB,cB),np.arctan2(sG,cG)]),2*np.pi)
    
#%% Frame calculations
def getFrame(v1,v2=None,return_angles=False):
    """
    Calculates the sines and cosines of the euler angles for the principle axis
    system of a frame defined by one or two vectors. The new frame has v1 along
    its z-axis and if a second vector is provided, then the second vector lies
    in the xz plane of the frame.
    
    We use zyz convention (alpha,beta,gamma), where rotation into a frame is 
    achieved by first applying gamma:
        X,Y=cos(gamma)*X+sin(gamma)*Y,-sin(gamma)*X+cos(gamma)*Y
    Then applying beta:
        X,Z=cos(beta)*X-sin(beta)*Z,sin(beta)*X+cos(beta)*Z
    Finally alpha:
        X,Y=cos(alpha)*X+sin(alpha)*Y,-sin(alpha)*X+cos(alpha)*Y
        
    gamma=arctan2(Y,X)
    beta=arccos(Z)
    alpha=arctan(Y1,X1) (Y1 and X1 after applying gamma and beta!)
    
    Note that we do not return alpha,beta,gamma! Instead, we return 
    cos(alpha),sin(alpha),cos(beta),sin(beta),cos(gamma),sin(gamma)!
    
    If only one vector is provided, then we simply require that this 
    vector lies along z, achieved by rotating the shortest distance to the 
    z-axis. Then, the euler angles are (-gamma,beta,gamma)
    
    
    cA,sA,cB,sB,cG,sG = getFrame(v1,v2)
    
        or
        
    cG,-sG,cB,sB,cG,sG = getFrame(v1)
    
    Finally, if you need the angles themselves:
        
    alpha,beta,gamma = getFrame(v1,v2,return_angles=True)
    """
        
    if np.ndim(v1)==1:
        v1=np.atleast_2d(v1).T
        oneD=True
        if v2 is not None:
            v2=np.atleast_2d(v2).T
    else:
        oneD=False
    
    "Normalize"
    X,Y,Z=norm(v1)
    
    "Gamma"
    lenXY=np.sqrt(X**2+Y**2)
    i=lenXY==0
    lenXY[i]=1  #cG and sG will be 0 since X and Y are zero
    cG,sG=X/lenXY,Y/lenXY
    cG[i]=1. #Set cG to 1 where cG/sG is undefined (gamma=0)
    
    "Beta"
    cB,sB=Z,np.sqrt(1-Z**2)
    
    "Alpha"
    if v2 is None:
#        cA,sA=np.ones(cG.shape),np.zeros(sG.shape)
        cA,sA=cG,-sG
    else:
        v2=Rz(v2,cG,-sG)
        X,Y,_=Ry(v2,cB,-sB)
        
        lenXY=np.sqrt(X**2+Y**2)
        i=lenXY==0
        lenXY[i]=1  #cA and sA will be 0 since X and Y are zero
        cA,sA=X/lenXY,Y/lenXY
        cA[i]=1. #Now set cG to 1 where cG/sG undefined (alpha=0)
        i=np.isnan(lenXY)
        cA[i],sA[i]=cG[i],-sG[i]    #nan also gets set to -gamma
    
    if oneD:
        cA,sA,cB,sB,cG,sG=cA[0],sA[0],cB[0],sB[0],cG[0],sG[0]
        #Recently added. May need removed if errors occur 11.09.2021
    
    if return_angles:
        return sc2angles(cA,sA,cB,sB,cG,sG)
    else:
        return cA,sA,cB,sB,cG,sG


def applyFrame(*vecs,nuZ_F=None,nuXZ_F=None):
    """
    Applies a frame, F, to a set of vectors, *vecs, by rotating such that the
    vector nuZ_F lies along the z-axis, and nuXZ_F lies in the xz-plane. Input
    is the vectors (as *vecs, so list separately, don't collect in a list), and
    the frame, defined by nuZ_F (a vector on the z-axis of the frame), and 
    optionally nuXZ_F (a vector on xy-axis of the frame). These must be given
    as keyword arguments.
    
    vecs_F = applyFrame(*vecs,nuZ_F=nuZ_F,nuXZ_F=None,frame_index=None)
    
    Note, one may also omit the frame application and just apply a frame index
    """

    if nuZ_F is None:
        out=vecs
    else:
        sc=pass2act(*getFrame(nuZ_F,nuXZ_F))
        out=[None if v is None else R(v,*sc) for v in vecs]
        
    if len(vecs)==1:
        return out[0]
    else:
        return out    

#%% Apply/invert rotations     
def Rz(v0,c,s=None):
    """
    Rotates a vector around the z-axis. One must provide the vector(s) and either
    the angle itself, or the cosine(s) and sine(s) of the angle(s). The number
    of vectors must match the number of angles, or only one angle is provided
    
    v=Rz(v0,c,s)
    
        or
        
    v=Rz(v0,theta)
    """
    
    if s is None:
        c,s=np.cos(c),np.sin(c)
        
    X,Y,Z=v0.copy()
    
    X,Y=c*X-s*Y,s*X+c*Y
    Z=np.ones(X.shape)*Z
    
    return np.array([X,Y,Z])

def Ry(v0,c,s=None):
    """
    Rotates a vector around the y-axis. One must provide the vector(s) and either
    the angle itself, or the cosine(s) and sine(s) of the angle(s). The number
    of vectors must match the number of angles, or only one angle is provided
    
    v=Ry(v0,c,s)
    
        or
        
    v=Ry(v0,theta)
    """
    
    if s is None:
        c,s=np.cos(c),np.sin(c)
        
    X,Y,Z=v0.copy()
    
    X,Z=c*X+s*Z,-s*X+c*Z
    Y=np.ones(c.shape)*Y
    
    return np.array([X,Y,Z])

def R(v0,cA,sA,cB,sB=None,cG=None,sG=None):
    """
    Rotates a vector using ZYZ convention. One must provide the vector(s) and 
    either the euler angles, or the cosine(s) and sine(s) of the angle(s). The 
    number of vectors must match the number of angles, or only one angle is 
    provided for alpha,beta,gamma (or the sines/cosines of alpha,beta,gamma)
    
    v=R(v0,cA,sA,cB,sB,cG,sG)
    
        or
        
    v=R(v0,alpha,beta,gamma)
    """
    if v0 is None:
        return None
    
    if sB is None:
        cA,sA,cB,sB,cG,sG=np.cos(cA),np.sin(cA),np.cos(sA),np.sin(sA),np.cos(cB),np.sin(cB)
        
    return Rz(Ry(Rz(v0,cA,sA),cB,sB),cG,sG)

def Rfull(cA,sA,cB,sB=None,cG=None,sG=None):
    """
    Returns a ZYZ rotation matrix for one set of Euler angles
    """
    
    if sB is None:
        a=cA
        b=sA
        g=cB
        cA,sA,cB,sB,cG,sG=np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.cos(g),np.sin(g)
    
    return np.array([[cA*cB*cG-sA*sG,-cG*sA-cA*cB*sG,cA*sB],\
                [cA*sG+cB*cG*sA,cA*cG-cB*sA*sG,sA*sB],\
                [-cG*sB,sB*sG,cB]])

def euler_prod(*euler,return_angles=False):
    """
    Calculates the product of a series of euler angles. Input is a list, where
    each list element is a set of euler angles. Each set of euler angles may be
    given as a list of 3 elements (alpha,beta,gamma) or six elements 
    (ca,sa,cb,sb,cg,sg).
    
    The individual elements (alpha,beta,gamma, ca, sa, etc.) may have any size,
    although all sizes used should be the same or consistent for broadcasting
    
    ca,sa,cb,sb,cg,sg=euler_prod(euler1,euler2,...,return_angles=False)
    
        or
    
    alpha,beta,gamma=euler_prod(euler1,euler2,...,return_angles=True)
    """
    
    if len(euler)==1:   #I think this is here in case a list is provided instead of multiple inputs
        euler=euler[0]  
    
    vZ=[0,0,1]  #Reference vectors
    vX=[1,0,0]
    
    for sc in euler:
        vZ=R(vZ,*sc)
        vX=R(vX,*sc)
    
    return getFrame(vZ,vX,return_angles)
        

def Rspher(rho,cA,sA,cB,sB=None,cG=None,sG=None):
    """
    Rotates a spherical tensor, using angles alpha, beta, and
    gamma. The cosines and sines may be provided, or the angles directly.
    
    One may provide multiple rho and/or multiple angles. If a single rho vector
    is given (5,), then any shape of angles may be used, and similarly, if a single
    set of euler angles is used, then any shape of rho may be used (the first 
    dimension must always be 5). Otherwise, standard broadcasting rules apply
    (the last dimensions must match in size)
    
    rho_out = Rspher(rho,alpha,beta,gamma)
    
    or
    
    rho_out = Rspher(rho,cA,sA,cB,sB,cG,sG)- cosines and sines of the angles
    """
    

    for k,r in enumerate(rho):
        M=D2(cA,sA,cB,sB,cG,sG,mp=k-2,m=None)   #Rotate from mp=k-2 to all new components
        if k==0:
            rho_out=M*r
        else:
            rho_out+=M*r
    return rho_out    
    
    

def R2euler(R,return_angles=False):
    """
    Input a rotation matrix in cartesian coordinates, and return either the
    euler angles themselves or their cosines and sines(default)
    
    cA,sA,cB,sB,cG,sG = R2euler(R)
    
        or
    
    alpha,beta,gamma = R2euler(R,return_angles=True)
    
    R can be a list of matrices
    """
    
#    R = np.array([R]) if np.ndim(R)==2 else np.array(R)
    
    
    """
    Note that R may be the result of an eigenvector decomposition, and does
    not guarantee that R is a proper rotation matrix. We can check the sign
    on the determinant: if it is 1, it's a proper rotation, if it's -1, it's not
    Then, we just multiply each matrix by the result to have only proper
    rotations.

    """
    sgn=np.sign(np.linalg.det(R))
        
    if np.ndim(R)>2:    #Bring the dimensions of the R matrix to the first two dimensions
        for m in range(0,R.ndim-2):
            for k in range(0,R.ndim-1):R=R.swapaxes(k,k+1)
    R=R*sgn
    
    if R.ndim>2:
        cB=R[2,2]
        cB[cB>1]=1.     #Some clean-up to make sure we don't get imaginary terms later (cB cannot exceed 1- only numerical error causes this)
        cB[cB<-1]=-1.
        sB=np.sqrt(1.-cB**2)
        i,ni=sB!=0,sB==0
        cA,sA,cG,sG=np.ones(i.shape),np.zeros(i.shape),np.ones(i.shape),np.zeros(i.shape)
        cA[i]=R[2,0,i]/sB[i]    #Sign swap, 30.09.21
        sA[i]=R[2,1,i]/sB[i]
        cG[i]=-R[0,2,i]/sB[i]   #Sign swap, 30.09.21
        sG[i]=R[1,2,i]/sB[i]
        
        cG[ni]=R[0,0,ni]
        sG[ni]=-R[1,0,ni]       #Sign swap, 30.09.21
    else:
        cB=R[2,2]
        if cB>1:cB=1
        if cB<-1:cB=-1
        sB=np.sqrt(1-cB**2)
        if sB>0:
            cA=R[2,0]/sB        #Sign swap, 30.09.21
            sA=R[2,1]/sB
            cG=-R[0,2]/sB       #Sign swap, 30.09.21
            sG=R[1,2]/sB
        else:
            cA,sA=1,0
            cG=R[0,0]
            sG=-R[1,0]          #Sign swap, 30.09.21

    
    if return_angles:
        return sc2angles(cA,sA,cB,sB,cG,sG)
    else:
        return np.array((cA,sA,cB,sB,cG,sG))
    
def R2vec(R):
    """
    Given a rotation matrix, R, this function returns two vectors, v1, and v2
    that have been rotated from v10=[0,0,1] and v20=[1,0,0]
    
    v1=np.dot(R,v10)
    v2=np.dot(R,v20)
    
    If a frame is defined by a rotation matrix, instead of directly by a set of
    vectors, then v1 and v2 have the same Euler angles to rotate back to their
    PAS as the rotation matrix
    
    R may be a list of rotation matrices
    
    Note: v1, v2 are trivially given by R[:,:,2] and R[:,:,0]
    """
    R = np.array([R]) if np.ndim(R)==2 else np.array(R)
    
    v1=R[:,:,2]
    v2=R[:,:,0]
    
    return v1.T,v2.T
    
    
#%% Tensor operations
def d2(c=0,s=None,m=None,mp=0):
    """
    Calculates components of the d2 matrix. By default only calculates the components
    starting at m=0 and returns five components, from -2,-1,0,1,2. One may also
    edit the starting component and select a specific final component 
    (mp=None returns all components, whereas mp may be specified between -2 and 2)
    
    d2_m_mp=d2(m,mp,c,s)  #c and s are the cosine and sine of the desired beta angle
    
        or
        
    d2_m_mp=d2(m,mp,beta) #Give the angle directly
    
    Setting mp to None will return all values for mp in a 2D array
    
    (Note that m is the final index)
    """
    
    if s is None:
        c,s=np.cos(c),np.sin(c)
    
    """
    Here we define each of the components as functions. We'll collect these into
    an array, and then call them out with the m and mp indices
    """
    "First, for m=-2"
    
    if m is None or mp is None:
        if m is None and mp is None:
            print('m or mp must be specified')
            return
        elif m is None:
            if mp==-2:
                index=range(0,5)
            elif mp==-1:
                index=range(5,10)
            elif mp==0:
                index=range(10,15)
            elif mp==1:
                index=range(15,20)
            elif mp==2:
                index=range(20,25)
        elif mp is None:
            if m==-2:
                index=range(0,25,5)
            elif m==-1:
                index=range(1,25,5)
            elif m==0:
                index=range(2,25,5)
            elif m==1:
                index=range(3,25,5)
            elif m==2:
                index=range(4,25,5)
    else:
        index=[(mp+2)*5+(m+2)]
    
    out=list()    
    for i in index:
        #mp=-2
        if i==0:x=0.25*(1+c)**2
        if i==1:x=0.5*(1+c)*s
        if i==2:x=np.sqrt(3/8)*s**2
        if i==3:x=0.5*(1-c)*s
        if i==4:x=0.25*(1-c)**2
        #mp=-1
        if i==5:x=-0.5*(1+c)*s
        if i==6:x=c**2-0.5*(1-c)
        if i==7:x=np.sqrt(3/8)*2*c*s
        if i==8:x=0.5*(1+c)-c**2
        if i==9:x=0.5*(1-c)*s
        #mp=0
        if i==10:x=np.sqrt(3/8)*s**2
        if i==11:x=-np.sqrt(3/8)*2*s*c
        if i==12:x=0.5*(3*c**2-1)
        if i==13:x=np.sqrt(3/8)*2*s*c
        if i==14:x=np.sqrt(3/8)*s**2
        #mp=1
        if i==15:x=-0.5*(1-c)*s
        if i==16:x=0.5*(1+c)-c**2
        if i==17:x=-np.sqrt(3/8)*2*s*c
        if i==18:x=c**2-0.5*(1-c)
        if i==19:x=0.5*(1+c)*s
        #mp=2
        if i==20:x=0.25*(1-c)**2
        if i==21:x=-0.5*(1-c)*s
        if i==22:x=np.sqrt(3/8)*s**2
        if i==23:x=-0.5*(1+c)*s
        if i==24:x=0.25*(1+c)**2
        out.append(x)
        
    if m is None or mp is None:
        return np.array(out)
    else:
        return out[0]

def D2(cA=0,sA=0,cB=0,sB=None,cG=None,sG=None,m=None,mp=0):
    """
    Calculates components of the Wigner rotation matrix from Euler angles or
    from the list of sines and cosines of those euler angles. All vectors must
    be the same size (or have only a single element)
    
    mp and m should be specified. m may be set to None, so that all components
    are returned in a 2D array
    
    D2_m_mp=D2(m,mp,cA,sA,cB,sB,cG,sG)  #Provide sines and cosines
    
        or
        
    D2_m_mp=D2(m,mp,alpha,beta,gamma) #Give the angles directly
    
    (Note that m is the final index)
    """
    if sB is None:
        cA,sA,cB,sB,cG,sG=np.cos(cA),np.sin(cA),np.cos(sA),np.sin(sA),np.cos(cB),np.sin(cB)

        
    d2c=d2(cB,sB,m,mp)
    
    "Rotation around z with alpha (mp)"
    if mp is None:
        ea1=cA-1j*sA
        eam1=cA+1j*sA
        ea2=ea1**2
        eam2=eam1**2
        ea0=np.ones(ea1.shape)
        ea=np.array([eam2,eam1,ea0,ea1,ea2])
    else:
        if mp!=0:
            ea=cA-1j*np.sign(mp)*sA
            if np.abs(mp)==2:
                ea=ea**2
        else:
            ea=1

    "Rotation around z with gamma (m)"
    if m is None:
        eg1=cG-1j*sG
        egm1=cG+1j*sG
        eg2=eg1**2
        egm2=egm1**2
        eg0=np.ones(eg1.shape)
        eg=np.array([egm2,egm1,eg0,eg1,eg2])
    else:
        if m!=0:
            eg=cG-1j*np.sign(m)*sG
            if np.abs(m)==2:
                eg=eg**2
        else:
            eg=1
            
    return ea*d2c*eg
    

def D2vec(v1,v2=None,m=None,mp=0):
    """
    Calculates the Wigner rotation elements that bring a vector or vectors from
    their own principle axis system into a reference frame (whichever frame
    v1 and v2 are defined in)
    """
    
    cA,sA,cB,sB,cG,sG=getFrame(v1,v2)
    "I think these are already the passive angles above"
    
    return D2(cA,sA,cB,sB,cG,sG,m,mp)

def getD2inf(v,n=2500):
    """
    Calculates the expectation value of the Spherical components of the D2 rotation
    elements, that is
    lim t->oo <D2_0p(Omega_{t+tau,t})>_tau
    
    These are estimated given a vector v. Note, we are always performing averaging
    from the PAS of a vector into a given frame. Then, there should never be 
    a contribution from asymmetry (arguably, we could correct for eta in case
    of CSA or quadrupolar relaxation, but we won't do that here)
    
    n specifies the maximum number of time points to take from a vector. Default
    is 500, although setting to None will set N=v.shape[-1]
    """ 
    
    if n is None or v.shape[-1]<n:
        n=v.shape[-1]
        
    step=np.array(v.shape[-1]/n,dtype=int)
    index=np.arange(0,step*n,step,dtype=int)
    
    x0,y0,z0=norm(v[:,:,index])
    
#    D2avg=list()
    
#    for m in range(-2,3):
#        D2avg.append([D2inf(x,y,z,m) for x,y,z in zip(x0,y0,z0)])
    D2avg=[D2inf(x,y,z) for x,y,z in zip(x0,y0,z0)]
        
    return np.array(D2avg).T

def D2inf(x,y,z,m=None):
    """
    Calculates spherical component expectation value for D2 rotation matrix 
    elements (for a single bond)
    lim t->oo <D2_0p(Omega_{t+tau,t})>_tau
    
    Provide normalized x,y,z and the desired component
    """

    if m==0:
        D2avg=-1/2
        for alpha in [x,y,z]:
            for beta in [x,y,z]:
                D2avg+=3/2*((alpha*beta).mean())**2
        return D2avg
    
    "Beta"
    cb,sb=z,np.sqrt(1-z**2)
    "Gamma"
    lenXY=np.sqrt(x**2+y**2)
    i=lenXY==0
    lenXY[i]=1  #cG and sG will be 0 since x and y are both zero
    cg,sg=x/lenXY,y/lenXY
    cg[i]=1. #Set cG to 1 where cG/sG is undefined (set gamma=0)

    n=x.shape[0]
    
    D2avg=np.zeros(5,dtype=complex)
    if m is None:   #Get all components
        m1=[-2,-1,0,1,2]
    else:
        m1=[m]
        
    
    for cb0,sb0,cg0,sg0 in zip(cb,sb,cg,sg):
        x1,y1,z1=x*cg0+y*sg0,-x*sg0+y*cg0,z
        x2,y2,z2=x1*cb0-z1*sb0,y1,x1*sb0+z1*cb0     #vectors in frame of current element
        
        for m0 in m1:
            if m0==-2:
                D2avg[0]+=np.sqrt(3/8)*((x2+1j*y2)**2).mean()
            elif m0==-1:
                D2avg[1]+=-np.sqrt(3/2)*((x2+1j*y2)*z2).mean()
            elif m0==1:
                if m is not None:   #Don't repeat this calculation if None
                    D2avg[3]+=np.sqrt(3/2)*((x2-1j*y2)*z2).mean()
            elif m0==2:
                if m is not None:   #Same as above
                    D2avg[4]+=np.sqrt(3/8)*((x2-1j*y2)**2).mean()
                    

    if m is not None:
        D2avg=D2avg[m+2]/n
    else:
        for k in range(2):
            D2avg[k]=D2avg[k]/n
        D2avg[3]=-np.conjugate(D2avg[1])
        D2avg[4]=np.conjugate(D2avg[0])

        d2=-1/2
        for alpha in [x,y,z]:
            for beta in [x,y,z]:
                d2+=3/2*((alpha*beta).mean())**2
        D2avg[2]=d2
       
    return D2avg   
    
#    x1,y1,z1=np.dot(np.array([x]).T,np.array([cg]))+np.dot(np.array([y]).T,np.array([sg])),\
#                -np.dot(np.array([x]).T,np.array([sg]))+np.dot(np.array([y]).T,np.array([cg])),np.repeat(np.array([z]).T,z.size,axis=1)
#    x2,y2,z2=x1*cb-z1*sb,y1,+x1*sb+z1*cb
#    

    
#    if m==-2:
#        return np.sqrt(3/8)*((x2+1j*y2)**2).mean()
#    elif m==-1:
#        return -np.sqrt(3/2)*((x2+1j*y2)*z2).mean()
#    elif m==1:
#        return np.sqrt(3/2)*((x2-1j*y2)*z2).mean()
#    elif m==2:
#        return np.sqrt(3/8)*((x2-1j*y2)**2).mean()

def D2inf_v2(vZ,m=None):
    if m is None:
        m1=[-2,-1,0]
    else:
        m1=[m]
    
    if m!=0:
        sc=getFrame(vZ)
        vX=R([1,0,0],*sc)
        vY=R([0,1,0],*sc)
    
    
    if vZ.ndim==2:
        N=0
    else:
        N=vZ.shape[1]

    D2inf=list()
    
    for m0 in m1:
        if N==0:
            d2=np.array(0,dtype=complex)
        else:
            d2=np.zeros(N,dtype=complex)
            
        if m0==-2:
            for ax,ay,az in zip(vX,vY,vZ):
                for bx,by,bz in zip(vX,vY,vZ):
                    d2+=np.sqrt(3/8)*((ax*bx).mean(axis=-1)-(ay*by).mean(axis=-1))*(az*bz).mean(axis=-1)\
                        +1j*np.sqrt(3/2)*(ax*by).mean(axis=-1)*(az*bz).mean(axis=-1)
        elif m0==-1:
            for ax,ay,az in zip(vX,vY,vZ):
                for bz in vZ:
                    d2+=-np.sqrt(3/2)*(ax*bz).mean(axis=-1)*(az*bz).mean(axis=-1)\
                        +1j*np.sqrt(3/2)*(ay*bz).mean(axis=-1)*(az*bz).mean(axis=-1)
        elif m0==0:
            d2+=-1/2
            for az in vZ:
                for bz in vZ:
                    d2+=3/2*(az*bz).mean(axis=-1)**2
        elif m0==1:
            for ax,ay,az in zip(vX,vY,vZ):
                for bz in vZ:
                    d2+=np.sqrt(3/2)*(ax*bz).mean(axis=-1)*(az*bz).mean(axis=-1)\
                        +1j*np.sqrt(3/2)*(ay*bz).mean(axis=-1)*(az*bz).mean(axis=-1)
        elif m0==2:
            for ax,ay,az in zip(vX,vY,vZ):
                for bx,by,bz in zip(vX,vY,vZ):
                    d2+=np.sqrt(3/8)*((ax*bx).mean(axis=-1)-(ay*by).mean(axis=-1))*(az*bz).mean(axis=-1)\
                        -1j*np.sqrt(3/2)*(ax*by).mean(axis=-1)*(az*bz).mean(axis=-1)
        D2inf.append(d2)
        
    if m is None:
        D2inf.append(-np.conjugate(D2inf[1]))
        D2inf.append(np.conjugate(D2inf[0]))
    else:
        D2inf=D2inf[0]
        
    return np.array(D2inf)

def D2avgLF(vZ,m=None):
    """
    """
    if m is None:
        m1=[-2,-1,0]
    else:
        m1=[m]
        
    ca,sa,cb,sb,cg,sg=getFrame(vZ)
    
        
#    if vZ.ndim==2:
#        N=0
#    else:
#        N=vZ.shape[1]
    
    D2avg=list()
    
    for m0 in m1:
#        if N==0:
#            d2=np.array(0,dtype=complex)
#        else:
#            d2=np.zeros(N,dtype=complex)
            
        if m0==-2:
            d2=np.sqrt(3/8)*((cg*sb)**2-(sg*sb)**2+2*1j*sg*sb*cg*sb).mean(-1)
        elif m0==-1:
            d2=-np.sqrt(3/2)*(cg*sb*cb+1j*sg*sb*cb).mean(-1)
        elif m0==0:
            d2=1/2*(3*cb**2-1).mean(-1)
        elif m0==1:
            d2=np.sqrt(3/2)*(cg*sb*cb-1j*sg*sb*cb).mean(-1)
        elif m0==2:
            d2=np.sqrt(3/8)*((cg*sb)**2-(sg*sb)**2-2*1j*sg*sb*cg*sb).mean(-1)
        D2avg.append(d2)
    
    if m is None:
        D2avg.append(-np.conjugate(D2avg[1]))
        D2avg.append(np.conjugate(D2avg[0]))
    else:
        D2avg=D2avg[0]

    return np.array(D2avg)

def Spher2Cart(rho):
    """
    Takes a set of components of a spherical tensor and calculates its cartesian
    representation (as a vector, with components in order of Axx,Axy,Axz,Ayy,Ayz)
    
    Input may be a list (or 2D array), with each new column a new tensor
    """
    
    rho=np.array(rho,dtype=complex)

    M=np.array([[0.5,0,-np.sqrt(1/6),0,0.5],
                 [0.5*1j,0,0,0,-0.5*1j],
                 [0,0.5,0,-0.5,0],
                 [-0.5,0,-np.sqrt(1/6),0,-.5],
                 [0,.5*1j,0,.5*1j,0]])
    SZ0=rho.shape
    SZ=[5,np.prod(SZ0[1:]).astype(int)]
    out=np.dot(M,rho.reshape(SZ)).real
    return out.reshape(SZ0)
    
    
def Spher2pars(rho,return_angles=False):
    """
    Takes a set of components of a spherical tensor and calculates the parameters
    describing that tensor (delta,eta,alpha,beta,gamma)
    
    
    delta,eta,cA,sA,cB,sB,cG,sG=Spher2pars(rho)
    
        or
        
    delta,eta,alpha,beta,gamma=Spher2pars(rho,return_angles=True)
    
    
    Input may be a list (or 2D array), with each new column a new tensor (5xN)
    """

    A0=Spher2Cart(rho)  #Get the Cartesian tensor
    if A0.ndim==1:
        A0=np.atleast_2d(A0).T

    R=list()
    delta=list()
    eta=list()
    
    
    for k,x in enumerate(A0.T):
        Axx,Axy,Axz,Ayy,Ayz=x
        A=np.array([[Axx,Axy,Axz],[Axy,Ayy,Ayz],[Axz,Ayz,-Axx-Ayy]])    #Full matrix
        D,V=np.linalg.eigh(A)   #Get eigenvalues, eigenvectors 
        i=np.argsort(np.abs(D))
        D,V=D[i[[1,0,2]]],V[:,i[[1,0,2]]]     #Ordering is |azz|>=|axx|>=|ayy|
        "V should have a determinant of +1 (proper vs. improper rotation)"
        V=V*np.sign(np.linalg.det(V))
        delta.append(D[2])
        eta.append((D[1]-D[0])/D[2])
        R.append(V)
    
    delta=np.array(delta)
    eta=np.array(eta)
    euler=R2euler(np.array(R))
    
    if return_angles:
        euler=sc2angles(*euler)
       
    return np.concatenate(([delta],[eta],euler),axis=0)
        

def pars2Spher(delta,eta=None,cA=None,sA=None,cB=None,sB=None,cG=None,sG=None):
    """
    Converts parameters describing a spherical tensor (delta, eta, alpha, beta,
    gamma) into the tensor itself. All arguments except delta are optional. Angles
    may be provided, or their cosines and sines may be provided. The size of the
    elements should follow the rules required for Rspher.
    """

    if cA is None:
        cA,sA,cB,sB,cG,sG=np.array([1,0,1,0,1,0])
    
    if eta is None:
        eta=np.zeros(np.shape(delta))
    
    rho0=np.array([-0.5*eta*delta,0,np.sqrt(3/2)*delta,0,-0.5*eta*delta])
    
    return Rspher(rho0,cA,sA,cB,sB,cG,sG)

        
        
#%% RMS alignment
def RMSalign(v0,vref):
    """
    Returns the optimal rotation matrix to rotate a set of vectors v0 to a set 
    of reference vectors, vref
    
    R=alignR(v0,vref)
    
    Uses the Kabsch algorithm. Assumes *vectors*, with origins at zero, not 
    points, so that no translation will be performed
    (reference https://en.wikipedia.org/wiki/Kabsch_algorithm)
    
    We minimize
    np.sum((np.dot(R,v0.T).T-vref)**2)
    """
    
    H=np.dot(v0,vref.T)
    
    U,S,Vt=svd(H)
    V=Vt.T
    Ut=U.T
    
    d=np.linalg.det(np.dot(V,Ut))
    m=np.eye(3)
    m[2,2]=d    #This is supposed to ensure a right-handed coordinate system
                #But I think it could equivalently be thought of as making this a proper rotation(??)
    
    R=np.dot(V,np.dot(m,Ut))
    return R


#%% Fit points to a plane
def RMSplane(v,weight=None):
    """
    For a set of points (v: 3xN array), calculates the normal vector for a plane
    fitted to that set of points. May include a weighting (weight: N elements)
    """
    # v=np.array(norm(v))
    
    "Default, uniform weighting"
    if weight is None:
        weight=np.ones(v.shape[1])
        
    "Subtract away the centroid"
    v=(v.T-v.mean(axis=1)).T
    
    """Applying weighting, taking singular value decomposition, return
    row of U corresponding to the smallest(last) singular value"""
    return svd(v*weight)[0].T[2]

#%% Get principle axes of moment of inertia
def principle_axis_MOI(v):
    """
    Calculates the principle axis system of the moment of inertia, without
    considering weights of individual particles. A 3xN numpy array should be 
    provided. The smallest component of the moment of inertia is returned in the
    0 element, and largest in the 2 element
    
    Note- the directions of the principle axes can switch directions (180 deg)
    between frames, due to the symmetry of the MOI tensor. This can be corrected
    for with a reference vector. The dot product of the reference vector and the
    vector for a given frame should remain positive. If it doesn't, then switch
    the direction of the vector (v=v*np.sign(np.dot(v.T,v)))
    """
    
    """
    Ixx=sum_i m_i*(y_i^2+z_i^2)
    Iyy=sum_i m_i*(x_i^2+z_i^2)
    Izz=sum_i m_i*(x_i^2+y_i^2)
    Ixy=Iyx=-sum_i m_i*x_i*y_i
    Ixz=Izx=-sum_i m_i*x_i*z_i
    Iyz=Izy=-sum_i m_i*y_i*z_i
    """
    
    
    v=v-np.atleast_2d(v.mean(axis=1)).T.repeat(v.shape[1],axis=1) #v after subtracting center of mass
    
    H=np.dot(v,v.T)
    
    I=-1*H
    I[0,0]=H[1,1]+H[2,2]
    I[1,1]=H[0,0]+H[2,2]
    I[2,2]=H[1,1]+H[0,0]
    _,V=np.linalg.eigh(I)
    
    return V

#%% Project onto axis
def projZ(v0,vr=[0,0,1]):
    """
    Takes the projection of a vector, v0, onto another vector, vr.
    
    Input should be 3xN vectors (vnorm can also be a 1D, 3 element vector, or
    both inputs can be 3 element vectors).
    
    Input does not need to be normalized, but also note that output is not 
    normalized
    
    Default project is along z
    """
#    v0=np.atleast_2d(v0)
#    if np.ndim(vr)==1 or np.shape(vr)[1]==1:    #Make matrices the same size
#        vr=np.atleast_2d(vr).T.repeat(np.shape(v0)[1],axis=1) 
    
    vr=norm(vr)
    a=v0[0]*vr[0]+v0[1]*vr[1]+v0[2]*vr[2]
    return np.array([a*vr[0],a*vr[1],a*vr[2]])
    
#    return np.atleast_2d((norm(v0)*norm(vr)).sum(axis=0))*vr

#%% Project onto plane
def projXY(v0,vnorm=[0,0,1]):
    """
    Takes the projection of a vector, v0, onto a plane defined by its normal 
    vector, vnorm. 
    
    Input should be 3xN vectors (vnorm can also be a 1D, 3 element vector, or
    both inputs can be 3 element vectors).
    
    Input does not need to be normalized, but also note that output is not 
    normalized
    """
    
    return v0-projZ(v0,vnorm)  
    
#%% Sort by distance
def sort_by_dist(v,maxd=1e4):
    """
    Returns an index that sorts a set of points such that each point is next
    to its nearest neighbors in space in the vector itself (although points will
    not repeat, so this has exceptions)
    
    Searchest for the point closest to (-Inf,-Inf,-Inf), then looks for its nearest
    neighbor, and then searchest for the nearest neighhbor of the next point
    (etc...)
    
    The purpose is that we can take a set of points, and take the difference in
    position of each one to generate a set of vectors (which may be subsequently
    corrected for periodic boundary conditions)
    
    Returns the sorting index, as oppposed to the vector itself
    
    i=sort_by_dist(v)
    
    such that v_sorted=v[i]
    
    Note 1: not a highly efficient algorithm- recommended only for setup of a 
    vector calculation, but should avoided inside loop over a trajectory

    Note 2: We presume here that the dimensions can't be larger than 1e4, rather
    than assuming the dimension is arbitrarily large (creating some numeric
    problems). If this is not true, set maxd to an appropriate value
    """
    
    v=v.copy()  #Avoid editing the vector itself...never quite sure when this is necessary
    X,Y,Z=v.T
    
    ref=maxd*2
    
    i=list()
    "Find the most negative element"
    i.append(np.argmin((X+ref)**2+(Y+ref)**2+(Z+ref)**2))
    
    for _ in range(X.size-1):
        x,y,z=X[i[-1]].copy(),Y[i[-1]].copy(),Z[i[-1]].copy()
        "Set the currect vector to be far away"
        X[i[-1]],Y[i[-1]],Z[i[-1]]=ref*np.array([-2,-2,-2])
        "Find the nearest vector"
        i.append(np.argmin((X-x)**2+(Y-y)**2+(Z-z)**2))
        
    return i
    
    
    
    
    