#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:38:52 2022

@author: albertsmith
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

def plot_rho(lbl,R,R_std=None,style='plot',color=None,ax=None,split=True,**kwargs):
    """
    Plots a set of rates or detector responses. 
    """
    
    if ax is None:
        ax=plt.figure().add_subplot(111)
    
    "We divide the x-axis up where there are gaps between the indices"
    lbl1=list()
    R1=list()
    R_u1=list()
    R_l1=list()
    
    lbl=np.array(lbl)   #Make sure this is a np array
    if not(np.issubdtype(lbl.dtype,np.number)):
        split=False
        lbl0=lbl.copy()
        lbl=np.arange(len(lbl0))
    else:
        lbl0=None
    
    if split:
        s0=np.where(np.concatenate(([True],np.diff(lbl)>1,[True])))[0]
    else:
        s0=np.array([0,np.size(R)])
    
    for s1,s2 in zip(s0[:-1],s0[1:]):
        lbl1.append(lbl[s1:s2])
        R1.append(R[s1:s2])
        if R_std is not None:
            if np.ndim(R_std)==2:
                R_l1.append(R_std[0][s1:s2])
                R_u1.append(R_std[1][s1:s2])
            else:
                R_l1.append(R_std[s1:s2])
                R_u1.append(R_std[s1:s2])
        else:
            R_l1.append(None)
            R_u1.append(None)
    
    "Plotting style (plot,bar, or scatter, scatter turns the linestyle to '' and adds a marker)"
    if style.lower()[0]=='s':
        if 'marker' not in kwargs:
            kwargs['marker']='o'
        if 'linestyle' not in kwargs:
            kwargs['linestyle']=''
        ebar_clr=color
    elif style.lower()[0]=='b':
        if 'linestyle' not in kwargs:
            kwargs['linestyle']=''
        ebar_clr='black'
    else:
        ebar_clr=color
    
    for lbl,R,R_u,R_l in zip(lbl1,R1,R_u1,R_l1):
        if R_l is None:
            ax.plot(lbl,R,color=color,**kwargs)
        else:
            ax.errorbar(lbl,R,[R_l,R_u],color=ebar_clr,capsize=3,**kwargs)
        if style.lower()[0]=='b':
            kw=kwargs.copy()
            if 'linestyle' in kw: kw.pop('linestyle')
            ax.bar(lbl,R,color=color,**kw)
        if color is None:
            color=ax.get_children()[0].get_color()
    
    if lbl0 is not None:
        ax.set_xticks(lbl)
        ax.set_xticklabels(lbl0,rotation=90)
                
    return ax


def plot_fit(lbl,Rin,Rc,Rin_std=None,info=None,index=None,exp_index=None,fig=None):
    """
    Plots the fit of experimental data (small data sizes- not MD correlation functions)
    Required inputs are the data label, experimental rates, fitted rates. One may
    also input the standard deviation of the experimental data, and the info
    structure from the experimental data.
    
    Indices may be provided to specify which residues to plot, and which 
    experiments to plot
    
    A figure handle may be provided to specifiy the figure (subplots will be
    created), or a list of axis handles may be input, although this must match
    the number of experiments
    
    plot_fit(lbl,Rin,Rc,Rin_std=None,info=None,index=None,exp_index=None,fig=None,ax=None)
    
    one may replace Rin_std with R_l and R_u, to have different upper and lower bounds
    """
    
    "Apply index to all data"
    if index is not None:
        lbl=lbl[index]
        Rin=Rin[index]
        Rc=Rc[index]
        if Rin_std is not None: Rin_std=Rin_std[index]
        
    "Remove experiments if requested"
    if exp_index is not None:
        if info is not None: 
            info=info.loc[:,exp_index].copy
            info.columns=range(Rin.shape[0])
            
        Rin=Rin[:,exp_index]
        Rc=Rc[:,exp_index]
        if Rin_std is not None: Rin_std=Rin_std[:,exp_index]
    
    nexp=Rin.shape[1]    #Number of experiments
    
    ax,xax,yax=subplot_setup(nexp,fig)
    SZ=np.array([np.sum(xax),np.sum(yax)])
    #Make sure the labels are set up
    """Make lbl a numpy array. If label is already numeric, then we use it as is.
    If it is text, then we replace lbl with a numeric array, and store the 
    original lbl as lbl0, which we'll label the x-axis with.
    """
    lbl=np.array(lbl)   #Make sure this is a np array
    if not(np.issubdtype(lbl.dtype,np.number)):
        split=False
        lbl0=lbl.copy()
        lbl=np.arange(len(lbl0))

                    
    else:
        lbl0=None
    
    "Use truncated labels if too many residues"
    if lbl0 is not None and len(lbl0)>50/SZ[0]:  #Let's say we can fit 50 labels in one figure
        nlbl=np.floor(50/SZ[0])
        space=np.floor(len(lbl0)/nlbl).astype(int)
        ii=range(0,len(lbl0),space)
    else:
        ii=range(0,len(lbl))

    #Sweep through each experiment
    clr=[k for k in colors.TABLEAU_COLORS.values()]     #Color table
    for k,a in enumerate(ax):
        a.bar(lbl,Rin[:,k],color=clr[np.mod(k,len(clr))])       #Bar plot of experimental data
        if Rin_std is not None:             
            a.errorbar(lbl,Rin[:,k],Rin_std[:,k],color='black',linestyle='',\
                       capsize=3) #Errorbar
        a.plot(lbl,Rc[:,k],linestyle='',marker='o',color='black',markersize=3)
        if xax[k]:
            if lbl0 is not None:
                a.set_xticks(ii)
                a.set_xticklabels(lbl0[ii],rotation=90)
        else:
            plt.setp(a.get_xticklabels(),visible=False)
            if lbl0 is not None:
                a.set_xticks(ii)
        if yax[k]:
            a.set_ylabel(r'R / s$^{-1}$')
        
        #Apply labels to each plot if we find experiment type in the info array
        if info is not None and 'Type' in info.keys:
            if info[k]['Type'] in ['R1','NOE','R2']:
                a.set_ylim(np.min(np.concatenate(([0],Rin[:,k],Rc[:,k]))),\
                   np.max(np.concatenate((Rin[:,k],Rc[:,k])))*1.25)
                i=info[k]
                string=r'{0} {1}@{2:.0f} MHz'.format(i['Nuc'],i['Type'],i['v0'])
                a.text(np.min(lbl),a.get_ylim()[1]*0.88,string,fontsize=8)
            elif info[k]['Type']=='S2':
                a.set_ylim(np.min(np.concatenate(([0],Rin[:,k],Rc[:,k]))),\
                   np.max(np.concatenate((Rin[:,k],Rc[:,k])))*1.25)
                string=r'$1-S^2$'
                a.text(np.min(lbl),a.get_ylim()[1]*0.88,string,fontsize=8)
            else:
                a.set_ylim(np.min(np.concatenate(([0],Rin[:,k],Rc[:,k]))),\
                   np.max(np.concatenate((Rin[:,k],Rc[:,k])))*1.45)
                i=info[k]
                string=r'{0} {1}@{2:.0f} MHz'.format(i['Nuc'],i['Type'],i['v0'])
                a.text(np.min(lbl),a.get_ylim()[1]*0.88,string,fontsize=8)
                string=r'$\nu_r$={0} kHz, $\nu_1$={1} kHz'.format(i['vr'],i['v1'])
                a.text(np.min(lbl),a.get_ylim()[1]*0.73,string,fontsize=8)
#    fig.show()            
    return ax 

def subplot_setup(nexp,fig=None):
    """
    Creates subplots neatly distributed on a figure for a given number of 
    experments. Returns a list of axes, and two logical indices, xax and yax, 
    which specify whether the figure sits on the bottom of the figure (xax) or
    to the left side of the figure (yax)
    
    Also creates the figure if none provided.
    
    subplot_setup(nexp,fig=None)
    """
    if fig is None:fig=plt.figure()
    
    "How many subplots"
    SZ=np.sqrt(nexp)
    SZ=[np.ceil(SZ).astype(int),np.floor(SZ).astype(int)]
    if np.prod(SZ)<nexp: SZ[1]+=1
    ntop=np.mod(nexp,SZ[1]) #Number of plots to put in the top row    
    if ntop==0:ntop=SZ[1]     
    
    ax=[fig.add_subplot(SZ[0],SZ[1],k+1) for k in range(ntop)]  #Top row plots
    ax.extend([fig.add_subplot(SZ[0],SZ[1],k+1+SZ[1]) for k in range(nexp-ntop)]) #Other plots
    
    xax=np.zeros(nexp,dtype=bool)
    xax[-SZ[1]:]=True
    yax=np.zeros(nexp,dtype=bool)
    yax[-SZ[1]::-SZ[1]]=True
    yax[0]=True
  
    return ax,xax,yax