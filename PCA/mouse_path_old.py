#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:46:13 2022

@author: albertsmith
"""

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from MDAnalysis import Writer
from pyDR.misc.tools import linear_ex
from scipy.signal import find_peaks

class Path2Energy():
    def __init__(self,pca):
        """
        Allows one to plot traces through one or more PCA histogram plots in
        order to define a reaction coordinate. From this coordinate, we can
        create both a histogram and corresponding energy landscape, but also
        create a trajectory that can be shown in chimeraX to see how the 
        system traverses this landscape

        Parameters
        ----------
        pca : pca object
            Object containing the principal component amplitudes.

        Returns
        -------
        None.

        """
        
        self.pca=pca     #Input pca
        self.axes=None #List of axes with principal components
        self.lines=None #List of line handles for each set of axes
        self.hists=None #List of the histograms
        self.start=None #List of start lines
        self.stop=None  #List of stop lines
        self.position=None #Line that indicates the current axis positions
        self.npairs=None     #List of principal component pairs to plot
        self.n=None #list of principal componets
        self.inflections=None #List where inflections occur
        
        self.rx_coord=None  #Calculated reaction coordinate (linear ax)
        self.PC=None        #Calculated value of the principal componts as a function of rx_coord
        
        self.cid=None       #ID for the
        
        
    def create_plots(self,nmax:int=1,npairs:list=None,axes:list=None,**kwargs):
        """
        Creates the 2D histograms for selecting the pathway to plot

        Parameters
        ----------
        nmax : int, optional
            Maximum principal component to plot. The default is 1.
        npairs : list, optional
            List of pairs of principal components for which to plot 2D 
            histograms. The default is None.
        axes : list, optional
            List of axes on which to plot. The default is None.
        **kwargs : TYPE
            Keyword arguments to be passed to the pca object's plot function.

        Returns
        -------
        axes 
            List of the axes of the plots

        """
        
        if npairs is None:npairs=[[n,n+1] for n in range(nmax)]
        self.npairs=npairs
        
        if axes is None:
            fig=plt.figure()
            N=len(npairs)
            nplots=np.array([np.ceil(np.sqrt(N)).astype(int)-1,np.ceil(np.sqrt(N)).astype(int)])
            if nplots.prod()<N:nplots[0]+=1
            self.axes=[fig.add_subplot(nplots[0],nplots[1],k+1) for k in range(N)]
        else:
            assert len(axes)==len(self.npairs),f"{len(self.npairs)} axes should be provided"
            self.axes=axes
            
        for a,(n0,n1) in zip(self.axes,self.npairs):
            self.pca.plot(n0=n0,n1=n1,ax=a,**kwargs)    
        
        self.lines=[a.plot(np.array([]),np.array([]),color='red',linewidth=1,visible=False)[0] for a in self.axes]
        self.start=[[a.plot([0,0],a.get_ylim(),linestyle='-',color='grey',visible=False)[0],
                     a.plot(a.get_xlim(),[0,0],linestyle='-',color='grey',visible=False)[0]] for a in self.axes]
        self.stop=[[a.plot([0,0],a.get_ylim(),linestyle=':',color='grey',visible=False)[0],
                     a.plot(a.get_xlim(),[0,0],linestyle=':',color='grey',visible=False)[0]] for a in self.axes]
        self.position=[[a.plot([0,0],a.get_ylim(),linestyle='--',color='blue',visible=False)[0],
                     a.plot(a.get_xlim(),[0,0],linestyle='--',color='blue',visible=False)[0]] for a in self.axes]
        
        
        def on_move(event):
            """
            Function to run when the mouse moves within one of the axes. 
            """
            if event.inaxes is None:
                # plt.disconnect(self.cid)
                # for pos in self.position:
                #     pos.set_visible(False)
                    
                # plt.draw()
                on_release(event)
            else:
                i=self.axes.index(event.inaxes)
                self.lines[i].set_xdata(np.concatenate((self.lines[i].get_xdata(),[event.xdata])))
                self.lines[i].set_ydata(np.concatenate((self.lines[i].get_ydata(),[event.ydata])))
                
                for pos,npair in zip(self.position,self.npairs):
                    if npair[0] in self.npairs[i]:
                        pos[0].set_xdata(np.ones(2)*(event.xdata if npair[0]==self.npairs[i][0] else event.ydata))
                        pos[0].set_visible(True)
                    if npair[1] in self.npairs[i]:
                        pos[1].set_ydata(np.ones(2)*(event.xdata if npair[1]==self.npairs[i][0] else event.ydata))
                        pos[1].set_visible(True)
                
                plt.draw()
                
        def on_click(event):
            """
            Initializes drawing of a path on an axis
            """
            
            if event.inaxes is None:
                return

            i=self.axes.index(event.inaxes)      
            self.lines[i].set_xdata(np.array([]))
            self.lines[i].set_ydata(np.array([]))
            
            n0,n1=self.npairs[i]
            for start,stop,npair in zip(self.start[i+1:],self.stop[i+1:],self.npairs[i+1:]):
                if n0 in npair or n1 in npair:
                    n=n1 in npair
                    i=npair.index(n)
                    for s in [start,stop]:s[i].set_visible(False)
                
                
            
            if event.button==3:
                self.lines[i].set_visible(False)
                plt.draw()
            elif event.button==1:
                self.lines[i].set_visible(True)
                self.cid=plt.connect('motion_notify_event',on_move)
                plt.draw()
                
        def on_release(event):
            if event.button==3:return
            plt.disconnect(self.cid)
            self.cid=None
            
            for pos in self.position:
                for p in pos:p.set_visible(False)
            
            plt.draw()
            if event.inaxes is not None:
                i=self.axes.index(event.inaxes)

                if self.lines[i].get_data()[0].__len__():
                    xy=[x[[0,-1]] for x in self.lines[i].get_data()]
                    n0,n1=self.npairs[i]
                    for a,start,stop,npair in zip(self.axes[i+1:],self.start[i+1:],self.stop[i+1:],self.npairs[i+1:]):
                        if n0 in npair or n1 in npair:
                            n=n1 in npair  #if n1 in npair, then we need index 1, otherwise we need index 0
                            i=npair.index(n)
                            for k,s in enumerate([start,stop]):
                                data=[d for d in s[i].get_data()]
                                data[i]=xy[n][k]
                                s[i].set_data(data)
                                s[i].set_visible(True)
                                
                if np.all([len(line.get_data()[0]) for line in self.lines]):
                    self.unify_PC_plots()
                    
                    
                
            
            
        plt.connect('button_press_event',on_click)
        plt.connect('button_release_event',on_release)
                
    def get_2D_paths(self,npoints:int=200):
        """
        Gets the list of 2D paths for each plot, where we have a normalized
        distance axis as the reaction coordinate

        Returns
        -------
        tuple
            list of PC pairs from each plot

        """
        
        rc=np.linspace(0,1,npoints)
        
        PC=list()
        for line in self.lines:
            x,y=line.get_data()
            if len(x)==0:
                PC.append([np.zeros(npoints),np.zeros(npoints)])
            else:
                d0=np.concatenate(([0],np.sqrt(np.diff(x)**2+np.diff(y)**2)))
                d=np.cumsum(d0)
                d/=d[-1]
                i=np.concatenate(([True],np.diff(d)>0))
                x,y,d=x[i],y[i],d[i]
                # i=[np.argwhere(q[:-1]==q[1:])[:,0] for q in [x,y]]
                # print(i)
                
                # for q,i0 in zip([x,y],i):
                #     q[i0+1]=(q[i0]+q[i0+2])/2

                PC.append([linear_ex(d,q,rc) for q in [x,y]])
        return PC
    
    def get_paths(self,npoints:int=200):
        """
        Determines a single, multidimensional path for all princle components. 
        Since we plot PC0xPC1, PC1xPC2, PC2xPC3, for example, some components
        are defined twice. We take the average of these components at each value
        of the reaction coordinate

        Parameters
        ----------
        npoints : int, optional
            Number of points in the PC axis. The default is 200.

        Returns
        -------
        None.

        """
        PC2D=self.get_2D_paths(npoints=npoints)
        
        nlist=np.unique(np.concatenate(self.npairs))
        
        PC=list()
        n=list()
        for npair,PC0 in zip(self.npairs,PC2D):
            if npair[0] in n and npair[1] in n:
                continue
            elif npair[0] in n:
                # x,y=PC0
                # d=np.linspace(x.min(),x.max(),npoints)
                # d=np.cumsum(np.abs(x))
                # i=list()
                # for x0,d0 in zip(PC[n.index(npair[0])],d):
                #     i.append(np.argmin((x-x0)**2+(d-d0)**2))
                # n.append(npair[1])
                # PC.append(y[i])
                
                ref=PC[n.index(npair[0])]
                # extrema=np.sort(np.concatenate(([0],argrelextrema(ref,np.less)[0],
                #                     argrelextrema(ref,np.greater)[0],[len(ref)-1])))
                
                extrema=np.sort(np.concatenate(([0],find_peaks(-ref)[0],find_peaks(ref)[0],[len(ref)-1])))
                values=ref[extrema]
                extrema[-1]+=1
                
                x=PC0[0]
                # xextrema=np.sort(np.concatenate(([0],argrelextrema(x,np.less)[0],
                #                      argrelextrema(x,np.greater)[0],[len(x)-1])))
                
                xextrema=np.sort(np.concatenate(([0],find_peaks(-x)[0],find_peaks(x)[0],[len(x)-1])))
                
                xvalues=x[xextrema]
                
                norm=[len(ref),ref.max()-ref.min()]
                
                PC1=list()
                for k in range(len(extrema)-1):
                    i=np.argmin([np.abs(xe0-extrema[k])/norm[0]+\
                                 np.abs(xe1-extrema[k+1])/norm[0]+\
                                 np.abs(xv0-values[k])/norm[1]+\
                                 np.abs(xv1-values[k+1])/norm[1] \
                        for xe0,xe1,xv0,xv1 in zip(xextrema[:-1],xextrema[1:],xvalues[:-1],xvalues[1:])])
                    if x[xextrema[i+1]]>x[xextrema[i]]: 
                        PC1.append(linear_ex(x[xextrema[i]:xextrema[i+1]],\
                                             PC0[1][xextrema[i]:xextrema[i+1]],ref[extrema[k]:extrema[k+1]]))
                    else:
                        PC1.append(linear_ex(x[xextrema[i+1]:xextrema[i]:-1],\
                                             PC0[1][xextrema[i+1]:xextrema[i]:-1],ref[extrema[k]:extrema[k+1]]))
                        
                PC.append(np.concatenate(PC1))
                n.append(npair[1])
                    
                    
                
                
            elif npair[1] in n:
                y,x=PC0
                d=np.linspace(x.min(),x.max(),npoints)
                d=np.cumsum(np.abs(x))
                i=list()
                for x0,d0 in zip(PC[n.index(npair[1])],d):
                    i.append(np.argmin((x-x0)**2+(d-d0)**2))
                n.append(npair[0])
                PC.append(y[i])
            else:
                n.extend(npair)
                PC.extend(PC0)
            
        return n,PC
    
    def unify_PC_plots(self,npoints:int=200):
        """
        Calculates the paths with consistent PC values (get_paths) and replaces
        the exist plots with these new plots
        
        Parameters
        ----------
        npoints : int, optional
            Number of points in the PC axis. The default is 200.

        Returns
        -------
        None.

        """
        
        n,PC=self.get_paths(npoints)
        
        for npair,line in zip(self.npairs,self.lines):
            line.set_data((PC[n.index(npair[0])],PC[n.index(npair[1])]))
                
                
        
    def get_count(self,d:float=1):
        """
        Count up the number of time points within a distance of d to points on
        the PC path

        Parameters
        ----------
        d : float, optional
            Distance to the position of the principle components to include in 
            the count. The default is 1.

        Returns
        -------
        None.

        """
        n,PC=self.get_paths()    
        PC=np.array(PC).T
        
        d2=d**2
        count=np.array([(((self.pca.PCamp[n].T-PC0)**2).sum(axis=1)<d2).sum() for PC0 in PC],dtype=int)
        return count
    
    
    def get_DeltaG(self,d:float=1,T=298):
        count=self.get_count(d=d)
        DelG=-np.log(count/count.max())*8.314*T
        return DelG
    
    
    def write_traj(self,filename:str,mode:str='uniform',npts:int=500,d:float=1):
        """
        Writes out a trajectory to show the path through the PCA histograms.
        Provide a filename (filetype derived from ending of filename). Default
        mode is 'uniform', which weights each point in the path equally. 
        Alternatively, can use 'weighted' which spends more time at points
        with higher probability or use 'stochastic', which produces a random
        trajectory with probabilities based on a random walk through the 
        energy landscape.

        Parameters
        ----------
        filename : str
            Location to store the trajectory.
        mode : str, optional
            Type of trajectory ['uniform','weighted','stochastic']. Note that
            we only compare the first letter ['u','w','s']. 
            The default is 'uniform'.
        npts : int, optional
            How many points to use in the trajectory. For 'uniform', this argument
            is ignored. 
            The default is 500.
            

        Returns
        -------
        None.

        """
        atoms=copy(self.pca.atoms)
        n,PC=self.get_paths()
        PC=np.array(PC).T
        if mode[0].lower()=='u':
            PCwt=PC
        elif mode[0].lower()=='w':
            count=self.get_count()    
            weight=np.diff(np.round(np.cumsum(count/count.sum()*npts))).astype(int)
            PCwt=np.zeros([npts,PC.shape[1]])
            i=0
            for PC0,wt in zip(PC,weight):
                PCwt[i:i+wt]=PC0
                i+=wt
            
        with Writer(filename, len(atoms)) as W:
            for PC1 in PCwt:
                pos=self.pca.mean
                for PC0,v in zip(PC1,self.pca.PC.T):
                    pos+=PC0*v.reshape([v.shape[0]//3,3])
                atoms.positions=pos
                W.write(atoms)
        
        
        
        
        
            
            
                
            