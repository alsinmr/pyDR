#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:00:48 2022

@author: albertsmith
"""

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from MDAnalysis import Writer
from scipy.interpolate import interp1d
from pyDR.misc.tools import linear_ex

class Path2Energy():
    def __init__(self,pca):
        """
        Allows one to plot traces through one or more PCA histogram plots in
        order to define a reaction coordinate. From this coordinate, we can
        create both a histogram and corresponding energy landscape, but also
        create a trajectory that can be shown in chimeraX to see how the 
        system traverses this landscape.
        
        If only 2 principal components are shown, then by default, one draws
        a line through the plot. Otherwise, one marks positions on all plots
        and a line is interpolated through those positions. Note that positions
        need to marked in the same order on all axes.

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
        self.positions=None #Line that indicates the current axis positions
        self.markers=None #List of lists with all markers in them
        self.npairs=None     #List of principal component pairs to plot
        self._path=None
        self._count=None
        self.mode=None
        
        self.rx_coord=None  #Calculated reaction coordinate (linear ax)
        self.PC=None        #Calculated value of the principal componts as a function of rx_coord
        
        self.cid=None       #ID for the
        
    
    def __setattr__(self,name,value):
        if name=='_path':
            self._count=None
        super().__setattr__(name,value)
        
    def load_points(self,pts):
        """
        Optionally, after creating plots, one may load points into a new
        instance of Path2Energy instead of picking them manually
        """
        pts=np.array(pts)
        assert len(self.n)==pts.shape[0],f'pts should have size {self.n}x3'
        marks=['x','o','+','v','>','s','1','*']
        clrs=plt.get_cmap('tab10')
        for (n0,n1),markers,line,ax in zip(self.npairs,self.markers,self.lines,self.axes):
            line.set_data(pts)
            line.set_visible(True)
            for mk in markers:
                markers.remove(mk)  #Remove existing markers
                ax.collections.pop(ax.collections.index(mk))
            i0=self.n.tolist().index(n0)
            i1=self.n.tolist().index(n1)
            for k,pt in enumerate(pts[[i0,i1]].T):
                markers.append(ax.scatter(*pt,s=100,color=clrs(k%10),marker=marks[k%len(marks)]))
            

    def get_points(self):
        """
        Extract the list of points marked in the plots

        Returns
        -------
        np.array

        """
        
        if self.mode=='p':
            nmks=len(self.markers[0])
            for k,markers in enumerate(self.markers):         
                assert len(markers)>=nmks,f"Axis {k+1} does not have enough marks"
                assert len(markers)<=nmks,f"Axis {k} does not have enough marks"
            
            path0=np.zeros([len(self.n),nmks])
            for k,n in enumerate(self.n):
                for npair,markers in zip(self.npairs,self.markers):
                    if n in npair:
                        for q,marker in enumerate(markers):
                            path0[k,q]=marker.get_offsets()[0,npair.index(n)]
        
            return path0
        
        
            
    
    def create_plots(self,nmax:int=1,npairs:list=None,axes:list=None,mode:str='auto',**kwargs):
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
        mode : str, optional
            Set to 'points' in order to select points instead of tracing a path.
            Path tracing is only available if only 1 2D plot is shown.
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

        if len(self.npairs)>1:mode='points'
        
        self.mode=mode.lower()[0]
        
        if self.mode!='p':
            def on_move(event):
                """
                Function that draws a line as the mouse is moved within one of
                the axes (initialized via on_click, terminated via on_release)

                """
                if event.inaxes is None:
                    on_release(event)
                else:
                    i=self.axes.index(event.inaxes)
                    self.lines[i].set_xdata(np.concatenate((self.lines[i].get_xdata(),[event.xdata])))
                    self.lines[i].set_ydata(np.concatenate((self.lines[i].get_ydata(),[event.ydata])))
                    plt.draw()
                    
            def on_click(event):
                """
                Function that initializes drawing in an axis
                """
                
                if event.inaxes is None or not(event.inaxes in self.axes):return
                self._path=None

                i=self.axes.index(event.inaxes)      
                self.lines[i].set_xdata(np.array([])) #Remove any existing line
                self.lines[i].set_ydata(np.array([]))
                
                if event.button==1:
                    self.lines[i].set_visible(True)
                    self.cid=plt.connect('motion_notify_event',on_move)
                    plt.draw()
                elif event.button==3:
                    self.lines[i].set_visible(False)
                    plt.draw()

            def on_release(event):
                """
                Function to terminate drawing of path
                """                    
                
                if event.button!=1:return
                plt.disconnect(self.cid) #Release on_move function                
                plt.draw()
                
            plt.connect('button_press_event',on_click)
            plt.connect('button_release_event',on_release)
                
        else:
            #Multiple plots: use markers and interpolate instead of drawing a line
            self.positions=[[a.plot([0,0],a.get_ylim(),linestyle='--',color='blue',visible=False)[0],
                         a.plot(a.get_xlim(),[0,0],linestyle='--',color='blue',visible=False)[0]] for a in self.axes]
            
            self.markers=[list() for _ in range(len(self.axes))] #List of markers on every axis
            
            marks=['x','o','+','v','>','s','1','*']
            clrs=plt.get_cmap('tab10')
            
            def on_click(event):
                """
                Function to run when clicking on an axis

                """
                
                if event.inaxes is None or not(event.inaxes in self.axes):return
                self._path=None
                i=self.axes.index(event.inaxes)
                
                if event.button==3: #Clear last marker
                    if len(self.markers[i]):
                        mk=self.markers[i].pop(-1)
                        self.axes[i].collections.pop(self.axes[i].collections.index(mk))
                        self.update_markers(last_plot=i)
                else:
                    markers=self.markers[i]
                    markers.append(self.axes[i].scatter(event.xdata,event.ydata,s=100,\
                                    marker=marks[len(markers)%len(marks)],color=clrs(len(markers)%10)))
                    self.update_markers(i)
                    
            plt.connect('button_press_event',on_click)
                
    def update_markers(self,last_plot):
        """
        Updates positions of the markers so that they are consistent across all
        plots. Adds lines corresponding to positions of the current marker.

        Returns
        -------
        None.

        """
        
        i=last_plot
        
        clrs=plt.get_cmap('tab10')
        
        if len(self.markers[i]):
            n0,n1=self.npairs[i]
            x,y=self.markers[i][-1].get_offsets()[0]
            
            mk_index=len(self.markers[i])-1  #how many markers on this plot
            
            
            for npair,markers in zip(self.npairs,self.markers):
                if len(markers)>mk_index:   #make existing markers consistent with the latest update
                    if n0 in npair:
                        data=markers[mk_index].get_offsets()[0]
                        data[npair.index(n0)]=x
                        markers[mk_index].set_offsets(data)
                    elif n1 in npair:
                        data=markers[mk_index].get_offsets()[0]
                        data[npair.index(n1)]=y
                        markers[mk_index].set_offsets(data)
                    
                    
        for pos,markers,(n0,n1) in zip(self.positions,self.markers,self.npairs):   #Update reference lines
            nmks=len(markers)
            for p in pos:
                p.set_visible(False)
                p.set_color(clrs(nmks%10))
            for npair,mks in zip(self.npairs,self.markers):
                if len(mks)>nmks:
                    if n0 in npair:
                        pos[0].set_xdata(mks[nmks].get_offsets()[0,npair.index(n0)])
                        pos[0].set_visible(True)
                    elif n1 in npair:
                        pos[1].set_ydata(mks[nmks].get_offsets()[0,npair.index(n1)])
                        pos[1].set_visible(True)
                        
        for line,markers in zip(self.lines,self.markers):
            data=list()
            for marker in markers:
                data.append(marker.get_offsets()[0])
            if len(data):
                line.set_data(*np.array(data).T)
                line.set_visible(True)
        plt.draw()
    
    @property
    def n(self):
        """
        List of principal components used

        Returns
        -------
        list

        """
        
        return np.unique(np.concatenate(self.npairs))
        
    def get_path(self,npts:int=200):
        """
        Extracts a path from the defined points

        Returns
        -------


        Parameters
        ----------
        npts : int, optional
            Number of data points in the interpolated path. The default is 200.

        Returns
        -------
        np.array
            Interpolated path

        """
        
        if self._path is None or len(self._path)!=npts:
            if self.mode=='p':
                nmks=len(self.markers[0])
                for k,markers in enumerate(self.markers):         
                    assert len(markers)>=nmks,f"Axis {k+1} does not have enough marks"
                    assert len(markers)<=nmks,f"Axis {k} does not have enough marks"
                
                path0=np.zeros([len(self.n),nmks])
                for k,n in enumerate(self.n):
                    for npair,markers in zip(self.npairs,self.markers):
                        if n in npair:
                            for q,marker in enumerate(markers):
                                path0[k,q]=marker.get_offsets()[0,npair.index(n)]
                                
                d0=np.sqrt(np.array([np.diff(path00)**2 for path00 in path0]).sum(0))
                
                d=np.cumsum(np.concatenate(([0],d0)))
                d/=d[-1]
                dnew=np.linspace(0,1,npts)
                
                path=np.array([interp1d(d,y,kind='cubic')(dnew) for y in path0])
                
                for line,(n0,n1) in zip(self.lines,self.npairs):
                    line.set_xdata(path[self.n.tolist().index(n0)])
                    line.set_ydata(path[self.n.tolist().index(n1)])
                    line.set_visible(True)
                plt.draw()
                self._path=path
            else:
                pass
        
        return self._path
    
    
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
        if self._count is None or self._count[0]!=d:
            PC=self.get_path()    
            n=self.n
            PC=np.array(PC).T
            
            d2=d**2
            self._count=d,np.array([(((self.pca.PCamp[n].T-PC0)**2).sum(axis=1)<d2).sum() for PC0 in PC],dtype=int)
        return self._count[1]
    
    
        
    def get_DelG(self,d:float=1,T=298):
        count=self.get_count(d=d)
        DelG=-np.log(count/count.max())*8.314*T
        return DelG
    
    def plot_DelG(self,d:float=1,T=298,ax=None,show_mks:bool=True,**kwargs):
        DelG=self.get_DelG(d=d,T=T)
        
        if ax is None:ax=plt.figure().add_subplot(111)
        d=np.cumsum(np.sqrt(((self.get_path()[:,1:]-self.get_path()[:,:-1])**2).sum(0)))
        ax.plot(d,DelG/1000,**kwargs)
        ax.set_ylabel(r'$\Delta G$ / (kJ/mol)')
        ax.set_xticks([])
        
    def chimera(self,pts='all'):
        """
        Plots the points from the histograms in chimera. One may provide an
        index determining what points to plot, or use the default option,
        pts='all', which will plot all points on the plots

        Parameters
        ----------
        pts : TYPE, optional
            DESCRIPTION. The default is 'all'.

        Returns
        -------
        None.

        """
        chimera=self.pca.project.chimera
        if chimera.current is None:chimera.current=0
        
        PC=self.get_points()
        if pts=='all':
            pts=np.arange(PC.shape[1])
        elif hasattr(pts,'__len__'):
            pts=np.array(pts)
        else:
            pts=np.array([pts])
            
        clrs=plt.get_cmap('tab10')
        
        for k,PC in enumerate(PC.T):
            if k in pts:
                PCamp=np.zeros([self.n.max()+1])
                for m,n in enumerate(self.n):
                    PCamp[n]=PC[m]
                self.pca.chimera(PCamp=PCamp)
                nmdls=chimera.CMX.how_many_models(chimera.CMXid)
                color=[int(c*100) for c in clrs(k%10)]
                self.pca.project.chimera.command_line(f'color #{nmdls} '+','.join([f'{c}' for c in color]))
        
            
        
        
        
        
                        
        
                    
            

                    
                
        
        
        
            
    
            
                    
                    
                    
                    
                    
                    
                    
                    
                    