#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:38:52 2022

@author: albertsmith
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
from copy import copy
import re
from ..misc.disp_tools import set_plot_attr,NiceStr


class DataPlots():
    def __init__(self,data=None,style='plot',errorbars=True,index=None,rho_index=None,title=None,fig=None,mode='auto',split=True,plot_sens=True,std_filter=True,**kwargs):
        self.fig=fig if fig is not None else plt.figure(title)
        self.fig.clear()
        self.fig.canvas.mpl_connect('close_event', self.close)
        self.project=None
        self.data=[]
        self.index=[]
        self.ax=[]
        self.ax_sens=None
        self.rho_index=[]
        self.hdls=[]
        self.hdls_sens=[]
        self.style=''
        self.colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.threshold=0.7  #Overlap threshold
        self.tclabels=None
        self._plot_sens=plot_sens
        self.std_filter=std_filter

        assert mode in ['auto','union','b_in_a'],"mode must be 'auto','union', or 'b_in_a'"
        self._mode=mode  
        if data is not None:
            self.append_data(data,style=style,errorbars=errorbars,index=index,rho_index=rho_index,split=split,plot_sens=plot_sens,**kwargs)
    
    def _ipython_display_(self):
        return self.fig        
    
    @property
    def mode(self):
        if self._mode=='auto':
            for d in self.data:
                if d.label.dtype.kind not in ['i','f']:
                    return 'b_in_a'
            return 'union'
        return self._mode
        
    
    def clear(self):
        proj=self.project
        self.__init__(fig=self.fig)
        self.project=proj
    
    def close(self,event=None):
        "Clear out the object on close and remove from project"
        "Some issues arise for updating figures after they have been closed, so we just discard the whole object"
        if self.project is not None and self in self.project.plots:
            i=self.project.plots.index(self)
            self.project.plots[i]=None
            if self.project.current_plot==i+1:self.project.current_plot=0
        # self.__init__(fig=self.fig)
        plt.close(self.fig)
    
    def show(self):
        self.fig.show()
    
    def append_data(self,data,style='plot',errorbars=True,index=None,rho_index=None,split=True,plot_sens=True,**kwargs):
        self._plot_sens=self._plot_sens if len(self.data) else plot_sens
        # if len(self.data)==0: #Why did we check for no data currently in project?
        if index is not None:
            data=copy(data)
            for f in ['R','Rstd','S2','S2std','label']:
                if getattr(data,f) is not None:setattr(data,f,getattr(data,f)[index])
            if data.select is not None and len(data.select):
                data.source=copy(data.source)
                data.source.select=copy(data.source.select)
                data.select.sel1=data.select.sel1[index]
                if data.select.sel2 is not None:
                    data.select.sel2=data.select.sel2[index]
            # I just uncommented the line below (26.09.2024)
            # The reduced data from above no longer requires an index, so we set index to None
            index=None
        
        self.data.append(data)        
        self.rho_index.append(self.calc_rho_index() if rho_index is None else np.array(rho_index,dtype=int))
        self.index.append(self.xindex() if index is None else np.array(index,dtype=int))
        if np.all(np.array(self.rho_index[-1])==None) or len(self.index[-1])==0:
            self.data.pop(-1)
            self.rho_index.pop(-1)
            self.index.pop(-1)
            return
        
        
        self.style+=style[0].lower() if style[0].lower() in ['p','s','b'] else 'p'
        
        if len(self.data)==1:
            self.setup_plots()
        for h in self.hdls:h.append(None)
        self.hdls_sens.append(None)
           
        if self._plot_sens:self.plot_sens()
        self.plot_data(errorbars=errorbars,split=split,**kwargs)
        
        # self.show_tc()
    
    
    def setup_plots(self):
        """
        Set up the subplots for this figure
        """
        rho_index=self.rho_index[0]
        plot_sens=self._plot_sens
        if plot_sens:
            self.ax_sens=self.fig.add_subplot(2*plot_sens+rho_index.size,1,1)
            bbox=self.ax_sens.get_position()
            bbox.y0-=0.5*(bbox.y1-bbox.y0)
            self.ax_sens.set_position(bbox)
        
        self.ax=[self.fig.add_subplot(2*plot_sens+rho_index.size,1,k+1+2*plot_sens)\
                 for k in range(rho_index.size)]
        self.hdls=[list() for _ in self.ax]
        for a in self.ax[1:]:a.sharex(self.ax[0])
        
        not_rho0=self.data[0].sens.rhoz[0,0]/self.data[0].sens.rhoz[0].max()<.98
        for m,(k,a) in enumerate(zip(rho_index,self.ax)):
            color=self.colors[m%len(self.colors)]
            if not((a if hasattr(a,'is_last_row') else a.get_subplotspec()).is_last_row()):
                   plt.setp(a.get_xticklabels(), visible=False)
            if hasattr(self.data[0].sens,'opt_pars') and 'R2ex' in self.data[0].sens.opt_pars['options'] \
                and k==self.data[0].sens.rhoz.shape[0]-1:
                    a.set_ylabel(r'$R_2^{ex}$ / s$^{-1}$')
            else:
                a.set_ylabel(r'$\rho_{'+'{}'.format(k+not_rho0)+r'}^{(\theta,S)}$')
            a.yaxis.label.set_color(color)
        
            
    def show_tc(self,show_z:bool = False) -> None:
        """
        Shows the mean correlation times for each detector in the upper right
        corner of the plot

        Parameters
        ----------
        show_z : bool, optional
            Toggle to show <z0> instead of 10**(<z0>). The default is False.

        Returns
        -------
        None
            
        """
        
        self.remove_tc()   
        self.tclabels=list()         
        string=NiceStr(r'$z_{1}^0$XXX{0:q2}' if show_z else r'$\tau_c$XXX{0:q2}',unit='' if show_z else 's')
        for m,(a,k) in enumerate(zip(self.ax,self.rho_index[0])):
            sym='~'
            xlim=a.get_xlim()
            ylim=a.get_ylim()
            
            val=self.data[0].sens.info['z0'][k]
            rhoz=self.data[0].sens.rhoz[k]
            if rhoz[0]>rhoz.max()*0.9:  #Detector sensitive to infinitely short tc
                b=np.argmin(np.abs(rhoz-rhoz.max()/2))
                val=self.data[0].sens.z[b]
                sym='<'
            elif rhoz[-1]>rhoz.max()*0.9: #Detector sensitive to infinitely long tc
                b=np.argmin(np.abs(rhoz-rhoz.max()/2))
                val=self.data[0].sens.z[b]
                sym='>'
            
            
            if not(show_z):val=10**val
            self.tclabels.append(\
                a.text(xlim[0]+(xlim[1]-xlim[0])*.025,ylim[0]+(ylim[1]-ylim[0])*0.95,
                   horizontalalignment='left',verticalalignment='top',
                   color=self.colors[m%len(self.colors)],fontsize='x-small',   
                   s=string.format(val,k).replace('XXX',sym)))
        return self
            
    def remove_tc(self):
        """
        Remove the mean correlation times for each detector

        Returns
        -------
        None.

        """
        if self.tclabels is not None:
            for v in self.tclabels:v.remove()
            self.tclabels=None
        return self
    
    def calc_rho_index(self,i=-1):
        if len(self.data)==1:return np.arange(self.data[0].R.shape[1])
        in0,in1=self.data[0].sens.overlap_index(self.data[i].sens,threshold=self.threshold)
        return [in1[ri==in0][0] if np.isin(ri,in0) else None for ri in self.rho_index[0]]

    
    def plot_sens(self,i=-1,**kwargs):
        """
        Adds the sensitivity of the current data object to the sensitivity plot
        (function will only return if self.ax_sens is None, i.e. there is no 
        sensitivity plot)
        """
        if not(self._plot_sens):return
        color,linestyle=(None,'-') if self.data.__len__()==1 or i==0 else ((.2,.2,.2),':')
        maxes=self.data[i].sens.rhoz.max(1)
        norm=maxes.max()/maxes.min()>20
        
        rho_index=list()
        for ri in self.rho_index[i]:
            if ri is not None:rho_index.append(ri)
        rho_index=np.array(rho_index,dtype=int)
        self.hdls_sens[i]=self.data[i].sens.plot_rhoz(rho_index,color=color,
                      linestyle=linestyle,ax=self.ax_sens,norm=norm)
        if color is None:
            self.colors=[h.get_color() for h in self.hdls_sens[i]]
        
    def plot_data(self,i=-1,errorbars=True,split=True,**kwargs):        
        x=self.xpos(i)        
        for k,(a,ri) in enumerate(zip(self.ax,self.rho_index[i])):
            if ri is not None:
                plt_style=self.plt_style(k,i,split,**kwargs)
                Rstd=self.data[i].Rstd[self.index[i],ri]
                i0=Rstd<1e3 if self.std_filter else np.ones(Rstd.shape,dtype=bool)
                self.hdls[k][i]=plot_rho(x[i0],self.data[i].R[self.index[i],ri][i0],
                              Rstd[i0] if errorbars else None,ax=a,**plt_style)[1]
        self.xlabel(i)        
        if self.style[i]=='b':self.adjust_bar_width()
    

    def comparex(self,i=-1):
        d0,di=self.data[0],self.data[i]
        
        if len(d0)==len(di):return np.arange(len(d0)),np.arange(len(d0))
        
        if d0.select is not None and di.select is not None and len(d0.select) and len(di.select):
            out=d0.select.compare(di.select)
            if len(out[0]):
                return d0.select.compare(di.select)

        in01=list()
        for da,db in [(d0,di),(di,d0)]:
            inab=list()
            for lbl in db.label:
                if lbl in da.label:inab.append(np.argwhere(da.label==lbl)[0,0])
            in01.append(np.array(inab))
        return in01
            # return np.array(in01,dtype=int)
            
    def xindex(self,i=-1):
        """
        Returns an index of which data points to plot for comparison to the 
        initial data object (in the correct order for comparison)
        """
        i%=len(self)
        if i<len(self.index):return self.index[i] #Index already calculated
        if i==0:  #First data entry (plot everything)
            return np.arange(self.data[0].R.shape[0])
        
        if len(self.data[i])==len(self.index[0]): #Matching lengths (plot everything)
            return np.arange(len(self.data[i]))
        
        mode=self.mode
        
        in1=self.comparex(i=i)[1]
        
        if mode=='b_in_a':return in1
        di=self.data[i]
        index=np.ones(di.R.shape[0],dtype=bool)
        index[in1]=False
        return np.concatenate((in1,np.arange(di.R.shape[0])[index]))
        
    def xpos(self,i=-1):
        """
        Returns the positions to plot data points for comparison to the initial
        data object.
        """
        i%=len(self)
        if i==0:
            lbl=self.data[0].label
            if lbl.dtype.kind in ['i','f']:
                # return lbl[self.xindex(i)]
                return lbl
            else:
                return np.arange(self.data[0].R.shape[0])[self.xindex(0)]
        

        mode=self.mode
        
        
        if mode=='b_in_a':
            return self.xpos(i=0)[self.comparex(i=-1)[0]]
        else:
            di=self.data[i]
            xindex=self.xindex(i=i)
            xpos=np.zeros(xindex.shape)
            in0,in1=self.comparex(i=-1)
            xpos[:in1.size]=self.xpos(i=0)[in0]
            index=np.ones(di.R.shape[0],dtype=bool)
            index[in1]=False
            if di.label.dtype.kind in ['i','f']:
                xpos[in1.size:]=di.label[index]
            else:
                start=(self.xpos(i=0)[in0]).max()+1
                xpos[index]=np.arange(start,start+index.sum())
            return xpos
        
    def xlabel(self,i=-1):
        """
        Returns the labels for the x-axis. Set update to True in order to actually
        apply those labels
        """
        if np.issubdtype(self.data[0].label.dtype,np.number) and np.issubdtype(self.data[0].label.dtype,np.number):
        # if self.data[0].label.dtype.kind in ['f','i'] and self.data[i].label.dtype.kind in ['f','i']:
            # for a in self.ax[:-1]:a.set_xticklabels([])   #Only show label on last axis
            return #Just use the automatic labels if all labels are numeric
        xpos0,xposi=self.xpos(0),self.xpos(i)
        xpos=np.union1d(xpos0,xposi)
        # for a in self.ax:a.set_xticks(xpos)
        
        lbl0,lbli=self.data[0].label,self.data[i].label
        xlabel=[lbl0[np.argwhere(xp==xpos0)[0,0]] if xp in xpos0 else lbli[np.argwhere(xp==xposi)[0,0]] \
                for xp in xpos]
            
        self.ax[-1].set_xticklabels([],rotation=90)
        def fun(i,pos):
            i=int(i)
            if i>=len(xlabel):return ''
            if i<0:return ''
            return xlabel[i]
        self.ax[-1].xaxis.set_major_locator(plt.MaxNLocator(30,integer=True))
        self.ax[-1].xaxis.set_major_formatter(plt.FuncFormatter(fun))
            
        # step=1 if len(xlabel)<=30 else len(xlabel)//30
        # self.ax[-1].set_xticklabels([xl if k%step==0 else '' for k,xl in enumerate(xlabel)],rotation=90)
        
        
        
            
    def __len__(self):
        return len(self.data)
    
    def plt_style(self,ax_index,i=-1,split=True,**kwargs):
        out=dict()
        i=i%len(self.data)
        out['style']=self.style[i]
        out['split']=split if self.style[i]=='p' else False  #split only useful for 'plot' style plots
        
        if self.style[i]=='b':
            out['color']=self.colors[ax_index]
            out['hatch']=['','///','+++','ooo','xxx','\\\\\\','..','OOO','***','|||'][len([_ for _ in re.finditer('b',self.style[:i])])%9]
        elif self.style[i]=='p':
            out['color']=self.colors[ax_index] if i==0 else (0,0,0)
            out['linestyle']=['-',':','--','-.'][len([_ for _ in re.finditer('p',self.style[:i])])%4]
        elif self.style[i]=='s':
            out['linestyle']=''
            out['marker']=['o','^','x','*','D','p','d','h'][len([_ for _ in re.finditer('s',self.style[:i])])%8]
            out['markersize']=5
            out['color']=(0,0,0) if 'b' in self.style else self.colors[ax_index]
        out.update(kwargs)
        return out  
    
    def set_ylim(self,lower=None,upper=None):
        
        if lower is None:
            lower=min([0,np.min()])
    
    def adjust_bar_width(self):
        BCclass=mpl.container.BarContainer
        for hd in self.hdls:    #Loop over the plot axes
            #How many bars do we need to fit into plot?
            nbars=0
            for ha in hd:            #Loop over the data in each axis
                if ha is not None:
                    if any([h0.__class__ is BCclass for h0 in ha]):nbars+=1
            
            if nbars>1:
                count=0
                for k,ha in enumerate(hd):           #Loop over data in each axis
                    if ha is not None:
                        #TODO can we really just sort xpos???
                        x=np.sort(np.array(self.xpos(k),dtype=float)) 
                        dx=np.diff(x).min()
                        x+=-0.45+dx*count*0.9/nbars
                        
                        ebc,bc=((None,ha[0]) if ha[0].__class__ is BCclass else (ha[0],ha[1])) if \
                            (ha[0].__class__ is BCclass or (len(ha)>1 and ha[1].__class__ is BCclass)) else (None,None)
                        
                        if bc:
                            for h00,x0 in zip(bc,x):      #Loop over each bar'
                                h00.set_width(0.9/nbars)  
                                h00.set_x(x0)
                        if ebc:
                            "Errorbar collections are REALLY annoying"
                            ebc.lines[0].set_xdata(x+dx*0.45/nbars)   #line position
                            ebc.lines[1][0].set_xdata(x+dx*0.45/nbars)  #Cap 1 position
                            ebc.lines[1][1].set_xdata(x+dx*0.45/nbars) #Cap 2 position
                            segs=ebc.lines[2][0].get_segments()
                            for x0,s in zip(x+dx*0.45/nbars,segs):
                                s[:,0]=x0
                            ebc.lines[2][0].set_segments(segs)
                        if bc:count+=1
                                    
     

def plot_rho(lbl,R,R_std=None,style='plot',color=None,ax=None,split=True,**kwargs):
    """
    Plots a set of rates or detector responses. 
    """
    i=np.argsort(lbl)
    lbl,R,R_std=[x[i] if x is not None else None for x in [lbl,R,R_std]]
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
    
    hdls=list()
    for lbl,R,R_u,R_l in zip(lbl1,R1,R_u1,R_l1):
        if R_l is None and style.lower()[0]!='b':
            hdls.append(ax.plot(lbl,R,color=color))
            set_plot_attr(hdls[-1],**kwargs)
        elif R_l is not None:
            hdls.append(ax.errorbar(lbl,R,[R_l,R_u],color=ebar_clr,capsize=3))
            set_plot_attr(hdls[-1],**kwargs)
        if style.lower()[0]=='b':
            kw=kwargs.copy()
            if 'linestyle' in kw: kw.pop('linestyle')
            hdls.append(ax.bar(lbl,R,color=color,**kw))
        if color is None:
            color=ax.get_children()[0].get_color()
    
    if lbl0 is not None:
        ax.set_xticks(lbl)
        ax.set_xticklabels(lbl0,rotation=90)

                
    return ax,hdls


def plot_fit(lbl,Rin,Rc,Rin_std=None,info=None,index=None,exp_index=None,fig=None,std_filter=True):
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
        i0=Rin_std[:,k]/Rin[:,k].max()<1e3 if std_filter else np.ones(Rin_std.shape[0],dtype=bool)
        a.bar(lbl[i0],Rin[i0,k],color=clr[np.mod(k,len(clr))])       #Bar plot of experimental data
        if Rin_std is not None:             
            a.errorbar(lbl[i0],Rin[i0,k],Rin_std[i0,k],color='black',linestyle='',\
                       capsize=3) #Errorbar
        a.plot(lbl[i0],Rc[i0,k],linestyle='',marker='o',color='black',markersize=3)
        if xax[k]:
            if lbl0 is not None:
                a.set_xticks(ii)
                a.set_xticklabels(lbl0[ii],rotation=90)
        else:
            plt.setp(a.get_xticklabels(),visible=False)
            if lbl0 is not None:
                a.set_xticks(ii)
        if yax[k]:
            if 'Type' in info.keys:
                a.set_ylabel(r'$R$ / s^${-1}$')
            elif 't' in info.keys:
                a.set_ylabel(r'$C(t)$')
            else:
                a.set_ylabel(r'$\rho_n^{(\theta,S)}$')
            a.set_ylabel(r'R / s$^{-1}$' if 'Type' in info.keys else r'')
        
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

def plot_fit_md(lbl,Rin,Rc,info=None,index=None,fig=None):
    
    if index is not None:
        lbl=lbl[index]
        Rin=Rin[index]
        Rc=Rc[index]
    
    if fig is None:
        fig=plt.figure(figsize=[12,9])
    ax,*_=subplot_setup(len(lbl),fig)
    
    t=info['t']
    
    for lbl0,Rin0,Rc0,a in zip(lbl,Rin,Rc,ax):
        a.plot(t,Rin0,color='red')
        a.plot(t,Rc0,color='black',linestyle=':')
        a.text(t[1],0.1,lbl0)
        a.set_ylim([-.1,1])
        if a.is_first_col():
            a.set_ylabel(r'$C(t)$')
        else:
            a.set_yticklabels('')
        if a.is_last_row():
            a.set_xlabel(r'$t$ / ns')
        else:
            a.set_xticklabels('')
        
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
    if np.prod(SZ)+.01<nexp: SZ[1]+=1
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