# -*- coding: utf-8 -*-


from . import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import copy

class CombinedPCA(PCA):
    def __init__(self,*pcas):
        super().__init__(pcas[0].select)
        # self.__dict__=copy(pcas[0].__dict__)
        self._pos=np.concatenate([pca0.pos for pca0 in pcas],axis=0)
        self.pcas=[copy(pca) for pca in pcas]
        
        self._traj=Traj(self)
        
        for pca,Range in zip(self.pcas,self.ranges):
             pca._PC=self.PC
             pca._pcamp=self.PCamp[:,Range[0]:Range[1]]
             pca._lambda=self.Lambda
            
        self.traj[0]
        
        
    @property
    def traj(self):
        return self._traj
    
    @property
    def n_trajs(self):
        return len(self.trajs)
        
    @property
    def trajs(self):
        return [pca.traj for pca in self.pcas]
    
    @property
    def lengths(self):
        return np.array([len(traj) for traj in self.trajs],dtype=int)
    
    @property
    def ranges(self):
        return [(self.lengths[:n].sum(),self.lengths[:n+1].sum()) for n in range(self.n_trajs)]
    
    def hist_by_traj(self,nmax:int=3,cmap='Reds',cmap0='jet',**kwargs):
        fig,ax=plt.subplots(nmax,self.n_trajs)
        if nmax==1:ax=np.array([ax])
        
        cm=plt.get_cmap(cmap)
        colors=np.array([cm(k) for k in range(256)])
        colors[:,-1]=np.linspace(0,1,257)[1:]**.25
        cm=ListedColormap(colors)
        
        maxbin=np.abs(self.PCamp[:nmax+1]).max()
        
        for q,ax0 in enumerate(ax.T):
            for ax00,n0,n1 in zip(ax0,range(nmax),range(1,nmax+1)):
                self.Hist.plot(n0,n1,ax=ax00,cmap=cmap0,maxbin=maxbin,**kwargs)
                index=np.zeros(len(self.traj),dtype=bool)
                index[self.ranges[q][0]:self.ranges[q][1]]=True
                self.Hist.plot(n0,n1,ax=ax00,cmap=cm,maxbin=maxbin,index=index)
                
        fig.tight_layout()
                
        return fig
    
    def hist2struct(self,nmax:int=3,from_traj:bool=True,select_str:str='protein',
                    ref_struct:bool=False,cmap='Reds',cmap0='jet',cmap_ch='gist_rainbow',n_colors=10,**kwargs):
        fig,ax=plt.subplots(nmax,self.n_trajs)
        
        cm=plt.get_cmap(cmap)
        colors=np.array([cm(k) for k in range(255)])
        colors[:,-1]=np.linspace(0,1,255)**.25
        cm=ListedColormap(colors)
        
        for q,ax0 in enumerate(ax.T):
            xlims=[]
            ylims=[]
            for ax00,n0,n1 in zip(ax0,range(nmax),range(1,nmax+1)):
                self.Hist.plot(n0,n1,ax=ax00,cmap=cmap0,**kwargs)
                ax00.set_xlim(ax00.get_xlim())
                ax00.set_ylim(ax00.get_ylim())
                xlims.append(ax00.get_xlim())
                ylims.append(ax00.get_ylim())
            self.pcas[q].Hist.hist2struct(nmax=nmax,from_traj=from_traj,select_str=select_str,
                                        ref_struct=ref_struct,ax=ax0.tolist(),
                                        cmap=cm,cmap_ch=cmap_ch,n_colors=n_colors,**kwargs)
            for ax00,xlim,ylim in zip(ax0,xlims,ylims):
                ax00.set_xlim(xlim)
                ax00.set_ylim(ylim)
        
        return fig
    
    def hist_t_depend(self,nmax:int=3,cmap='jet',cmap0='Greys',step:int=20,**kwargs):
        fig,ax=plt.subplots(nmax,self.n_trajs)
        if nmax==1:ax=np.array([ax])
        
        if isinstance(cmap,str):cmap=plt.get_cmap(cmap)
        
        
        for q,ax0 in enumerate(ax.T):
            for ax00,n0,n1 in zip(ax0,range(nmax),range(1,nmax+1)):
                self.Hist.plot(n0,n1,ax=ax00,cmap=cmap0,**kwargs)
                index=np.zeros(len(self.traj),dtype=bool)
                index[self.ranges[q][0]:self.ranges[q][1]:step]=True
                cmap=cmap.resampled(index.sum())
                c=cmap(np.arange(index.sum()))
                ax00.scatter(self.PCamp[n0][index],self.PCamp[n1][index],s=1,c=c)
                
        fig.tight_layout()
                
        return fig
        
        
            
        
    
    @property
    def filenames(self):
        return [pca.uni.filename for pca in self.pcas]
        
        
        




class Traj():
    def __init__(self,combinePCA):
        self.cPCA=combinePCA
        self._index=0
        self._traj_index=0
        self.mda_traj=self.trajs[0].mda_traj
        
    @property
    def trajs(self):
        return self.cPCA.trajs
    
    @property
    def pcas(self):
        return self.cPCA.pcas
    
    @property
    def lengths(self):
        return self.cPCA.lengths
        
    def __getitem__(self,i):
        # q=np.argmax(i<np.cumsum(np.concatenate([[0],self.cPCA.lengths])))
        q=np.argmax(i<np.cumsum(self.cPCA.lengths))
        pca=self.pcas[q]
        self.cPCA._uni=pca.uni
        self.cPCA._atoms=pca.atoms
        self.mda_traj=pca.traj.mda_traj
        self._traj_index=q
        self._index=i
        self.cPCA.select=pca.select
        return pca.traj[(i%len(self))-self.lengths[:q].sum()]
    
    @property
    def traj_index(self):
        return self._traj_index
    
    @property
    def index(self):
        return self._index
        
    
    def __len__(self):
        return self.tf
        
    @property
    def t0(self):
        return 0
    
    @t0.setter
    def t0(self,t0):
        pass
    
    @property
    def tf(self):
        return np.sum([len(traj) for traj in self.cPCA.trajs])
    
    @tf.setter
    def tf(self,tf):
        pass
    
    @property
    def step(self):
        return self.cPCA.trajs[0].step
    
    @step.setter
    def step(self,step):
        pass
    
    @property
    def frame(self):
        return (self.mda_traj.frame-self.trajs[self._traj_index].t0)//self.step