# -*- coding: utf-8 -*-

import numpy as np
from .. import clsDict,Defaults
from ..MDtools import vft
from ..misc.tools import AA
from ..IO.bin_IO import read_EntropyCC,write_EntropyCC
from copy import copy
import matplotlib.pyplot as plt
import os

from matplotlib.colors import ListedColormap


sel_names={'ARG':['CZ','NE'],'HIS':['CD2','CG'],'HSD':['CD2','CG'],'LYS':['NZ','CE'],
           'ASP':['CG','CB'],'GLU':['CD','CG'],'SER':['OG','CB'],'THR':['CG2','CB'],
           'ASN':['OD1','CG'],'GLN':['OE1','CD'],'CYS':['SG','CB'],'VAL':['CG1','CB'],
           'ILE':['CD','CG1'],'LEU':['CD1','CG'],'MET':['CE','SD'],'PHE':['CG','CB'],
           'TYR':['CG','CB'],'TRP':['CD1','CG']}

sel_mult={'ARG':[3,3,3,3],'HIS':[2,3],'HSD':[2,3],'LYS':[3,3,3,3],
          'ASP':[3],'GLU':[3,3],'SER':[3],'THR':[3],
          'ASN':[2,3],'GLN':[2,3,3],'CYS':[3],'VAL':[3],
          'ILE':[3,3],'LEU':[3,3],'MET':[3,3,3],'PHE':[3],'TYR':[3],'TRP':[2,3]}


class EntropyCC:
    R=8.31446261815324 #J/mol/K
    
    def __init__(self,select,PCA=None):
        """
        Initialize the Entropy Cross-Correlation object with either a selection
        object or filename from a saved EntropyCC object

        Parameters
        ----------
        select : MolSelect or string
            pyDR selection object or filename
        pca : PCA object (optional)

        Returns
        -------
        None

        """
        filename=None
        if isinstance(select,str):
            filename=select
            with open(filename,'rb') as f:
                assert bytes.decode(f.readline()).strip()=='OBJECT:ENTROPYCC','Not an EntropyCC object'
                temp=read_EntropyCC(f)
                select=temp.select

    

        self.select=copy(select)
        self.select.molsys=copy(select.molsys)
        self.select.molsys._traj=copy(select.traj)
        
        self.reset()
        
        if filename is not None:
            self._Sres=temp._Sres
            self._Scc=temp._Scc
            
        self.PCA=PCA
            
        
    def reset(self):
        """
        Sets various fields back to unitialized state

        Returns
        -------
        self

        """
        self._resi=None
        self._resid=None
        self._sel=None
        self._index=None
        self._vt=None
        self._chi=None
        self._v_avg=None
        self._state=None
        self._state0=None
        self._mult=None
        self._Sres=None
        self._CCstate=None
        self._Scc=None
        self._CC=None
        self._CCpca=None
        self._Chi=None
        
        return self
    
    @property
    def project(self):
        if self.select.project is None:
            self.select.molsys.project=clsDict['Project']()
        return self.select.project
    
    @project.setter
    def project(self,project):
        self.select.molsys.project=project
        
    @property
    def resi(self):
        if self._resi is None:
            if self.select.sel1 is None:return None
            
            self._resi=[]
            self._resids=[]
            for s in self.select.sel1:
                self._resi.append(s.residues[0])
                self._resids.append(self._resi[-1].resid)
            self.select.repr_sel=[r.atoms.select_atoms('not name H*') for r in self._resi]
                
            self._resi=np.array(self._resi,dtype=object)
            self._resids=np.array(self._resids,dtype=int)
            
        return self._resi
    
    @property
    def resids(self):
        if self.resi is None:return None
        return self._resids
    
    @property
    def N(self):
        if self.resi is None:return 0
        return len(self.resi)
    
                
    
    @property
    def sel(self):
        """
        List of selections that are used to define the vectors required for
        extracting the sidechain rotamer angles

        Returns
        -------
        list
            7-element list of atom groups stepping from the outermost to innermost
            atoms

        """
        if self._sel is None:
            if self.select.sel1 is None:return []
            
            self._sel=[self.select.uni.atoms[:0] for _ in range(7)]
            self._index=np.zeros([7,self.N],dtype=bool)
            
            for k,resi in enumerate(self._resi):
                if resi.resname in sel_names:
                    atoms=get_chain(resi)
                    if len(atoms)-3!=len(sel_mult[resi.resname]):
                        print(resi)
                        for a in atoms:print(a.names)
                    for m,(a,b) in enumerate(zip(atoms,self._sel)):
                        self._sel[m]=b+a
                        self._index[m,k]=True
                    
        return self._sel
    
    @property
    def index(self):
        """
        Logical index that specifies for each residue how many of the atoms are
        defined (from 0 up to 7 atoms)

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self._index is None:
            if self.sel is None:return np.zeros([7,0],dtype=bool)
            
        return self._index
    
    @property
    def mult(self):
        """
        Returns how many possible states exist for each rotamer of each side
        chain.

        Returns
        -------
        list
            4-element list of the number of states existing for each rotamer of
            each side chain

        """
        if self._mult is None:
            out=[]
            for k,i in enumerate(self.index[3:]):
                out.append(np.array([sel_mult[res.resname][k] for res in self.resi[i]],dtype=int))
            self._mult=out
        return self._mult
    
    @property
    def total_mult(self):
        """
        Returns the total number of states for each residue, that is, the product
        of multiplicity of each rotamer of the side chain.
        
        Returns
        -------
        None.

        """
        out=np.ones(self.N,dtype=int)
        
        for mult,i in zip(self.mult,self.index[3:]):
            out[i]*=mult
        return out
            
    
    @property
    def v(self):
        """
        6-element list of bond vectors, used to extract the sidechain rotamer
        angles. Bond vectors are corrected for the periodic boundary conditions
        and normalized.

        Returns
        -------
        out : list
            6-element list of 3xN numpy arrays

        """
        out=[]
        for sel0,sel1,i0,i1 in zip(self.sel[:-1],self.sel[1:],self.index[:-1],self.index[1:]):
            v0=sel0.positions[i1[i0]]-sel1.positions
            out.append(vft.norm(vft.pbc_corr(v0.T,self.select.box)))
            
        return out
    
    @property
    def v_in_frame(self):
        """
        4-element list of bond vectors that are given in the frame of the next
        two bonds inwards
            *The next bond defines z
            *One bond further defines the xz plane
            
        We store this as just the x and y components, including normalization 
        to 1.

        Returns
        -------
        out : list
            4-element list of 2xN numpy arrays

        """
        out=[]
        for v0,v1,v2,i0,i1,i2 in zip(self.v[:-2],self.v[1:-1],self.v[2:],
                                     self.index[1:-2],self.index[2:-1],self.index[3:]):
            v=vft.applyFrame(v0[:,i2[i0]],nuZ_F=v1[:,i2[i1]],nuXZ_F=v2)
            out.append(v[:2]/np.sqrt(v[0]**2+v[1]**2))
            
        return out
    
    #%% Functions requiring loading of the trajectory
    @property
    def traj(self):
        return self.select.traj
    
    def load(self):
        """
        Sweeps through the trajectory and store v_in_frame at each step

        Returns
        -------
        None.

        """
        
        out=[np.zeros([2,v.shape[1],len(self.traj)]) for v in self.v_in_frame]
        
        self.traj.ProgressBar=Defaults['ProgressBar']
        
        for k,_ in enumerate(self.traj):
            for m,v in enumerate(self.v_in_frame):
                out[m][:,:,k]=v
        self._vt=out
        
        self.traj.ProgressBar=False
        return self
    
    @property
    def t(self):
        """
        Time axis in nanoseconds. Note that for appended trajectories, t is not
        valid and will return None

        Returns
        -------
        TYPE
            np.array

        """
        if self._state is not None:
            if self.state.shape[1]!=len(self.traj):return None
            
        return np.arange(len(self.traj))*self.traj.dt/1000
    
    @property
    def vt(self):
        """
        Returns the bond vectors in their reference frames for all times in the
        trajectory

        Returns
        -------
        list:
            4-element list of 3xNxnt numpy arrays

        """
        if self._vt is None:
            self.load()
        return self._vt
        
    
    @property
    def chi(self):
        """
        4-element list of rotamer angles (in degrees)

        Returns
        -------
        out : list
            4-element list of N-element numpy arrays

        """
        if self._chi is None:
            out=[]
            for v in self.vt:
                out.append(np.mod(np.arctan2(v[1],v[0])*180/np.pi+180,360))
            self._chi=out
        return self._chi
    
    @property
    def Chi(self):
        """
        Index-able Chi

        Returns
        -------
        np.array

        """
        if self._Chi is None:self._Chi=Chi(self)
        return self._Chi
    
    @property
    def v_avg(self):
        """
        Returns the time-averaged vector direction for each bond, where before
        averaging, we rotate vectors with angles greater than 360/mult such that
        all angles are less than 360/mult

        Returns
        -------
        None.

        """
        
        if self._v_avg is None:
            out=[]
            for v,chi,mult in zip(self.vt,self.chi,self.mult):
                v=copy(v)
                vout=np.zeros(v.shape[:-1])
                
                for q in [2,3]:
                    for angle in [m*360/q for m in range(1,q)]:
                        i=(np.logical_and(chi.T>=angle,mult==q)).T
                        v0=v[0,i]*np.cos(2*np.pi/q)+v[1,i]*np.sin(2*np.pi/q)
                        v1=-v[0,i]*np.sin(2*np.pi/q)+v[1,i]*np.cos(2*np.pi/q)
                        v[0,i]=v0
                        v[1,i]=v1
                    
                for q in [2,3]:
                    i=mult==q
                    angle=np.arctan2(v[1,i],v[0,i])*q
                    v0=np.median(np.cos(angle),axis=-1)
                    v1=np.median(np.sin(angle),axis=-1)
                    
                    angle=np.arctan2(v1,v0)/q
                    
                    vout[0,i]=-np.cos(angle)
                    vout[1,i]=-np.sin(angle)
                    
                    
                # v=v.mean(-1)
                # v/=np.sqrt(v[0]**2+v[1]**2) #Renormalization
                # out.append(v)
                out.append(vout)
            self._v_avg=out
            
        return self._v_avg
    
    @property
    def state0(self):
        """
        Returns the rotameric state for each sidechain and rotamer

        Returns
        -------
        None.

        """
        if self._state0 is None:
            state=[]
            for v,vref0,mult in zip(self.vt,self.v_avg,self.mult):
                state.append(np.zeros(v.shape[1:],dtype=int))
                for q in [2,3]:
                    i=mult==q
                    overlap=[]
                    for m in range(q):
                        angle=2*np.pi*m/q
                        vref=np.concatenate(([vref0[0,i]*np.cos(angle)-vref0[1,i]*np.sin(angle)],
                                             [vref0[1,i]*np.sin(angle)+vref0[0,i]*np.cos(angle)]),
                                            axis=0)
                        overlap.append(v[0,i].T*vref[0]+v[1,i].T*vref[1])
                        
                    state[-1][i]=np.argmax(overlap,axis=0).T
                    
            self._state0=state
        return self._state0
    
    @property
    def state(self):
        """
        Returns an index for each residue and timepoint, indicating the rotameric
        state for that residue

        Returns
        -------
        state : TYPE
            DESCRIPTION.

        """
        if self._state is None:
            x=[np.ones(self.N,dtype=int)]
            for mult,i in zip(self.mult[:-1],self.index[3:]):
                x.append(copy(x[-1]))
                x[-1][i]*=mult
                
            state=np.zeros([self.N,self.state0[0].shape[1]],dtype=int)
            for state0,x0,i in zip(self.state0,x,self.index[3:]):
                state[i]+=(x0[i]*state0.T).T
            self._state=state
            
        return self._state
    
    @property
    def CCstate(self):
        """
        Returns the states for all pairs of residues, which can then be used in
        CC analysis

        Returns
        -------
        np.array

        """
        if self._CCstate is None:
            out=np.zeros([self.N,self.N,self.state.shape[1]],dtype=int)
            
            for k,(tm,state) in enumerate(zip(self.total_mult,self.state)):
                out[k]=state+tm*self.state
            self._CCstate=out
        return self._CCstate
            
    
    #%% Entropy calculations
    @property
    def Sres(self):
        """
        Returns the entropy of individual side chains (J*mol^-1*K^-1)

        Returns
        -------
        np.array

        """
        if self._Sres is None:
            S=np.zeros(self.N)
            state=self.state
            for k in range(self.total_mult.max()):
                p=(state==k).mean(-1)
                i=p>0
                S[i]-=p[i]*np.log(p[i])
                
            self._Sres=S*self.R
        return self._Sres
            
    @property
    def Scc(self):
        """
        Returns the pairwise entropy for sidechains (J*mol^-1*K^-1)

        Returns
        -------
        np.array

        """
        if self._Scc is None:
            S=np.zeros([self.N,self.N])
            state=self.CCstate
            Nt=self.state.shape[1]
            for k in range(self.N):
                for j in range(k,self.N):
                    p=np.unique(self.CCstate[k,j],return_counts=True)[1]/Nt
                    S[k,j]=-(p*np.log(p)).sum()
                    S[j,k]=S[k,j]
                
            self._Scc=S*self.R
        return self._Scc
    
    @property
    def CC(self):
        """
        Returns the correlation coefficients between residues, which varies
        between 0 and 1

        Returns
        -------
        np.array

        """
        if self._CC is None:
            bottom=np.zeros([self.N,self.N])
            bottom+=self.Sres
            bottom=bottom.T+self.Sres
            top=2*bottom-2*self.Scc
            i=top==0
            bottom[i]=1
            self._CC=top/bottom
        return copy(self._CC)
    
    @property
    def Smax(self):
        """
        Returns the maximum possible entropy for each sidechain, resulting
        from uniform distribution of orientations.

        Returns
        -------
        np.array

        """
        return np.log(self.total_mult)*self.R
        
    #%% Lagged CC
    def LaggedCC(self,i0:int,i1:int,Nt:int=1000,step=1):
        """
        

        Parameters
        ----------
        i0 : int
            DESCRIPTION.
        i1 : int
            DESCRIPTION.
        Nt : int, optional
            DESCRIPTION. The default is 1000.
        step : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        t : TYPE
            DESCRIPTION.
        out : TYPE
            DESCRIPTION.

        """
        out=np.zeros([2,Nt])
        for k,N0 in enumerate(np.arange(Nt)*step):
            if isinstance(i0,str) and i0.lower()=='pca':
                state0=self.PCA.Cluster.state[:(None if N0==0 else -N0)]
                mult=self.PCA.Cluster.n_clusters
            else:
                state0=self.state[i0,:(None if N0==0 else -N0)]
                mult=self.total_mult[i0]
            p=np.unique(state0,return_counts=True)[1]/len(state0)
            S0=-(p*np.log(p)).sum()
            
            if isinstance(i1,str) and i1.lower()=='pca':
                state1=self.PCA.Cluster.state[N0:]
            else:
                state1=self.state[i1,N0:]
            p=np.unique(state1,return_counts=True)[1]/len(state0)
            S1=-(p*np.log(p)).sum()
            
            state=state0+mult*state1
            p=np.unique(state,return_counts=True)[1]/len(state0)
            Scc=-(p*np.log(p)).sum()

            out[0,k]=2*(S0+S1-Scc)/(S1+S0)
            
        for k,N0 in enumerate(np.arange(Nt)*step):
            if isinstance(i0,str) and i0.lower()=='pca':
                state0=self.PCA.Cluster.state[N0:]
                mult=self.PCA.Cluster.n_clusters
            else:
                state0=self.state[i0,N0:]
                mult=self.total_mult[i0]
            p=np.unique(state0,return_counts=True)[1]/len(state0)
            S0=(p*np.log(p)).sum()
            
            if isinstance(i1,str) and i1.lower()=='pca':
                state1=self.PCA.Cluster.state[:(None if N0==0 else -N0)]
            else:
                state1=self.state[i1,:(None if N0==0 else -N0)]
            p=np.unique(state1,return_counts=True)[1]/len(state0)
            S1=(p*np.log(p)).sum()
            
            state=state0+mult*state1
            p=np.unique(state,return_counts=True)[1]/len(state0)
            Scc=(p*np.log(p)).sum()

            out[1,k]=2*(S0+S1-Scc)/(S1+S0)
            
        t=np.arange(Nt)*step*self.traj.dt/1e3
            
        return t,out
    
    def plot_lag(self,i0:int,i1:int,Nt:int=1000,step:int=1,ax=None):
        t,out=self.LaggedCC(i0=i0,i1=i1,Nt=Nt,step=step)
        
        if ax is None:ax=plt.subplots()[1]
        
        ax.plot(t,out.T)
        ax.set_xlabel(r'$\Delta$t / ns')
        ax.set_ylabel('C.C.')
        label0='PCA' if isinstance(i0,str) and i0.lower()=='pca' else f'{self.resids[i0]}{AA(self.resi[i0].resname).symbol}'
        label1='PCA' if isinstance(i1,str) and i1.lower()=='pca' else f'{self.resids[i1]}{AA(self.resi[i1].resname).symbol}'
        ax.legend((f'{label1} follows {label0}',f'{label0} follows {label1}'))
        
        return ax
        
    #%% vs PCA
    @property
    def Spca(self):
        """
        Return the entropy of states identified in PCA clustering

        Returns
        -------
        float

        """
        if self.PCA is None:return None
        
        p=np.unique(self.PCA.Cluster.state,return_counts=True)[1]/len(self.PCA.Cluster.state)
        return -self.R*(p*np.log(p)).sum()
    
    @property
    def CCpca(self):
        """
        Return correlation coefficient between PCA states and individual side
        chains

        Returns
        -------
        np.array

        """
        if self._CCpca is None:
            Spca=self.Spca
            S=self.Sres
            
            state=self.state*(self.PCA.Cluster.state.max()+1)+self.PCA.Cluster.state
            
            SCC=np.zeros(self.N)
            for k in range(state.max()):
                p=(state==k).mean(-1)
                i=p>0
                SCC[i]+=-self.R*(p[i]*np.log(p[i]))
            
            self._CCpca=2*(S+Spca-SCC)/(S+Spca)
        
        
        return self._CCpca
    
    def CCpca_states(self,states:list):
        """
        Evaluates the pca CC for specific states in the PCA

        Parameters
        ----------
        states : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        PCAstate=self.PCA.Cluster.state
        i=np.sum([PCAstate==state for state in states],axis=0).astype(bool)
        
        p=np.unique(PCAstate[i],return_counts=True)[1]/i.sum()
        
        Spca=-(p*np.log(p)).sum()  #PCA entropy
        
        CC=[]
        for state in self.state:
            p=np.unique(state[i],return_counts=True)[1]/i.sum()
            Ssc=-(p*np.log(p)).sum()
            
            Tstate=state*(PCAstate.max()+1)+PCAstate
            p=np.unique(Tstate[i],return_counts=True)[1]/i.sum()
            Stotal=-(p*np.log(p)).sum()
            
            CC.append(2*(Ssc+Spca-Stotal)/(Ssc+Spca))

        return np.array(CC)            
    
    def plotCCpca(self,index=None,states:list=None,ax=None,**kwargs):
        if ax is None:ax=plt.subplots()[1]
        if index is None:index=np.ones(len(self.CCpca),dtype=bool)
        
        if 'color' not in kwargs:kwargs['color']='black'
        ax.plot(self.CCpca[index],**kwargs)
        ax.xaxis.set_major_locator(plt.MaxNLocator(30,integer=True))
        ax.xaxis.set_major_formatter(self._axis_formatter(index))
        
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_ylabel('C.C')
        
        return ax
        

    
    #%% Plotting
    
    
    def plotCC(self,index=None,ax=None,CCsum:bool=True,**kwargs):
        """
        Make a CC plot for all residue pairs

        Parameters
        ----------
        index : np.array, optional
            Index of residues to include. The default is None (all residues).
        ax : matplotlib axis, optional
            Axis to use for plot. The default is None.
        **kwargs : TYPE
            Keyword arguments passed to plt.imshow

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        
            
        
        
        if index is None:index=np.ones(self.N,dtype=bool)
        if ax is None:
            ax=plt.subplots()[1]

        index=np.array(index)
        if index.size==1 and np.issubdtype(index.dtype,np.integer):
            ax.plot(self.CC[index],**kwargs)
            ax.xaxis.set_major_locator(plt.MaxNLocator(30,integer=True))
            ax.xaxis.set_major_formatter(self._axis_formatter(np.arange(len(self.resids))))
            
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_ylabel('C.C')
            return ax
            
        
        if 'cmap' not in kwargs:
            kwargs['cmap']='binary'
        CC=copy(self.CC)
        CC-=np.diag(np.diag(CC))
        npts=CC[index][:,index].shape[0]
        ax.imshow(CC[index][:,index],**kwargs)
        
        

        cmap0=plt.get_cmap('binary')
        cmap=cmap0(np.arange(cmap0.N))
        cmap[:,-1]=np.linspace(0,1,cmap0.N)
        cmap=ListedColormap(cmap)
        kwargs['cmap']=cmap
        
        ax.imshow(np.eye(npts),**kwargs)
        
        # plt.rcParams['figure.constrained_layout.use']=True
        
        
        if CCsum:
            x=CC[index].sum(0)
            x/=x[np.logical_not(np.isnan(x))].max()/30
            ax.plot(-x-npts/7,np.arange(npts),color='black',clip_on=False)
             
        hdl=ax.imshow(CC[index][:,index],**kwargs)
        ax2=ax.figure.add_subplot(3,1,3)
        ax2.set_position([.9,.125,.01,.85])
        
        hdl=ax.figure.colorbar(hdl,ax=ax,cax=ax2)
        hdl.set_label('C.C')
        # ax.set_position([.2,.125,.85,.85])
    
        ax.xaxis.set_major_formatter(self._axis_formatter(index))
        ax.yaxis.set_major_formatter(self._axis_formatter(index))
        ax.xaxis.set_major_locator(plt.MaxNLocator(30,integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(30,integer=True))
        ax.tick_params(axis='x', labelrotation=90)
        # ax.set_aspect('auto')
        if CCsum:
            ax.set_position([.12,.125,.85,.85])
        else:
            ax.set_position([.05,.125,.85,.85])
        
        return ax
    
    def plotS(self,index=None,Smax:bool=True,ax=None,**kwargs):
        """
        Plot the entropy for each residue. By default, also shows the max
        entropy for each residue

        Parameters
        ----------
        index : np.array, optional
            Index of residues to include. The default is None (all residues).
        Smax : bool, optional
            Include a plot of the maximum S. The default is True.
        ax : matplotlib axis, optional
            Axis to use for plot. The default is None.
        **kwargs : TYPE
            Keyword arguments passed to plt.bar

        Returns
        -------
        None.

        """
        if index is None:index=np.ones(self.N,dtype=bool)
        if ax is None:ax=plt.subplots()[1]
        
        color=kwargs.pop('color') if 'color' in kwargs else 'steelblue'
        
        if Smax:
            x=self.Smax[index]
            ax.bar(np.arange(len(x)),x,color='silver',**kwargs)
        x=self.Sres[index]
        ax.bar(np.arange(len(x)),x,color=color,**kwargs)
        ax.xaxis.set_major_formatter(self._axis_formatter(index))
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_ylabel(r'S$_{res}$ / J*mol$^{-1}$*K$^{-1}$')
        
        return ax
    
    def _axis_formatter(self,index):
        label=[]
        for resi in self.resi[index]:
            if resi.resname.capitalize() in AA.codes:
                label.append(f'{AA(resi.resname).symbol}{resi.resid}')
            else:
                label.append(f'X{resi.resid}')
            
        
        def format_func(value,tick_number):
            if int(value)>=len(label) or int(value)<0:return ''
            return label[int(value)]
        
        return plt.FuncFormatter(format_func)
    
    
    def plotChi(self,index:int,ax:list=None,step:int=1,cmap='tab10'):
        """
        Creates one or more Ramachandran histogram plots, depending on the number
        of chi angles, i.e. 1D histogram for Valine, 1 Ramachandran plot for
        Isoleucine, 2 plots for Glutamine, etc.

        Parameters
        ----------
        index : int
            DESCRIPTION.

        Returns
        -------
        fig

        """
        chi=self.Chi[index]
        
        index=np.array(index)
        if index.dtype==bool:index=np.argmax(index)
        
        N=self.index[3:,index].sum()
        nplots=max([1,N-1])
        if ax is None:
            fig,ax=plt.subplots(1,nplots)
        ax=np.atleast_1d(ax).flatten()
        assert len(ax)==nplots,f"Residue {self.resid[index].resname}{self.resid[index].resid} has {N} chi angles, and therefore requires {nplots} plots"
    
        # chi=[self.chi[k][self.index[k+3,:index].sum()][::step] for k in range(N-1,-1,-1)]
        
        
        
        nstates=self.total_mult[index]
        if isinstance(cmap,str):cmap=plt.get_cmap(cmap)
        
        for a,k in zip(ax,range(N-1)):
            for m in range(nstates):
                i=self.state[index][::step]==m
                a.scatter(chi[k][i],chi[k+1][i],color=cmap(m),s=1)
            a.set_xlim([0,360])
            a.set_ylim([0,360])
            a.set_xlabel(rf'$\chi_{k+1}$ / $^\circ$')
            a.set_ylabel(rf'$\chi_{k+2}$ / $^\circ$')
        
        fig.tight_layout()        
        return ax
    
            
    
#%% Chimera functions
    def chimera(self,index=None,scaling:float=None,norm:bool=True,color=[1,0,0,1]):
        """
        Plots the entopy (or normalized entropy, relative to max possible entropy)
        of each side chain onto the molcule

        Parameters
        ----------
        index : np.array, optional
            Index which sidechains to show. The default is None.
        scaling : float, optional
            Scale the radii from the entropy. The default is None.
        norm : bool, optional
            Normalize relative to the maximum possible entropy. The default is True.
        color : tuple, optional
            Color of spheres. The default is [1,0,0,1].

        Returns
        -------
        None.

        """
        
        if norm:
            n=copy(self.Smax)
            n[n==0]=1
            x=self.Sres/n
        else:
            x=copy(self.Sres)
        x[np.isnan(x)]=0

        if scaling is None:scaling=1/x.max()
        
        x*=scaling
        
        self.select.chimera(x=x,index=index,color=color)
        
        
        
        CMXRemote=clsDict['CMXRemote']
        if self.project is not None:
            ID=self.project.chimera.CMXid
            if ID is None:
                self.project.chimera.current=0
                ID=self.project.chimera.CMXid
        else:
            ID=CMXRemote.launch()
        mn=CMXRemote.valid_models(ID)[-1]   
        CMXRemote.send_command(ID,f'~show #{mn}')
        CMXRemote.send_command(ID,f'ribbon #{mn}')
        CMXRemote.send_command(ID,f'show #{mn}&~@H*&~:GLY,ALA,PRO')
        
            
    def CCchimera(self,index=None,indexCC:int=None,states:list=None,scaling:float=None,norm:bool=True) -> None:
        """
        Plots the cross correlation of motion.
    
        Parameters
        ----------
        index : list-like, optional
            Select which residues to plot. The default is None (all residues).
        indexCC : int,optional
            Select which row of the CC matrix to show. Must be used in combination
            with rho_index. Note that if index is also used, indexCC is applied
            AFTER index.
            
            Set indexCC to 'PCA' to correlate with the backbone
        scaling : float, optional
            Scale the display size of the detectors. If not provided, a scaling
            will be automatically selected based on the size of the detectors.
            The default is None.
        norm : bool, optional
            Normalizes the data to the amplitude of the corresponding CC coeffiecent 
            (makes diagonal of CC matrix equal to 1).
            The default is True
    
        Returns
        -------
        None
    
        """
        
        CMXRemote=clsDict['CMXRemote']
    
        index=np.arange(self.resids.size) if index is None else np.array(index)
    
        if isinstance(indexCC,str) and indexCC.lower()=='pca':        
            if states is not None:
                x=self.CCpca_states(states)[index]
            else:
                x=self.CCpca[index]
            x[np.isnan(x)]=0
            x *= 1/x.max() if scaling is None else scaling
        else:
            x=self.CC[index][:,index]
            x[np.isnan(x)]=0
            x[x<0]=0
            x -=np.eye(x.shape[0])
            x *= 1/x.max() if scaling is None else scaling
            x+=np.eye(x.shape[0])
        
        
    
        if self.project is not None:
            ID=self.project.chimera.CMXid
            if ID is None:
                self.project.chimera.current=0
                ID=self.project.chimera.CMXid
        else:
            ID=CMXRemote.launch()
    
    
        ids=np.array([s.atoms.indices for s in self.resi[index]],dtype=object)
    
    
    
        # CMXRemote.send_command(ID,'close')
    
    
        
        if indexCC is not None:
            if not(isinstance(indexCC,str) and indexCC.lower()=='pca'):
                x=x[indexCC].squeeze()
            else:
                indexCC=None
            x[np.isnan(x)]=0
            self.select.chimera(x=x,index=index)

            
            
            mn=CMXRemote.valid_models(ID)[-1]
            if indexCC is not None:
                sel0=self.select.repr_sel[index][indexCC]
                if hasattr(sel0,'size'):sel0=sel0[0]  #Depending on indexing type, we may still have a numpy array here
                CMXRemote.send_command(ID,'color '+'|'.join([f'#{mn}/{s.segid}:{s.resid}@{s.name}' for s in sel0])+' black')
            
            CMXRemote.send_command(ID,f'~show #{mn}')
            CMXRemote.send_command(ID,f'ribbon #{mn}')
            CMXRemote.send_command(ID,f'show #{mn}&~@H*&~:GLY,ALA,PRO')
            CMXRemote.send_command(ID,'set bgColor gray')
            # print('color '+'|'.join(['#{0}/{1}:{2}@{3}'.format(mn,s.segid,s.resid,s.name) for s in sel0])+' black')
        else:

            self.select.chimera()
            mn=CMXRemote.valid_models(ID)[-1]
            CMXRemote.send_command(ID,f'color #{mn} tan')
            CMXRemote.send_command(ID,f'~show #{mn}')
            CMXRemote.send_command(ID,f'ribbon #{mn}')
            CMXRemote.send_command(ID,f'show #{mn}&~@H*&~:GLY,ALA,PRO')
            CMXRemote.send_command(ID,'set bgColor gray')
                        
            out=dict(R=x,ids=ids)
            CMXRemote.add_event(ID,'CC',out)
        
        
        if self.project is not None:
            self.project.chimera.command_line(self.project.chimera.saved_commands)
            
    def save(self,filename:str,overwrite:bool=False):
        """
        Save the EntropyCC object to a file. This will store the states for 
        reanalysis. Re-accessing the vectors will require a full reload. The project
        object is also not saved, but reloading will create a new dummy project
        (Entropy data is not stored in projects, so the project function only
         provides chimera functionality)
        
        Reloading is performed by calling EntropyCC with the filename rather than
        the selection object.
    
        Parameters
        ----------
        filename : str
            Storage location.
        overwrite : bool
            Overwrite existing file
    
        Returns
        -------
        None.
    
        """
        if not(overwrite):
            assert not(os.path.exists(filename)),'File already exists. Set overwrite=True to save anyway'
        with open(filename,'wb') as f:
            write_EntropyCC(f,self)
        

class Chi():
    def __init__(self,ECC):
        self.ECC=ECC
        
    def __getitem__(self,index):
        ECC=self.ECC
        index=np.array(index)
        if index.dtype==bool:index=np.argmax(index)
         
        N=ECC.index[3:,index].sum()

        return np.array([ECC.chi[k][ECC.index[k+3,:index].sum()] for k in range(N-1,-1,-1)])
    
            
# Tools for selecting the correct atoms to get rotamers
                


def get_chain(resi=None,atom=None,sel0=None,exclude=None):
    """
    Returns the chain of atoms required to calculate rotamers for a given residue

    Parameters
    ----------
    resi : MDanalysis residue
        Residue for desired rotamers

    Returns
    -------
    atom group
        DESCRIPTION.

    """
    
    if resi is not None:
        resname=resi.resname
        sel0=resi.atoms-resi.atoms[[name[0]=='H' for name in resi.atoms.names]]
        i=sel0.names==sel_names[resname][0] #This is one bond away from our starting point
        exclude=[sel0[i][0]]
        i=sel0.names==sel_names[resname][1] #This is our starting point
        atom=sel0[i][0]
    
    
    if exclude is None:exclude=[]
    '''searching a path from a methyl group of a residue down to the C-alpha of the residue
    returns a list of atoms (mda.Atom) beginning with the Hydrogens of the methyl group and continuing
    with the carbons of the side chain
    returns empty list if atom is not a methyl carbon'''
    final=False
    def get_bonded():
        '''it happens, that pdb files do not contain bond information, in that case, we switch to selection
        by string parsing'''
        return np.sum(find_bonded([atom],sel0,n=4,d=2.1))
    
    a_name=atom.name.lower()
    if 'c'==a_name and len(exclude):
        return [atom]
    elif a_name == "n":
        return []
    connected_atoms = []
    bonded = get_bonded()
    if len(exclude)==1:
        if resi is not None:
            final=True
            # if not "c"==a_type:
            #     return []
        else:
            return []
    connected_atoms.append(atom)
    exclude.append(atom)
    for a in bonded:
        if not a in exclude:
            nxt = get_chain(atom=a,sel0=sel0,exclude=exclude)
            for b in nxt:
               connected_atoms.append(b)
    if len(connected_atoms)>1:
        if final:
            return exclude[0]+np.sum(connected_atoms)
        else:
            return connected_atoms
    else:
        return []  


def find_bonded(sel,sel0=None,exclude=None,n=4,sort='dist',d=1.65):
    """
    Finds bonded atoms for each input atom in a given selection. Search is based
    on distance. Default is to define every atom under 1.65 A as bonded. It is 
    recommended to also provide a second selection (sel0) out of which to search
    for the bound atoms. If not included, the full MD analysis universe is searched.
    
    Note- a list of selections is returned. Sorting may be determined in one
    of several ways (set sort)
        'dist':     Sort according to the nearest atoms
        'mass':     Sort according to the largest atoms first
        'massi':    Sort according to smallest atoms first (H first)
        'cchain':   Sort, returing C atoms preferentially (followed by sorting by mass)
    
    One may also exclude a set of atoms (exclude), which then will not be returned
    in the list of bonded atoms. Note that exclude should be a list the same
    size as sel (either a selection the same size as sel, or a list of selections
    with a list length equal to the number of atoms in sel)
    """
    
    if not(hasattr(sel,'__len__')):sel=[sel]
    
    out=[sel[0].universe.atoms[0:0] for _ in range(n)]  #Empty output
    
    if sel0 is None:
        sel0=sel[0].universe
    
    for m,s in enumerate(sel):
        sel01=sel0.select_atoms('point {0} {1} {2} {3}'.format(*s.position,d))
        sel01=sel01-s #Exclude self
        if exclude is not None:
            sel01=sel01-exclude[m]
        if sort[0].lower()=='d':
            i=np.argsort(((sel01.positions-s.position)**2).sum(axis=1))
        elif sort[0].lower()=='c':
            C=np.array([s.name[0]=='C' for s in sel01])
#            C=sel01.type=='C'
            nC=np.logical_not(C)
            i1=np.argsort(sel01[nC].masses)[::-1]
            C=np.argwhere(C)[:,0]
            nC=np.argwhere(nC)[:,0]
            i=np.concatenate((C,nC[i1]))
        elif sort.lower()=='massi':
            i=np.argsort(sel01.masses)
        else:
            i=np.argsort(sel01.masses)[::-1]
        sel01=sel01[i]
        for k in range(n):
            if len(sel01)>k:
                out[k]+=sel01[k]
            else:
                #Append self if we don't find enough bound atoms
                out[k]+=s #Why do we do this? Here we add the original selection where no bonds are found....very strange, I think.
                #Apparently, this breaks find_methyl without the above line.
                # pass           
    return out           


def CombineEntropy(*ECC):
    """
    Takes multiple EntropyCC objects and combines them into one new EntropyCC 
    object which evaluates the entropy across all objects. This is useful in
    case the entropy/correlation across one state that occurs in different parts
    of the trajectory should be combined for  evaluation.

    Selection should be the same for all objects. An error will be thrown if
    residue numbers do not match

    Parameters
    ----------
    *ECC : EntropyCC objects
        Objects to be combined.

    Returns
    -------
    EntropyCC
        Combined object

    """
    
    
    
    assert np.all([len(ECC[0].resi)==len(ECC0.resi) for ECC0 in ECC[1:]]),"Residue selection must match for all EntropyCC objects"
    assert np.all([np.all([r0.resnum==r1.resnum for r0,r1 in zip(ECC[0].resi,ECC0.resi)]) for ECC0 in ECC[1:]])
    
    out=EntropyCC(ECC[0].select)
    
    out._vt=[np.concatenate([ECC0._vt[k] for ECC0 in ECC],axis=-1) for k in range(len(ECC[0]._vt))]
    
    return out