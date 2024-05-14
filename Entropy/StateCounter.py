# -*- coding: utf-8 -*-

import numpy as np
from .. import clsDict,Defaults
from ..MDtools import vft
from ..misc.tools import AA
from copy import copy
import matplotlib.pyplot as plt


sel_names={'ARG':['CZ','NE'],'HIS':['CD2','CG'],'HSD':['CD2','CG'],'LYS':['NZ','CE'],
           'ASP':['CG','CB'],'GLU':['CD','CG'],'SER':['OG','CB'],'THR':['CG2','CB'],
           'ASN':['OD1','CG'],'GLN':['OE1','CD'],'CYS':['SG','CB'],'VAL':['CG1','CB'],
           'ILE':['CD','CG1'],'LEU':['CD1','CG'],'MET':['CE','SD'],'PHE':['CG','CB'],
           'TYR':['CG','CB'],'TRP':['CD1','CG']}

sel_mult={'ARG':[3,3,3,3],'HIS':[2,3],'HSD':[2,3],'LYS':[3,3,3,3],
          'ASP':[3],'GLU':[3,3],'SER':[3],'THR':[3],
          'ASN':[2,3],'GLN':[2,3,3],'CYS':[3],'VAL':[3],
          'ILE':[3,3],'LEU':[3,3],'MET':[3,3,3],'PHE':[3],'TYR':[3],'TRP':[2,3]}


class StateCounter:
    R=8.31446261815324 #J/mol/K
    
    def __init__(self,select):
        self.select=select
        
        self.reset()
        
        
    def reset(self):
        """
        Sets various fields back to unitialized state

        Returns
        -------
        self

        """
        self._resi=None
        self._sel=None
        self._index=None
        self._FrObj=None
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
        
        return self
        
    @property
    def resi(self):
        if self._resi is None:
            if self.select.sel1 is None:return None
            
            self._resi=[]
            for s in self.select.sel1:
                self._resi.append(s.residues[0])
                
            self._resi=np.array(self._resi,dtype=object)
            
        return self._resi
    
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
                
                for q in [2,3]:
                    for angle in [m*360/q for m in range(1,q)]:
                        i=(np.logical_and(chi.T>=angle,mult==q)).T
                        v0=v[0,i]*np.cos(2*np.pi/q)+v[1,i]*np.sin(2*np.pi/q)
                        v1=-v[0,i]*np.sin(2*np.pi/q)+v[1,i]*np.cos(2*np.pi/q)
                        v[0,i]=v0
                        v[1,i]=v1
                v=v.mean(-1)
                v/=np.sqrt(v[0]**2+v[1]**2) #Renormalization
                out.append(v)
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
            self._CC=top/bottom
        return self._CC
    
    #%% Plotting
    def plotCC(self,ax=None,**kwargs):
        
        if ax is None:ax=plt.subplots()[1]
        
        if 'cmap' not in kwargs:
            kwargs['cmap']='cool'
        CC=copy(self.CC)
        CC-=np.diag(np.diag(CC))
        hdl=ax.imshow(CC,**kwargs)
        ax.figure.colorbar(hdl,ax=ax)
        kwargs['cmap']='binary'
        # ax.imshow(np.eye(self.N),**kwargs)
        
        label=[]
        for resi in self.resi:
            if resi.resname.capitalize() in AA.codes:
                label.append(f'{AA(resi.resname).symbol}{resi.resid}')
            else:
                label.append(f'X{resi.resid}')
            
        
        def format_func(value,tick_number):
            if int(value)>=len(label) or int(value)<0:return ''
            return label[int(value)]
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.tick_params(axis='x', labelrotation=90)
        
        return ax
        
        
        
    
            
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
        sel0=resi.atoms
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
        return np.sum(find_bonded([atom],sel0,n=4,d=1.9))
    
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