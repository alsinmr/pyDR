#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:09:27 2022

@author: albertsmith

iRED analysis is based on the work of Prompers and Brueschweiler:
    
    Jeanine J. Prompers, Rafael Brueschweiler. "Reorientational Eigenmode Dynamics: 
    A Combined MD/NMR Relaxation Analysis Method for Flexible Parts in Globular 
    Proteins." J. Am. Chem. Soc. (2001) 123, 7305-7313
    https://www.doi.org/10.1021/ja012750u
    
    and
    
    Jeanine J. Prompers, Rafael Brueschweiler. "General Framework for Studying 
    the Dynamics of Folded and Nonfolded Proteins by NMR Relaxation Spectroscopy 
    and MD Simulation" J. Am. Chem. Soc. (2002) 124, 4522-4534
    https://www.doi.org/10.1021/ja0107226
    
We have repurposed it for correlation analysis of detector responses:
    
    Albert A. Smith, Matthias Ernst, Sereina Riniker, Beat H. Meier. "Localized 
    and Collective Motions in HET-s(218-289) Fibrils from Combined NMR Relaxation 
    and MD Simulation." Angew. Chem. Int. Ed. (2019) 131, 9483-9488 
    https://www.doi.org/10.1002/anie.201901929
"""

import numpy as np
from pyDR.MDtools.Ctcalc import sparse_index,get_count,Ctcalc
from pyDR import Data #Note, this means iRED needs to get imported to pyDR after Data
from pyDR import clsDict,Defaults
from pyDR.Data.Plotting import subplot_setup
import matplotlib.pyplot as plt

dtype=Defaults['dtype']

class iRED():
    """
    Object for managing iRED analysis of 
    """
    def __init__(self,vec,rank=2):
        self.rank=rank
        self._vec=vec
        self.source.Type='iREDmode'
        
        
    
    def __setattr__(self,name:str,value:object):
        """
        Override normal attribute setting behavior
        ->Delete stored calculations if rank is changed.

        Parameters
        ----------
        name : str
            Object attribute name.
        value : object
            Objection attribute value.

        Returns
        -------
        None.

        """
        
        if name=='rank':
            assert value in [1,2],"Rank should be 1 or 2"
            if not(hasattr(self,'rank')) or self.rank!=value:
                super().__setattr__('rank',value)
                self._M=None
                self._lambda=None
                self._m=None
                self._Aqt=None
                self._Ylm=None
                self._t=None
                self._Cqt=None
                self._CqtInf=None
                self._Ct=None
                self._CtInf=None
                self._DelCt=None
            return
        super().__setattr__(name,value)
        
    @property
    def t0(self) -> np.ndarray:
        """
        Time axis corresponding to stored vectors (self.v, self.Aqt). Note this
        may be different than the time axis corresponding to correlation functions,
        if sparse sampling applied.
                                

        Returns
        -------
        np.ndarray
        Array of time points in ps corresponding to stored vectors
        """
        return self._vec['t']/1e3
    
    @property
    def t(self) -> np.ndarray:
        """
        Time axis corresponding to correlation functions. Note that this may
        be different than the time axis corresponding to stored vectors (t0), in
        case sparse sampling has been applied.

        Returns
        -------
        np.ndarray
        Array of time points corresponding to correlation functions

        """
        if self._t is None:
            N=get_count(self.index)
            i=N!=0
            N=N[i]
            dt=self.sampling_info['dt']
            self._t=(np.cumsum(i)-1)*dt/1e3
        return self._t
    
    @property
    def sampling_info(self) -> dict:
        """
        Returns information about sampling of the MD trajectory

        Returns
        -------
        dict
            Dictionary containing final frame (tf), time step (dt/ps), n and nr
            which are used if sparse sampling implemented.
        """
        return self._vec['sampling_info']
    
    @property
    def v(self) -> np.ndarray:
        """
        Returns the vectors for performing iRED analysis

        Returns
        -------
        np.ndarray
        3xnrxnt array of vectors to be analyzed

        """
        return self._vec['v']
    @property
    def index(self) -> np.ndarray:
        """
        Index providing the sampling schedule for the stored vectors 

        Returns
        -------
        np.ndarray
        Array of indices giving the sampling schedule
        """
        return self._vec['index']
    @property
    def source(self) -> np.ndarray:
        """
        Returns the source info for the iRED object

        Returns
        -------
        Source object.
        """
        return self._vec['source']
    
    @property
    def M(self) -> np.ndarray:
        """
        Returns the iRED matrix for the input vectors

        Returns
        -------
        Square iRED matrix.
        """
        if self._M is None:self._M=Mmat(self.v,self.rank)
        return self._M
    
    @property
    def Lambda(self) -> np.ndarray:
        """
        Returns the eigenvalues of the M matrix

        Returns
        -------
        1D array

        """
        if self._lambda is None:
            self._lambda,self._m=np.linalg.eigh(self.M)
        return self._lambda
    
    @property
    def m(self) -> np.ndarray:
        """
        Returns the eigenvectors of the M matrix

        Returns
        -------
        2D array
            Square matrix where columns corresponds to eigenvectors of M.

        """
        if self._m is None:self.Lambda
        return self._m
    
    @property
    def Ylm(self) -> dict:
        """
        Returns the spherical tensor components of the individual bond vectors

        Returns
        -------
        dict
            Dictionary with components of the rank-1 or rank-2 tensor components.

        """
        if self._Ylm is None:self._Ylm=Ylm(self.v,self.rank)
        return self._Ylm
    
    @property
    def Aqt(self) -> dict:
        """
        Returns the projections of the spherical components onto the eigenbasis
        of the M matrix, thus yielding the time dependence of the individual
        eigenmodes.

        Returns
        -------
        dict
            Dictionary of the time trajectory of the spherical components of
            the eigenmodes
        """
        if self._Aqt is None:
            aqt = {}
            for k,ylm in self.Ylm.items():
                aqt[k] = self.m.T@ylm
            self._Aqt=aqt
        return self._Aqt
    
    @property
    def Cqt(self) -> dict:
        """
        Returns the correlation functions for the Aqt

        Returns
        -------
        dict
            Dictionary of the time correlation functions of the Aqt.

        """
        if self._Cqt is None:
            cqt = {}
            cqtInf={}
            ctc=Ctcalc(index=self.index)
            for k,aqt in self.Aqt.items():
                ctc.reset()
                ctc.a=np.real(aqt)
                ctc.add()
                ctc.a=np.imag(aqt)
                ctc.add()
                cqt[k],cqtInf[k]=ctc.Return()
                
            self._Cqt=cqt
            self._CqtInf=cqtInf
        return self._Cqt
    
    @property
    def Ct(self) -> np.ndarray:
        """
        Returns the total correlation functions for the eigenmodes

        Returns
        -------
        np.ndarray

        """
        
        if self._Ct is None:
            ct=np.zeros([self.M.shape[0],self.t.shape[0]])
            for k,cqt in self.Cqt.items():
                ct+=cqt*(1 if '0' in k else 2)
            self._Ct = ct
        return self._Ct
    
    @property
    def CqtInf(self) -> np.ndarray:
        """
        Returns the equilibrium value of the correlation functions

        Returns
        -------
        np.ndarray

        """
        if self._CqtInf is None:
            self.Cqt
        return self._CqtInf
    
    @property
    def CtInf(self) -> np.ndarray:
        """
        Returns the equilibrium values of the correlation functions

        Returns
        -------
        np.ndarray

        """
        if self._CtInf is None:
            ctinf=np.zeros([self.M.shape[0]])
            for k,cqtinf in self.CqtInf.items():
                ctinf+=cqtinf*(1 if '0' in k else 2)
            self._CtInf=ctinf
        return self._CtInf
    
    @property
    def DelCt(self) -> np.ndarray:
        """
        Returns the normalized correlation functions which equilibrate at zero

        Returns
        -------
        np.ndarray.

        """
        if self._DelCt is None:
            self._DelCt=self.Ct.T-self.CtInf
            self._DelCt/=self._DelCt[0]
            self._DelCt=self._DelCt.T
        return self._DelCt
    
    def iRED2data(self) -> Data:
        """
        Creates a data object - technically a Data_iRED object – which can be
        analyzed with detectors to get the detector responses for the individual
        iRED modes

        Returns
        -------
        Data
            DESCRIPTION.

        """
        
        out=Data_iRED(sens=clsDict['MD'](t=self.t))
        out.source=self.source
        
        out.label=np.arange(self.M.shape[0])
        out.R=self.DelCt
        out.Rstd=np.repeat(np.array([out.sens.info['stdev']],dtype=dtype),self.M.shape[0],axis=0)
        out.iRED={'M':self.M,'m':self.m,'Lambda':self.Lambda,'rank':self.rank}
        return out
        
        
            
                   
class Data_iRED(Data):
    def __init__(self,CC=None,totalCC=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.iRED={}
        self.CC=CC
        self.totalCC=totalCC
        self._CCnorm=None
        self._totalCCnorm=None
        
    def fit(self,bounds:bool=True,parallel:bool=False):
        out=super().fit(bounds=bounds,parallel=parallel)
        out.iRED=self.iRED
        return out
    
    @property
    def select(self):
        """
        Returns the selection if iRED in bond mode

        Returns
        -------
        molselect
            Selection object.

        """
        if 'mode' in self.source.Type:
            return None
        return self.source.select
    
    @property
    def CCnorm(self) -> np.ndarray:
        """
        Calculates and returns the normalized cross-correlation matrices for
        each detector

        Returns
        -------
        np.ndarray

        """
        if self.CC is None:
            print('Warning: Cross-correlation not calculated for this data object')
            return

        if self._CCnorm is None:
            self._CCnorm=np.zeros(self.CC.shape,dtype=dtype)
            for k,CC in enumerate(self.CC):
                dg=np.sqrt([np.diag(CC)])
                self._CCnorm[k]=CC/(dg.T@dg)
        return self._CCnorm
    
    @property
    def totalCCnorm(self) -> np.ndarray:
        """
        Calculates and returns the normalized total cross-correlation

        Returns
        -------
        np.ndarray

        """
        if self.totalCC is None:
            print('Warning: Cross-correlation not calculated for this data object')
            return
        if self._totalCCnorm is None:
            dg=np.sqrt(np.diag(self.totalCC))
            self._totalCCnorm=self.totalCC/(dg.T@dg)
        return self._totalCCnorm
    
    def modes2bonds(self,inclOverall:bool=False,calcCC='auto') -> Data:
        """
        Converts iRED mode detector responses into bond-specific detector 
        responses, including calculation of cross-correlation matrices for each 
        detector. These are stored in CC and CCnorm, where CC is the unnormalized
        correlation and CCnorm is the correlation coefficient, i.e. Pearson's r

        Parameters
        ----------
        inclOverall : bool, optional
            Determines whether to include the 3 or 5 overall modes of motion
            (depends on rank). The default is False
        calcCC : bool, optional
            Usually modes2bonds is run after fitting the modes with detectors.
            Then, cross correlation is only calculated for a few detectors. 
            However, if run before fitting, then a large number of cross
            correlation terms would be calculated. Therefore, by default, we only
            calculate CC if there are less than 40 different detectors/time points.
            Override this behavior by setting to True or False
            The default is 'auto'.

        Returns
        -------
        data_iRED

        """
        assert 'Lambda' in self.iRED.keys(),'This iRED data object has already been converted to bonds'
        
        out=self.__class__(sens=self.sens,src_data=self)
        out.details.append('Converted from iRED modes to iRED bond data')
        out.source.Type=self.source.Type.replace('mode','bond') #Mode analysis converted to bond analysis
        out.source.n_det=self.source.n_det
        if self.R.shape[0]==self.source.select.label.shape[0]:
            out.label=self.source.select.label #Use default label selections
        else:
            out.label=self.label
            print("Warning: Shape of selection does not match data shape. Labels will simply be numbered from 0")
        
        if hasattr(calcCC,'lower') and calcCC.lower()=='auto':
            calcCC=self.R.shape[1]<40 #Don't calculate the cross-correlation for large data sets
            
        m=self.iRED['m']
        Lambda=self.iRED['Lambda']
        rank=self.iRED['rank']
        
        if not(inclOverall): #Eliminate overall modes
            m=m[:,:-(2*rank+1)]
            Lambda=Lambda[:-(2*rank+1)]
        
        out.R=np.zeros(self.R.shape,dtype=dtype)
        out.Rstd=np.zeros(self.R.shape,dtype=dtype)
        
        for k,(rho,std) in enumerate(zip(self.R.T,self.Rstd.T)):
            out.R[:,k]=(Lambda*(m**2)*rho[:Lambda.shape[0]]).sum(axis=1)
            out.Rstd[:,k]=np.sqrt((Lambda*m**2*std[:Lambda.shape[0]]**2).sum(axis=1))
        
        if calcCC:
            out.totalCC=np.zeros([out.R.shape[0],out.R.shape[0]],dtype=dtype)
            out.CC=np.zeros([out.R.shape[1],out.R.shape[0],out.R.shape[0]],dtype=dtype)
            for m0,l0,rho_m in zip(m.T,Lambda,self.R): #Loop over all eigenmodes
                mat=(np.array([m0]).T@np.array([m0]))*l0
                out.totalCC+=mat
                for k,rho_m0 in enumerate(rho_m):
                    out.CC[k]+=mat*rho_m0
            
        
        if self.source.project is not None:self.source.project.append_data(out)
            
        return out
    
    def plot_CC(self,rho_index:int=None,norm:bool=True,abs_val:bool=True,color=None,ax:plt.Axes=None,**kwargs) -> plt.Axes:
        """
        Plots the cross-correlation between bonds for iRED cross correlation
        data. Only applies to iRED data converted to bonds

        Parameters
        ----------
        rho_index : int, optional
            Specifies which detector window to plot. The default None plots the
            total correlation (not timescale specific)
        norm : bool, optional
            Determines whether to plot normalized data. The default is True.
        abs_val : bool, optional
            Determines whether to plot the absolute value of the correlation.
            The default is True.
        ax : plt.Axes, optional
            Matplotlib Axes object in which to plot the correlation. The default is None.
        kwargs : type, optional
            Use to pass arguments to imshow in Matplotlib

        Returns
        -------
        plt.Axes

        """
        
        #TODO we might want to fill in empty space where integer-labeled data skips values
        #TODO we should also consider adjusting the behavior of the labels for integer-labeled data
        #since currently, the first and last values always appear on the plot labels
        #(ex, say we start with 11, then even if all the other labels appear at mutliples
        # of 10, the 11 will show up anyway)
        
        assert hasattr(self,'CCnorm'),"No CC data present in data object"
        
        if hasattr(rho_index,'lower') and rho_index.lower()=='all':
            ax,*_=subplot_setup(self.CCnorm.shape[0])
            for k,a in enumerate(ax):
                self.plot_CC(rho_index=k,norm=norm,abs_val=abs_val,color=color,ax=a,**kwargs)
            return ax
            
            
        
        if ax is None:ax=plt.figure().add_subplot(111)
        
        if rho_index is None:
            CC=self.totalCCnorm-np.eye(self.totalCC.shape[0]) if norm else self.totalCC
        else:
            CC=self.CCnorm[rho_index]-np.eye(self.totalCC.shape[0]) if norm else self.CC[rho_index]
        
        if abs_val:
            # CC/=CC.max()
            if color is None:
                color=plt.get_cmap("tab10")(rho_index) if rho_index is not None else (0,0,0,1)
            color=np.array([1+(c-1)*np.abs(CC) for c in color]).T
            color[color>1]=1
            color[color<0]=0
        else:
            cp,cm=(1,0,0,1),(0,0,1,1)
            color=np.zeros([*CC.shape,4])
            for k in range(4):  
                color.T[k].T[CC>0]=1+(cp[k]-1)*CC[CC>0]
                color.T[k].T[CC<=0]=1-(cm[k]-1)*CC[CC<=0]
            color[color>1]=1
            color[color<0]=0
        ax.imshow(color)
        if norm:
            im=[np.ones([color.shape[0],color.shape[0]])-np.eye(color.shape[0]) for k in range(3)]
            im.append(np.eye(color.shape[0]))
            im=np.array(im).T
            ax.imshow(im)
        
        def format_func(value,tick_number):
            if value>=self.label.shape[0]:value=self.label.shape[0]-1
            if value<0:value=0
            return str(self.label[int(value)])
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        if rho_index is not None:
            ax.set_title(r'Cross-correlation for $\rho_{}$'.format(rho_index))
        else:
            ax.set_title(r'Totall cross-correlation')
                
        return ax
        
    def CCchimera(self,index=None,rho_index:int=None,scaling:float=None) -> None:
        """
        Plots the cross correlation of motion for a given detector window in 
        chimera. 

        Parameters
        ----------
        index : list-like, optional
            Select which residues to plot. The default is None.
        rho_index : int, optional
            Select which detector to initiall show. The default is None.
            NOT IMPLEMENTED
        scaling : float, optional
            Scale the display size of the detectors. If not provided, a scaling
            will be automatically selected based on the size of the detectors.
            The default is None.

        Returns
        -------
        None

        """
        
        CMXRemote=clsDict['CMXRemote']

        index=np.arange(self.R.shape[0]) if index is None else np.array(index)

        if rho_index is None:rho_index=np.arange(self.R.shape[1])
        if not(hasattr(rho_index, '__len__')):
            rho_index = np.array([rho_index], dtype=int)
        # R = self.CCnorm[:,index,bond_index].T
        #TODO add some options for including the sign of the correlation (??)
        R=np.abs(self.CCnorm[:,index][:,:,index].T)
        R *= 1/R.T[rho_index].max() if scaling is None else scaling
        
        R[R < 0] = 0

        if self.source.project is not None:
            ID=self.source.project.chimera.CMXid
            if ID is None:
                self.source.project.chimera.current=0
                ID=self.source.project.chimera.CMXid
                print(ID)
        else: #Hmm....how should this work?
            ID=CMXRemote.launch()


        ids=np.array([s.indices for s in self.select.repr_sel[index]],dtype=object)



        # CMXRemote.send_command(ID,'close')
        CMXRemote.send_command(ID,'open "{0}"'.format(self.select.molsys.topo))
        nm=CMXRemote.how_many_models(ID)
        nm+=nm>1
        # CMXRemote.send_command(ID,'sel #{0}'.format(nm))
        CMXRemote.command_line(ID,'sel #{0}'.format(nm))

        CMXRemote.send_command(ID,'set bgColor gray')
        CMXRemote.send_command(ID,'style sel ball')
        CMXRemote.send_command(ID,'size sel stickRadius 0.2')
        CMXRemote.send_command(ID,'size sel atomRadius 0.8')
        CMXRemote.send_command(ID,'~ribbon')
        CMXRemote.send_command(ID,'show sel')
        CMXRemote.send_command(ID,'color sel tan')
        CMXRemote.send_command(ID,'~sel')


        out=dict(R=R,rho_index=rho_index,ids=ids)
        # CMXRemote.remove_event(ID,'Detectors')
        CMXRemote.add_event(ID,'DetCC',out)
        
        
                
        

            
                
        
            
            



#%% Core functions
def Mmat(vec, rank):
    #todo move inside class
    # M = np.eye(vec[:, 0, :].shape[0])*0
    M=np.eye(vec.shape[1])
    if rank == 2:
        for dim in range(3):
            for j in range(dim,3):
                # M += ((vec[:, j] * vec[:, dim])@(vec[:, j] * vec[:, dim]).T)*(1 if dim == j else 2)
                M += ((vec[j] * vec[dim])@(vec[j] * vec[ dim]).T)*(1 if dim == j else 2)
        M *= 3/2/vec.shape[-1]
        M -= 1/2
    elif rank == 1:
        #todo
        # test if this is correct
        M=np.eye(vec.shape[1])
        for v in vec:
            M+=v@v.T
        M/=vec.shape[-1]
        # M=(vec**2).sum(0).mean(-1)
        # for k in range(3):
        #     for j in range(k, 3):
        #         M += vec[k]@vec[j].T
    else:
        assert 0, "rank should be 1 or 2"
    return M

def Ylm(vec, rank):
    #todo move inside class
    X,Y,Z = vec
    Yl = {}
    if rank == 1:
        c = np.sqrt(3 / (2 * np.pi))
        Yl['1,0'] = c / np.sqrt(2) * Z
        a = (X + Y * 1j)
        b = np.sqrt(X ** 2 + Y ** 2)
        Yl['1,+1'] = -c / 2 * b * a
        # Yl['1,-1'] = Yl['1,+1'].conjugate() #Necessary?
    elif rank == 2:
        c = np.sqrt(15 / (32 * np.pi))
        Yl['2,0'] = c * np.sqrt(2 / 3) * (3 * Z ** 2 - 1)
        a = (X + Y * 1j)
        b = np.sqrt(X ** 2 + Y ** 2)
        Yl['2,+1'] = 2 * c * Z * b * a
        # Yl['2,-1'] = 2 * c * Z * b * a.conjugate() #Necessary?
        a *= a
        b *= b
        Yl['2,+2'] = c * b * a
        # Yl['2,-2'] = c * b * a.conjugate()  #Necessary?
    return Yl