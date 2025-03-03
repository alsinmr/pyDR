#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% Setup for google colab
import sys
import os
from time import sleep as _sleep

if 'google.colab' in sys.modules:
    #MDAnalysis setup
    cont=True
    install=True
    count=0
    while cont:
        try:
            import MDAnalysis as _
            cont=False
        except:
            if install:
                os.popen('pip3 install MDAnalysis')
                install=False
            else:
                _sleep(2)
                count+=1
                if count==60:
                    cont=False
                    print('Timeout on MDAnalysis installation (2 minutes)')
            
    
    # NGLviewer setup

    try:
        import nglview as _
    except:

        os.popen('pip install -q nglview')
        from google.colab import output as _
        _.enable_custom_widget_manager()

#%% Init file for all of pyDR

clsDict=dict()
from pyDR.Defaults import Defaults

from pyDR import Selection
from pyDR.Selection.MolSys import MolSys,MolSelect

#from pyDR import FRET
from pyDR.Data import Data,Plotting
from pyDR import Fitting
from pyDR import MDtools
from pyDR import Sens
from pyDR.misc.tools import tools
import sys as _sys
if 'MDAnalysis' in _sys.modules:
    from pyDR import Frames
else:
    print('MDAnalysis not available, MD processing will not be possible')
    
from pyDR import IO
from pyDR.iRED.iRED import iRED, Data_iRED

if 'MDAnalysis' in _sys.modules:
    from pyDR.Frames.eval_fr import md2data,md2iRED
from pyDR.chimeraX.CMXRemote import CMXRemote
from pyDR.chimeraX.Movies import Movies

from pyDR.Project import Project,Source

from pyDR.Entropy import EntropyCC



clsDict.update({'Data':Data,'Data_iRED':Data_iRED,'Source':Source,'Info':Sens.Info,
         'Sens':Sens.Sens,'Detector':Sens.Detector,'NMR':Sens.NMR,'MD':Sens.MD,'SolnNMR':Sens.SolnNMR,
         'MolSys':MolSys,'MolSelect':MolSelect,'Project':Project,
         'FrameObj':Frames.FrameObj,'Ctcalc':MDtools.Ctcalc,
         'DataPlots':Plotting.DataPlots,'CMXRemote':CMXRemote,'Movies':Movies,
         'EntropyCC':EntropyCC})

from pyDR import PCA

#%% Edit matlabplot subplotspec behavior to be consistent across versions
from matplotlib.axes import Subplot as _Subplot
from matplotlib.gridspec import SubplotSpec as _SubplotSpec
if hasattr(_SubplotSpec,'is_first_col'):
    def _fun(self):
        return self.get_subplotspec().is_first_col()
    _Subplot.is_first_col=_fun
    def _fun(self):
        return self.get_subplotspec().is_first_row()
    _Subplot.is_first_row=_fun
    def _fun(self):
        return self.get_subplotspec().is_last_col()
    _Subplot.is_last_col=_fun
    def _fun(self):
        return self.get_subplotspec().is_last_row()
    _Subplot.is_last_row=_fun


#%% Dark mode
if 'google.colab' in sys.modules:
    from google.colab import output
    is_dark = output.eval_js('document.documentElement.matches("[theme=dark]")')
    if is_dark:
        import matplotlib.pyplot as _plt
        x=56
        _plt.rcParams["figure.facecolor"]=(x/256,x/256,x/256)
        _plt.rcParams["axes.facecolor"]=(x/256,x/256,x/256)
        _plt.rcParams["axes.edgecolor"]=(1,1,1)
        _plt.rcParams["axes.labelcolor"]=(1,1,1)
        _plt.rcParams["xtick.color"]=(1,1,1)
        _plt.rcParams["ytick.color"]=(1,1,1)
        _plt.rcParams["text.color"]=(1,1,1)

#%% Warning handling
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', module='MDAnalysis')