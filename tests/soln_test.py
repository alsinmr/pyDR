#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 16:51:14 2026

@author: albertsmith
"""

import pyDR
import numpy as np

proj=pyDR.Project()

proj.append_data('https://raw.githubusercontent.com/alsinmr/pyDR_tutorial/main/data/ubi_soln.txt')
data=proj[0]
data.info['zeta_rot']=1.18  #1.18=Dzz/Dperp=Dpara/Dperp (Dperp=0.5(Dxx+Dyy)  https://pubs.acs.org/doi/pdf/10.1021/ja409820g?ref=article_openPDF
data.info['eta_rot']=0       #(Dyy-Dxx)/(Dzz-Diso)
data.info['theta']=20  #H–N dipole/15N CSA angle: We should dig a bit to figure out what this should really be
                       # https://pubs.acs.org/doi/pdf/10.1021/ja910186u?ref=article_openPDF
                       
# Euler angles between diffusion tensor and pdb
for k in range(data.info.N):   #You have to loop over all experiments for this one because of the 3 entries
    data.info['euler',k]=np.array([120,155,0])*np.pi/180

# Add a pdb to the data, which will let us calculate relaxation from tumbling as a
# function of the H–N dipole orientation and 15N CSA orientation
data.select=pyDR.MolSelect('1d3z')
data.select.select_bond('15N',resids=data.label)

# Optimize detectors
data.detect.r_auto(5)
data.detect.R2ex()   #Add in fitting parameter to compensate for fast exchange

fit=data.fit()       #Fit the date

fit.plot(style='scatter')   #Plot

# Refit, but just use isotropic tumbling
data.info['zeta_rot']=1  #1.18=Dzz/Dperp=Dpara/Dperp (Dperp=0.5(Dxx+Dyy)  https://pubs.acs.org/doi/pdf/10.1021/ja409820g?ref=article_openPDF
data.info['eta_rot']=0       #(Dyy-Dxx)/(Dzz-Diso)

data.detect.r_auto(5)
data.detect.R2ex()

fit1=data.fit()

fit1.plot(style='scatter',color='black')


# We note that 