#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 16:51:14 2026

@author: albertsmith
"""

import pyDR

data=pyDR.IO.readNMR('https://raw.githubusercontent.com/alsinmr/pyDR_tutorial/main/data/ubi_soln.txt')
# data.info['zeta_rot']=1.18  #Dzz/Dperp=Dpara/Dperp (Dperp=0.5(Dxx+Dyy)  https://pubs.acs.org/doi/pdf/10.1021/ja409820g?ref=article_openPDF
data.info['zeta_rot']=1  #Dzz/Dperp=Dpara/Dperp (Dperp=0.5(Dxx+Dyy)  https://pubs.acs.org/doi/pdf/10.1021/ja409820g?ref=article_openPDF
data.info['eta_rot']=0       #(Dyy-Dxx)/(Dzz-Diso)
data.label=data.label.astype(int)
data.select=pyDR.MolSelect('1d3z')
data.select.select_bond('15N',resids=data.label)

data.detect.r_auto(5)

fit=data.fit()