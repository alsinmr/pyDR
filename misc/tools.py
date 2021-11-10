#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:11:31 2021

@author: albertsmith
"""

from pyDR.Sens import Info
import os
import re
import numpy as np


class NucInfo(Info):
    """ Returns the gyromagnetic ratio for a given nucleus. Usually, should be 
    called with the nucleus and mass number, although will default first to 
    spin 1/2 nuclei if mass not specified, and second to the most abundant 
    nucleus. A second argument, info, can be specified to request the 
    gyromagnetic ratio ('gyro'), the spin ('spin'), the abundance ('abund'), or 
    if the function has been called without the mass number, one can return the 
    default mass number ('mass'). If called without any arguments, a pandas 
    object is returned containing all nuclear info ('nuc','mass','spin','gyro',
    'abund')
    """
    def __init__(self):
        h=6.6260693e-34
        muen=5.05078369931e-27
        super().__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))    
        with open(dir_path+'/GyroRatio.txt','r') as f:
            for line in f:
                line=line.strip().split()
                self.new_exper(Nuc=line[3],mass=float(line[1]),spin=float(line[5]),\
                               gyro=float(line[6])*muen/h,abundance=float(line[7])/100)
    
    def __call__(self,Nuc=None,info='gyro'):
        if Nuc is None:
            return self
        
        if Nuc=='D':
            Nuc='2H'
 
        #Separate the mass number from the nucleus type       
        mass=re.findall(r'\d+',Nuc)
        if not mass==[]:
            mass=int(mass[0])
                   
        Nuc=re.findall(r'[A-Z]',Nuc.upper())
        
        #Make first letter capital
        if np.size(Nuc)>1:
            Nuc=Nuc[0].upper()+Nuc[1].lower()
        else:
            Nuc=Nuc[0]
            
            
        ftd=self[self['Nuc']==Nuc]  #Filtered by nucleus input
        
        "Now select which nucleus to return"
        if not mass==[]:    #Use the given mass number
           ftd=ftd[ftd['mass']==mass]
        elif any(ftd['spin']==0.5): #Ambiguous input, take spin 1/2 nucleus if exists
            ftd=ftd[ftd['spin']==0.5] #Ambiguous input, take most abundant nucleus
        elif any(ftd['spin']>0):
            ftd=ftd[ftd['spin']>0]
            
        ftd=ftd[np.argmax(ftd['abundance'])]
        
        if info is None:
            return ftd
        else:
            assert info in self.keys,"info must be 'gyro','mass','spin','abundance', or 'Nuc'"
            return ftd[info]

NucInfo=NucInfo()