#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:42:34 2021

@author: albertsmith
"""

import numpy as np

#%% Sets plot attributes from kwargs
def _set_plot_attr(hdl,**kwargs):
    """
    Get properties for a list of handles. If values in kwargs are found in props,
    then that attribute is set (ignores unmatched values)
    """
    if not(hasattr(hdl,'__len__')): #Make sure hdl is a list
        hdl=[hdl]
    
    for m in hdl:
        props=m.properties().keys()
        for k,v in kwargs.items():
            if k in props and hasattr(m,'set_{}'.format(k)):getattr(m,'set_{}'.format(k))(v)
                
#%% Some classes for making nice labels with units and unit prefixes   
class NiceStr():
    def __init__(self,string,unit=None,include_space=True,no_prefix=False,is_range=False):
        self.string=string
        self.is_range=is_range
        self.index={}
        self.unit=unit
        self.include_space=include_space
        self.no_prefix=no_prefix
        
#        self.unit
#        self.include_space
#        self.no_prefix
 
    def __repr__(self):
        return self.string

    def prefix(self,value):
        if value==0:
            return '',0,1    
        pwr=np.log10(np.abs(value))
        x=np.concatenate((np.arange(-15,18,3),[np.inf]))
        pre=['a','f','p','n',r'$\mu$','m','','k','M','G','T']
        #Maybe the mu doesn't work
        for x0,pre0 in zip(x,pre):
            if pwr<x0:return '' if self.no_prefix else pre0,value*10**(-x0+3),10**(-x0+3)
            
    def format(self,*args,**kwargs):        
        string=self.string
        count=-1
        space=' ' if self.include_space else ''
        unit=self.unit if self.unit else ''
        parity=True
        while ':q' in string:
            parity=not(parity)
            count+=1
            
            #Find the 'q' tagged formating strings
            i=string.find(':q')
            #Extract the correct value from args and kwargs
            if string[i-1]=='{':
                v=args[count]
                start=i-1
            else:
                i1=string[:i].rfind('{')
                start=i1
                try:
                    v=args[int(string[i1+1:i])]
                except:
                    v=kwargs[string[i1+1:i]]
            #If we are specifying ranges, we only put units on every other number (fairly restricted implementation)
            if self.is_range and parity:  #Second steps only
                i1=string[i:].find('}')+i
                end=i1+1
                bd=1 if v==0 else np.floor(np.log10(np.abs(v))).astype(int)+1
                dec=np.max([prec-bd,0])
                if dec==0:v=np.round(v,prec-bd)
                v*=scaling #Use the same scaling as the previous step               
            else:

                if string[i+2]=='}':
                    prec=2
                    end=i+3
                else:
                    i1=string[i:].find('}')+i
                    end=i1+1
                    prec=int(string[i+2:i1])
                pre,v,scaling=self.prefix(v)
                
                bd=1 if v==0 else np.floor(np.log10(np.abs(v))).astype(int)+1
                dec=np.max([prec-bd,0])
                if dec==0:v=np.round(v,prec-bd)
                
            if not(self.is_range) or parity: #Second steps
                string=string[:start]+('0' if v==0 else '{{:.{}f}}'.format(dec).format(v))+space+pre+unit+string[end:]
            else: #First steps only
                string=string[:start]+('0' if v==0 else '{{:.{}f}}'.format(dec).format(v))+string[end:]  

        return string.format(*args,**kwargs)