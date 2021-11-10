#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:29:46 2021

@author: albertsmith
"""

import numpy as np

class Info():
    """
    Info stores a 2D array describing experimental parameters for a set of 
    experiments, where the experiments are, for example, NMR measurements or
    elements from MD correlation functions. 
    """
    def __init__(self,**kwargs):
        self.keys=[]
        self.__values=np.zeros([0,0],dtype=object)
        self.N=0
        self.__edited=False
        

        for key,value in kwargs.items():
            self.new_parameter(key,value)
    
    def new_parameter(self,key,value=None):
        """
        Add a new parameter to all existing experiments. Provide the name of the
        parameter (usually a string, cannot not be an integer), and a list of
        values where the length should be equal to the number of experiments 
        (Info.N, where first entry into info can have any length)
        """
        assert not(isinstance(key,int)),"Parameters should not be of type 'int'"
        if self.__values.size==0:
            if value is None:
                self.keys.append(key)
                self.__values=np.zeros([1,0],dtype=object)
            else:
                assert np.array(value).ndim<2,"Parameters stored in Info should be given as 1D arrays"
                self.keys.append(key)
                self.__values=np.atleast_2d(value).astype(object)
                self.N=len(value)
        else:
            if value is None:value=[None for _ in range(self.N)]   #Empty parameter
            assert len(value)==self.N,"Length of the new parameter should equal the number of experiments ({})".format(self.N)
            self.__values=np.concatenate((self.__values,[value]),axis=0)
            self.keys.append(key)
        self.__edited=True
            
    def new_exper(self,**kwargs):
        """
        Adds a new experiment, where one provides all parameters for that experiment
        as keywork arguments.
        """
        for key in kwargs.keys():   #Add new keys in case they do not exist yet
            if key not in self.keys:
                self.new_parameter(key)
                
        new=np.array([None for _ in range(len(self.keys))])
        for key,value in kwargs.items():
            new[self.keys.index(key)]=value
        if self.__values.size==0:
            self.__values=np.array([new]).T
        else:
            self.__values=np.concatenate((self.__values,np.array([new]).T),axis=1)
        self.__edited=True
        self.N+=1
    
    def __getitem__(self,index):
        """
        Returns either a given parameter (provide the key), a given experiment
        (provide the index), or a specific parameter for a specific experiment (
        provide the key followed by the index)
        """
        x=index
        if isinstance(x,tuple):
            assert len(x)==2 and not(isinstance(x[0],int)) and isinstance(x[1],int),\
            "Request either a given parameter (provide the key), a given experiment (provide the index as int), or provide the key,index pair"
            return self.__values[self.keys.index(x[0]),x[1]]
        else:
            if isinstance(x,int) or (hasattr(x,'dtype') and np.issubdtype(x.dtype,int)):
                assert x<self.N,"Index must be less than the number of experiments ({0})".format(self.N)
                return {key:value for key,value in zip(self.keys,self.__values[:,x])}
            elif isinstance(x,slice):
                start,stop,step=x.start if x.start else 0, x.stop if x.stop else self.N,x.step if x.step else 1
                if stop==-1:stop=self.N
                out=Info()
                for k in range(start,stop,step):
                    out.new_exper(**self[k])
                return out
            elif x in self.keys:
                return self.__values[self.keys.index(x)]
            elif hasattr(x,'__len__'):
                out=Info()
                if np.array(x).dtype=='bool':
                    x=[int(x0) for x0 in np.argwhere(x)[:,0]]

                for x0 in x:
                    assert isinstance(x0,int),"Indices must be integers or boolean"
                    out.new_exper(**self[x0])
                    
                return out                   
            else:
                assert 0,"Unknown parameter"
                
    def __setitem__(self,index,value):
        """
        Sets an item in the Info object. Provide a parameter name, key, and value
        """
        print('udpated')
        assert isinstance(index,tuple),"Both the parameter name and index must be provided"
        assert index[0] in self.keys,"Unknown parameter"
        print(index[1])
        assert index[1]<self.N,"Index must be less than the number of experiments ({0})".format(self.N)
        self.__values[self.keys.index(index[0]),index[1]]=value
        self.__edited=True
    
    def __repr__(self):
        """
        Print a nice table of all parameters and experiments
        """
        
        "Collect all the strings to print"
        out=['']
        for k in range(self.N):out.append('{}'.format(k))
        out.append('\n')
        for key,values in zip(self.keys,self.__values):
            out.append('{}'.format(key))
            for v in values:out.append('{}'.format(v))
            out.append('\n')
        out=out[:-1]
        
        "Find out how long all entries are (make table line up by printing to equal-length strings)"
        ml=np.zeros(self.N+2)
        for k,o in enumerate(out):
            k0=np.mod(k,self.N+2)
            ml[k0]=np.min([np.max([ml[k0],len(o)]),10])
        ml+=1
            
        string=''
        for k,o in enumerate(out):
            if o=='\n':
                string+=o
            else:
                l=ml[np.mod(k,self.N+2)]
                fmt_str='{:<'+'{:.0f}'.format(l)+'.'+'{:.0f}'.format(l)+'s}' \
                if np.mod(k,self.N+2)==0 else\
                 '{:>'+'{:.0f}'.format(l)+'.'+'{:.0f}'.format(l)+'s}'
                string+=fmt_str.format(o)
        string+='\n\n[{0} experiments with {1} parameters]'.format(self.N,len(self.keys))        
        return string
    
    @property
    def edited(self):
        """
        Returns True if any parameters in Info have been changed, thus requiring
        the sensitivities to be re-calculated.
        """
        return self.__edited
    
    def updated(self):
        """
        Call self.updated if the sensitivities have been updated, thus setting
        self.edited to False
        """
        self.__edited=False
        
    def del_exp(self,index):
        """
        Delete one or more experiments from info
        """
        
        if hasattr(index,'__len__'):
            for i in np.sort(index)[::-1]:
                self.del_exp(i)
        else:
            assert index<self.N,"Index must be less than the number of experiments ({0})".format(self.N)
            self.__values=np.concatenate((self.__values[:,:index],self.__values[:,index+1:]),axis=1)
            self.N+=-1
                
            
            
        