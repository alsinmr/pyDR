#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:07:59 2022

@author: albertsmith
"""

from pyDR import clsDict
import numpy as np
from copy import copy

def product_mat(detector) -> np.ndarray:
    """
    Calculates the product matrix required to take the product of detector
    responses of two correlation functions.
    
    Current functionality only allows analyzing correlation functions which have
    the same sampling, and have been analyzed with the same set of detectors.

    Parameters
    ----------
    r : np.ndarray
        DESCRIPTION.

    Returns
    -------
    np.ndarray
    Matrix for calculating the product of detector responses. For n detectors,
    the output matrix dimension will be nxn^2

    """
    r=detector.r
    ri=np.linalg.pinv((r.T*detector.sens.norm).T)
    
    n=r.shape[1]
    pmat=np.zeros([n,n**2],dtype=r.dtype)
    for k in range(n):
        for j in range(n):
            pmat[:,k+n*j]=ri@(r[:,k]*r[:,j]*detector.sens.norm)
    return pmat

def data_product(data:list):
    """
    If we have detector responses of two correlation functions which were originally
    the same length, then we may calculate the detector responses which would
    result from the product of the original two correlation functions.
    
    It should be noted: this function is essentially 'fitting' the product of
    the back-calculated correlation functions, via a pseudo-inverse matrix. There
    are, therefore, no bounds placed on the resulting detectors, and any imperfect
    fitting that occured in the initially fit of the two or more correlation 
    functions is furthermore carried back into this product. Then, note that 
    this operation will often not yield perfect reproduction of the original
    correlation function.
    
    While we are in principle fitting the back calculated correlation functions,
    note that we are not actually required to calculate the correlation function.
    The back-calculation and fit can be collected into a single matrix, thus
    bypassing the actually calculation of the correlation functions- see the
    function product_mat above.

    Parameters
    ----------
    data : list or Project
        List of data objects for which we take the product.

    Returns
    -------
    Data object

    """
    out=copy(data[0])
    pmat=product_mat(data[0].sens)
    
    nr,n=data[0].R.shape
    
    for d in data[1:]:
        Rp=np.zeros([nr,n**2],dtype=d.R.dtype)
        for k in range(n):
            Rp[:,k*n:(k+1)*n]=(out.R.T*d.R[:,k]).T
        out.R=(pmat@Rp.T).T

    out.source.additional_info='DetProduct'
    out.source.details.append('Result of a detector product calculation')
    out.source.details.append('The following data was multiplied: '+','.join([d.title for d in data]))
    out.source.details.append('Warning: some details above may apply only to the first data object of the product calculation')
    out.source.src_data=None #Multiple pieces of source data, therefore we omit this here.   
    
    if data[0].source.project is not None:data[0].source.project.append_data(out)

    return out
        
        