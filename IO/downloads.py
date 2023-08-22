# -*- coding: utf-8 -*-


import os
from urllib.request import urlretrieve

def download(url:str,filename:str='temp'):
    """
    Downloads a file from a web URL. Saved by default to the current directory
    as temp. Provide a filename to change the save location.

    Parameters
    ----------
    url : str
        url containing the file.
    filename : str, optional
        Filename for the stored file. The default is None.

    Returns
    -------
    None if failed, otherwise returns the resulting filepath

    """
    
    if 'drive.google.com' in url:
        return download_google_drive(url,filename)
    
    try:
        out=urlretrieve(url,filename)
        return out[0]
    except:
        print('File not found')
        return None

def getPDB(PDBid):
    pass



import pandas as pd
import numpy as np

def download_google_drive(url:str,filename:str='temp'):
    """
    Downloads publicly shared google drive csv files. May also work if somehow 
    google drive is mounted in Colab

    Parameters
    ----------
    url : str
        Share link for the file.
    filename : str, optional
        Output filename for the file. The default is 'temp'.

    Returns
    -------
    None.

    """
    url=cleanup_google_link(url)
    
    try:
        out=pd.read_csv(url)
        lines=np.concatenate(([out.columns[0]],out.to_numpy().flatten()))
    except:
        print('File not found')
        return None
    
    with open(filename,'w') as f:
        for line in lines:
            f.write(line+'\n')
    return filename        

def cleanup_google_link(link):
    return link