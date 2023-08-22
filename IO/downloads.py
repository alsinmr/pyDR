# -*- coding: utf-8 -*-


from urllib.request import urlretrieve
import os

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

def getPDB(PDBid:str,filename:str=None):
    """
    Downloads and stores a pdb by its pdb code (4 characters). Filename is by 
    default the code with .pdb as endign

    Parameters
    ----------
    PDBid : str
        4 character pdb code.
    filename : str, optional
        File to store pdb. The default is 'None'.

    Returns
    -------
    None.

    """
    
    assert len(PDBid)==4,'The four character PDB id must be used'
    
    if filename is None:filename=PDBid.upper()+'.pdb'
    
    url='https://files.rcsb.org/download/{}.pdb'.format(PDBid.upper())
    try:
        out=urlretrieve(url,filename)
        return out[0]
    except:
        print('PDB not found')
        return None


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
        out=urlretrieve(url,filename)
        return out[0]
    except:
        print('File not found')
        return None
 

def cleanup_google_link(link:str):
    """
    Creates the correct link for downloading from Google Drive

    Parameters
    ----------
    link : str
        Original sharing link for Google Drive.

    Returns
    -------
    link : TYPE
        Link for downloading data.

    """
    
    a,b=os.path.split(link)
    if 'view?' in b:
        link=a
        ID=os.path.split(link)[1]
    else:
        ID=b

    link=f'https://drive.google.com/uc?id={ID}'
    
    return link