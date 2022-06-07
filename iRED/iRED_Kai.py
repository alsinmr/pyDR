
import pyDR.Data.Data as Data
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pyDR.MDtools.Ctcalc import Ct


class iRED():
    def __init__(self, molsel = None, vecs = None):
        """

        :param molsel: pyDR.Selction.MolSelect object
        """
        if molsel:
            assert hasattr(molsel,"select_bond") or (hasattr(molsel, "sel1") and hasattr(molsel, "sel2")), "this is not " \
               "the right object for initialisation, please use pyDR.Selection.MolSelect"
            self.molsel = molsel
            assert molsel.sel1 and molsel.sel2, f"please use select_bond() on {molsel.__class__.__name__} first"
        else:
            self.molsel = None
        if vecs:
            self.vecs = vecs
        else:
            self.vecs = None

        if molsel and vecs:
            warnings.warn("You assigned the iRED with a molsel AND a set of vectors. So be sure the vectors are fitting "+
                          "for the Selection or run get_vecs() manually")


    def full_analysis(self):
        # iRed fast line 45
        """
        Runs the full iRED analysis for a given selection (or set of vec_special functions)
        Arguments are the rank (0 or 1), the sampling (n,nr), whether to align the
        vectors (align_iRED='y'/'n', and refVecs, which may be a dict containing
        a vector, created by DIFRATE, a tuple of strings selecting two sets of atoms
        defining bonds), or simply 'y', which will default to using the N-CA bonds in
        a protein.

        ired=iRED_full(mol,rank=2,n=100,nr=10,align_iRED='n',refVecs='n',**kwargs)

        """
        if self.vecs is None:
            self.get_vecs()
        nt = len(self.molsel.traj)
        vec = self.vecs# get_trunc_vec(mol, index, **kwargs)
        rank = 2
        ired = self.vecs2iRED(vec, rank)#, align_iRED, refVecs=vec0, molecule=mol, **kwargs)

        return ired

    def vecs2iRED(self, vec, rank):
        M = Mmat(vec, rank)#['M']
        print(M)
        plt.imshow(M['M'])
        Yl = Ylm(vec, rank)
        aqt = Aqt(Yl, M)
        print("Aqt",aqt)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.cts = np.zeros((vec.shape[0],vec.shape[-1]))
        for i in range(self.cts.shape[0]):
            self.cts[i] = Ct(vec[i])[0]
        ax.semilogx(self.cts.T)
        plt.show()


    def get_vecs(self):
        traj = self.molsel.molsys.traj
        self.vecs = np.zeros((len(self.molsel.sel1),3,len(traj)))
        for i,_ in enumerate(traj):
            if i%10000==0:
                print(i)
            self.vecs[:,:,i] = self.molsel.v
            ### aware these are not normalized

    def to_data(self) -> Data:
        assert 0, "implement!"
        data = Data()
        return data


def Mmat(vec, rank):
    #todo move inside class
    M = np.eye(vec[:, 0, :].shape[0])*0
    if rank == 2:
        for dim in range(3):
            for j in range(dim,3):
                M += ((vec[:, j] * vec[:, dim])@(vec[:, j] * vec[:, dim]).T)*(1 if dim == j else 2)
        M *= 3/2/vec.shape[-1]
        M -= 1/2
    elif rank == 1:
        #todo
        # test if this is correct
        for k in range(3):
            for j in range(k, 3):
                M += vec[k]@vec[j].T
    else:
        assert 0, "rank should be 1 or 2"
    a = np.linalg.eigh(M)
    return {'M': M, 'lambda': a[0], 'm': a[1], 'rank': rank}

def Ylm(vec, rank):
    #todo move inside class
    X = vec[:, 0, :]
    Y = vec[:, 1, :]
    Z = vec[:, 2, :]
    Yl = {}
    if rank == 1:
        c = np.sqrt(3 / (2 * np.pi))
        Yl['1,0'] = c / np.sqrt(2) * Z
        a = (X + Y * 1j)
        b = np.sqrt(X ** 2 + Y ** 2)
        Yl['1,+1'] = -c / 2 * b * a
        Yl['1,-1'] = c / 2 * b * a.conjugate()
    elif rank == 2:
        c = np.sqrt(15 / (32 * np.pi))
        Yl['2,0'] = c * np.sqrt(2 / 3) * (3 * Z ** 2 - 1)
        a = (X + Y * 1j)
        b = np.sqrt(X ** 2 + Y ** 2)
        Yl['2,+1'] = 2 * c * Z * b * a
        Yl['2,-1'] = 2 * c * Z * b * a.conjugate()
        a = np.exp(2 * np.log(X + Y * 1j))
        b = b ** 2
        Yl['2,+2'] = c * b * a
        Yl['2,-2'] = c * b * a.conjugate()
    Yl['t'] = vec.shape[-1]
    return Yl

def Aqt(Yl, M):
    #todo move inside class
    "Project the Ylm onto the eigenmodes"
    aqt = {}
    for k in Yl.keys():
        if k != 't':
            aqt[k] = np.dot(M.get('m').T, Yl.get(k))

    aqt['t'] = Yl.get('t')
    return aqt