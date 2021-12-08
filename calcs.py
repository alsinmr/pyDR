import numpy as np
from numba import *
from numba import cuda
from SpeedTest import *
from math import sqrt

def search_methyl_groups(residue, v=True):
    """this function will search a residue of a protein for methyl groups by iterating over all atoms and if the atom
    is surrounded by 3 hydrogens it will look for the shortest path to carbonyl carbon of the peptide plane
    It happened before that when pdb and xtc are not 100 % fitting, that the positions are totally screwed. futhermore,
    nor bond information were available. in this case this function will just print out garbage"""
    if v:
        print(residue.resname)
    try:
        #todo this is not working properly, repair it
        get_by_dist = False if len(residue.universe.bonds)>len(residue.universe.atoms) else True
        # check if bond information of the universe is available, see 'get_bonded'
    except:
        get_by_dist = True
        if v:
            print("No bond information")

    #todo this is just a workaround right now, fix this!
    if "SER" in residue.resname:
        return []


    def search(atom, exclude=[]):  # ignore possible warning on the mutable [], this is on purpose here
        """searching a path from a methyl group of a residue down to the C-alpha of the residue
        returns a list of atoms (MDA.Atom) beginning with the hydrogens of the methyl group and continuing
        with the carbons of the side chain
        returns empty list if atom is not a methyl carbon"""

        def get_bonded():
            """it happens, that pdb files do not contain bond information, in that case, we switch to selection
            by string parsing"""
            if not get_by_dist:
                return atom.bonded_atoms
            return residue.atoms.select_atoms("around 1.85 name " + atom.name)

        if "C" == atom.name and len(exclude):
            return [atom]
        elif atom.name == "N" or atom.name == "C":
            return []
        connected_atoms = []
        bonded = get_bonded()
        if v:
            print(bonded)
        if len(exclude) == 0:
            if np.sum(np.fromiter(["H" in a.name for a in bonded], dtype=bool)) == 3:
                for a in bonded:
                    if "H" in a.name:
                        connected_atoms.append(a)
                if not ("C" in atom.name or "S" in atom.name):
                    return []
            else:
                return []
        connected_atoms.append(atom)
        exclude.append(atom)
        for a in bonded:
            if a not in exclude:
                next_atom = search(a, exclude)
                for b in next_atom:
                    connected_atoms.append(b)
        if len(connected_atoms) > 1:
            return connected_atoms
        else:
            return []

    methyl_groups = []
    for resatom in residue.atoms:
        chain = search(resatom, [])
        if len(chain):
            for at in chain:
                if v:
                    print(at)
            if v:
                print()
            methyl_groups.append(chain)
    return methyl_groups


@njit
def get_x_y_z_old(p1, p2, p3, p):
    """p1 is the carbon of the methyl group
    p is a hydrogen
    p2 and p3 are carbons to define the planes in the frame"""
    p_n_x = np.cross(p2 - p1, p3 - p1)  # setting z-y-plane
    # p_n_x /= np.linalg.norm(p_n_x)  # normalized vector of the plane, going in x direction
    p_n_x /= np.sqrt((p_n_x ** 2).sum())

    x = np.dot(p - p1, p_n_x)  # therefore get the x distance of the point p
    # print(dist)  #ok so we got the x value, or y, whatever
    # now we have to rotate the plane by 90° and recalculate the distance for y, or x
    p3 = p2 + p_n_x  # substituting p3 by vector in x direction
    p_n_y = np.cross(p2 - p1, p3 - p1)  # this will be the vector in y direction
    # p_n_y /= np.linalg.norm(p_n_y)  # normalized vector of the plane
    p_n_y /= np.sqrt((p_n_y ** 2).sum())
    y = np.dot(p - p1, p_n_y)

    p2 = p1 + p_n_x
    p3 = p1 + p_n_y

    p_n_z = np.cross(p2 - p1, p3 - p1)
    # p_n_z /= np.linalg.norm(p_n_z)
    p_n_z /= np.sqrt((p_n_z ** 2).sum())
    z = np.dot(p - p1, p_n_z)
    return [x, y, z]


@njit
def get_x_y_z(p):
    """p[0] is the carbon of the methyl group
    p[3] is a hydrogen
    p[1] and p[2] are carbons to define the planes in the frame"""
    p_n_x = np.cross(p[1] - p[0], p[2] - p[0])  # setting z-y-plane
    # p_n_x /= np.linalg.norm(p_n_x)  # normalized vector of the plane, going in x direction
    p_n_x /= np.sqrt((p_n_x ** 2).sum())
    x = np.dot(p[3] - p[0], p_n_x)  # therefore get the x distance of the point p
    # print(dist)  #ok so we got the x value, or y, whatever
    # now we have to rotate the plane by 90° and recalculate the distance for y, or x
    p[2] = p[1] + p_n_x  # substituting p3 by vector in x direction
    p_n_y = np.cross(p[1] - p[0], p[2] - p[0])  # this will be the vector in y direction
    # p_n_y /= np.linalg.norm(p_n_y)  # normalized vector of the plane
    p_n_y /= np.sqrt((p_n_y ** 2).sum())
    y = np.dot(p[3] - p[0], p_n_y)

    p[1] = p[0] + p_n_x
    p[2] = p[0] + p_n_y

    p_n_z = np.cross(p[1] - p[0], p[2] - p[0])
    # p_n_z /= np.linalg.norm(p_n_z)
    p_n_z /= np.sqrt((p_n_z ** 2).sum())
    z = np.dot(p[3] - p[0], p_n_z)
    return x, y, z

@njit
def fast_dihedral(p):
    # TODO put this to https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    b1 = p[2] - p[1]
    # b0, b1, b2 = -(p[1] - p[0]) , b1 / np.linalg.norm(b1), p[3] - p[2]
    b0, b1, b2 = -(p[1] - p[0]), b1 / np.sqrt((b1 ** 2).sum()), p[3] - p[2]
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

@njit
def faster_dihedral(p):
    # TODO put this to https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    b1 = p[2] - p[1]
    # b0, b1, b2 = -(p[1] - p[0]) , b1 / np.linalg.norm(b1), p[3] - p[2]
    b0, b1, b2 = -(p[1] - p[0]), b1 / np.sqrt((b1 ** 2).sum()), p[3] - p[2]
    v = b0 - (b0[0] * b1[0] + b0[1] * b1[1] + b0[2] * b1[2]) * b1
    w = b2 - (b2[0] * b1[0] + b2[1] * b1[1] + b2[2] * b1[2]) * b1
    x = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
    c = np.cross(b1, v)  # maybe when one gets rid of this one it can get a little faster
    y = c[0] * w[0] + c[1] * w[1] + c[2] * w[2]
    return np.degrees(np.arctan2(y, x))


@time_runtime
@njit
def fastest_dihedral(p):
    # TODO put this to https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    b1 = p[2] - p[1]
    # b0, b1, b2 = -(p[1] - p[0]) , b1 / np.linalg.norm(b1), p[3] - p[2]
    b0, b1, b2 = -(p[1] - p[0]), b1 / np.sqrt((b1 ** 2).sum()), p[3] - p[2]
    v = b0 - (b0[0] * b1[0] + b0[1] * b1[1] + b0[2] * b1[2]) * b1
    w = b2 - (b2[0] * b1[0] + b2[1] * b1[1] + b2[2] * b1[2]) * b1
    x = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
    c = np.cross(b1, v)  # maybe when one gets rid of this one it can get a little faster
    y = c[0] * w[0] + c[1] * w[1] + c[2] * w[2]
    return 180 * np.arctan2(y, x) / np.pi


#@time_runtime
@njit(parallel=True)
def fast_dihedral_multi(p, arr):
    """this was supposed to be faster  than fast dihedral, but it didnt change program runtime significant
    maybe one day I have an idea to improve it"""
    for i in prange(p.shape[0]):
        ind = 4 * i
        b1 = p[ind + 2] - p[ind + 1]
        b0, b1, b2 = -(p[ind + 1] - p[ind]), b1 / np.sqrt((b1 ** 2).sum()), p[ind + 3] - p[ind + 2]
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        arr[i] = 180.0*np.arctan2(y, x)/np.pi


@njit
def calc_distances(carbon, others):
    # todo i think there must be a faster way to calc this by using all carbons in one function at once
    # todo problem here is the subtraction
    return np.sqrt(((carbon - others) ** 2).sum(1))


@njit
def pos_xyz(p, npos, capos, cpos):
    """this funciton is a little outdated, but i don't want to delete it right now, improved version is
    pos_xyz_o"""
    # capos = CA.position = (0,0,0) in the end
    # create the z-axis as normal vector of plane between N,CA and C(O)
    nz = np.cross(npos - capos, cpos - capos)
    nz /= np.linalg.norm(nz)
    # calculate the z position as distance from the plane
    z = np.dot(p - capos, nz)
    # new_c is a made up carbon to define the y-plane
    new_c = capos + nz
    # create the y-axis as normal vector of plane between N,CA and the made up C
    ny = np.cross(npos - capos, new_c - capos)
    ny /= np.linalg.norm(ny)
    # calculate the y position as distance from the plane
    y = np.dot(p - capos, ny)
    # replace N position with made up pos to define a new plane
    new_n = capos + ny
    nx = np.cross(new_n - capos, new_c - capos)
    nx /= np.linalg.norm(nx)
    x = np.dot(p - capos, nx)

    return [x, y, z]

@time_runtime
@njit
def pos_xyz_o(out, p, npos, capos, cpos,  p2=np.array([0]), norm=True):
    """this function calculates the position of point p depending on the plane that is span by npos,capos,cpos
    naming if the three points comes from the original idea that the plane is spanned by the peptide plane defined
    by N-CA-C(O)
    The resulting vector is written in out, optional normalized, while the (0,0,0) point of the so defined "frame"
    is in CA
    p2 is optional position of a atom to be substracted from p, for example if you want the C-H vector of an ILE methyl
    group relative to the peptide plane, the default value had to be chosen like this because of njit, but maybe there
    is a better way to do this"""
    # capos = CA.position = (0,0,0) in the end
    # create the z-axis as normal vector of plane between N,CA and C(O)
    nmca = npos - capos  # saves a calculation step later, but looks ugly
    nz = np.cross(nmca, cpos - capos)
    nz /= np.sqrt((nz ** 2).sum())
    # create the y-axis as normal vector of plane between N , CA and nz
    ny = np.cross(nmca, nz)
    ny /= np.sqrt((ny ** 2).sum())
    # make normal vector nx from ny and nz
    nx = np.cross(ny, nz)
    # nx doesnt have to be normalized
    pmca = p - capos

    out[0] = np.dot(pmca, nx)
    out[1] = np.dot(pmca, ny)
    out[2] = np.dot(pmca, nz)
    if p2[0]:
        pmca2 = p2 - capos
        out[0] -= np.dot(pmca2,nx)
        out[1] -= np.dot(pmca2,ny)
        out[2] -= np.dot(pmca2,nz)
    # out = np.dot(p-capos,[nx,ny,nz])
    if norm:
        out /= np.sqrt((out ** 2).sum())

@njit
def P2(x):
    """second legendre polynomial"""
    return (3 * x * x - 1) / 2

@time_runtime
@njit(parallel=True)  # speedup by factor 1600!
def get_ct_S2(ct,S2, v, indices,sparse=1):
    """
    calculating the correlation functions or an array of vectors

    C(t) = <P2(v(0) * v(t))>

    the bigger the arg sparse is, the more computation time is needed
    also, calculating the S2 for the vector with

    S2 = 3/2 (<xi²>² + <yi²>² + <zi²>² + 2 <xiyi>² + 2<xizi>² + 2<yizi>²) - 1/2
    """
    l = ct.shape[1]
    for i in prange(1, l):
        # r is doing somehow sparsely sampling, since 'early' timesteps of the correlation funciton cause the biggest
        # impact for calculatio
        if sparse >0:
            r = max(int(np.sqrt((l-i)/sparse)), 1)
        else:
            r = 1
        l2 = l - i  # second range, just not for recalculating it every time<<<<<x
        for k in prange(ct.shape[0]):  # iterating over the number of vectors    v
            if indices[k]:
                ct[k, i] = P2(np.array([np.dot(v[k, j], v[k, j + i]) for j in range(0, l2, r)])).mean()
        # just leaving the old calculation here to make it better visible what i actually calculate:
        # arr = np.zeros(l-i)
        # for j in prange(l-i):
        #    arr[j] = P2(np.dot(v[j],v[j+i]))
        # ct[i] = arr.mean()#np.average(P2(np.array([np.dot(v[j], v[j + i]) for j in prange(l - i)])))
    for k in prange(ct.shape[0]):
        if indices[k]:
            S2[k] = 3/2*((v[k, :, 0]**2).mean()**2    # <xi²>²
                       + (v[k, :, 1]**2).mean()**2    # <yi²>²
                       + (v[k, :, 2]**2).mean()**2    # <zi²>²
                       + 2*(v[k, :, 0]*v[k, :, 1]).mean()**2  # <xiyi>²
                       + 2*(v[k, :, 0]*v[k, :, 2]).mean()**2  # <xizi>²
                       + 2*(v[k, :, 1]*v[k, :, 2]).mean()**2  # <yizi>²
                         )-1/2

#@guvectorize([(float32[:, :, :], float32[:, :])], '(n,m,o)->(n,m)', target="parallel")
#just put the comment to remove the warning
def get_ct2( v, ct):
    """
    I leave this here as an example. For sum reason it is slower AND the result is a little different
    dont understand why
    """
    l = ct.shape[1]
    for i in prange(1, l):
        # r is doing somehow sparsely sampling, since 'early' timestepts of the correlation funciton cause the biggest
        # impact for calculatio
        r = max(int(np.sqrt((l-i)/1)), 1)
        l2 = l - i  # second range, just not for recalculating it every time<<<<<x
        for k in prange(ct.shape[0]):#ct.shape[0]):  # iterating over the number of vectors    v
            ct[k, i] = np.array([np.dot(v[k, j], v[k, j + i]) for j in range(0, l2, r)]).mean()



@time_runtime
@njit(parallel=True)  # speedup by factor 1600!
def get_S2(S2, v, indices):
    """
    S2 = 3/2 (<xi²>² + <yi²>² + <zi²>² + 2 <xiyi>² + 2<xizi>² + 2<yizi>²) - 1/2
    """
    for k in prange(v.shape[0]):
        if indices[k]:
            S2[k] = 3/2*((v[k, :, 0]**2).mean()**2    # <xi²>²
                       + (v[k, :, 1]**2).mean()**2    # <yi²>²
                       + (v[k, :, 2]**2).mean()**2    # <zi²>²
                       + 2*(v[k, :, 0]*v[k, :, 1]).mean()**2  # <xiyi>²
                       + 2*(v[k, :, 0]*v[k, :, 2]).mean()**2  # <xizi>²
                       + 2*(v[k, :, 1]*v[k, :, 2]).mean()**2  # <yizi>²
                         )-1/2

@cuda.jit(device=True)
def cuP2(x):
    """second legendre polynomial"""
    return (3 * x * x - 1) / 2

@cuda.jit(device=True)
def dot(v1,v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]


@cuda.jit()
def ct_kernel(cts,vecs, indices, sparse):
    startX,startY= cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    for i in prange(startX,cts.shape[1],gridX):  #length of trajectory
        for k in prange(startY,cts.shape[0],gridY):  # number of correlation funcitons
            if indices[k]:
                if sparse:
                    r = max(int(sqrt((cts.shape[1] - i))/sparse),1)
                else:
                    r=1
                c = 0

                for j in range(0,cts.shape[1]-i,r):  # number of pairs
                    cts[k, i] += cuP2(dot(vecs[k, j], vecs[k, j + i]))
                    c+=1
                cts[k,i] /= c


@time_runtime
def calc_CT_on_cuda(cts, S2s, vecs, indices, sparse):
    #todo calculate griddim depending on ct shape
    blockdim = (32,4)
    griddim = (4096,16)
    cts[indices,1:]=0
    cts[indices,0]=1

    dev_vecs = cuda.to_device(vecs)
    dev_cts = cuda.to_device(cts)
    dev_ind = cuda.to_device(indices)
    ct_kernel[griddim,blockdim](dev_cts,dev_vecs, dev_ind, sparse)
    dev_cts.copy_to_host(cts)
    get_S2(S2s, vecs, indices)

