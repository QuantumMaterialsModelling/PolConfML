import numpy as np

def _periodic_distance(x0, x1, dimensions):
    """helper function to compute the distance of two points
    in a periodic box

    Parameters
    ----------
    x0, x1 : numpy.array - cartesian coordinates
    dimensions - box size for periodic boundary conditions

    Returns
    -------
    shortest euclidean distance in a periodic box
    """
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def _rescale_and_pad(distances, n, R_c=15):
    """ rescales and pads a list of distances

    - orders a list by magnitude
    - either pads or truncates list s.t. n elements remain
    - rescales values smoothly between 0 and 1 for values within the cutoff radius

    Parameters
    ----------
    distances - list or numpy array of distances
    n - integer specific the number of elements the returned list has
    R_c - cutoff radius
    
    Returns
    -------
    array of length n with rescaled distances
    """
    distances = np.sort(distances)
    distances = distances[distances!=0]
    num_distances = distances.size
    if num_distances < n:
        distances = np.pad(distances,(0,n-num_distances),constant_values = R_c)
    else:
        distances = distances[:n]
    return np.where(
            distances < R_c, (np.cos(np.pi*distances/R_c)+1) / 2, 0)

def save_descs(path, descs, idxs):
    """ helper function that saves set of descriptors
    
    Parameters
    ----------
    path : str - path of directory where to save data
    descs : dict - descriptor dictionary
    idxs : dict - indices dictionary
    """
    for site in descs:
        np.save(path+site+'.npy', descs[site])
        np.save(path+site+'_ind.npy', idxs[site])
    
def load_descs(path, sites=['S0_A','S0_B',
                            'S1_A','S1_B',
                            'S2_A','S2_B']):
    """ helper function that loads set of descriptors
    
    Parameters
    ----------
    path : str - path of directory where to load data
    sites : list - list of site layer combinations to load (probably no change necessary)

    Returns
    -------
    descs : dict - dictionary containing descriptors
    idxs : dict - dictionary containing indices specifying the configuration a polaron
                  descriptor belongs to
    """
    descs = {}
    idxs = {}
    for site in sites:
        descs[site] = np.load(path+site+'.npy')
        idxs[site] = np.load(path+site+'_ind.npy')
    return descs, idxs