import numpy as np
from .configurations import Configuration_TiO2, Configuration_SrTiO3
from .model import Dataset
from itertools import combinations, product
import linecache

class Search_TiO2():
    """class that allows construction of new configurations

    Attributes
    ----------
    path : str - path to a POSCAR file for which the new configurations should be constructed

    Methods
    -------
    _get_possible_idxs():
        gives indices of sites in the first three layers (could be restricted to only specific
        sites or layers)
    exhaustive_search(config, i):
       takes a configuration and adds i polarons to all possible sites and return the
       resulting configurations
    """
    def __init__(self, path):
        box_size = np.diag(np.loadtxt(path,skiprows=2,max_rows=3))
        self.positions = np.loadtxt(path,skiprows=9,usecols=(0,1,2))[:180]*box_size
        self.layer = self._layer_index()
        self.site = self._site_index()
        self.site['A'] = np.concatenate(self.site['A1']+self.site['A2'])
        self.site['B'] = np.concatenate(self.site['B1']+self.site['B2'])

    def _layer_index(self):
        idx = {}
        idx['S0'] = (self.positions[:,2]>12).nonzero()
        idx['S1'] = np.logical_and(
                self.positions[:,2]<12,self.positions[:,2]>10).nonzero()
        idx['S2'] = np.logical_and(
                self.positions[:,2]<10,self.positions[:,2]>7).nonzero()
        self.layer = idx
        return idx

    def _site_index(self):
        y = 13.1824998856
        y_8 = y/8
        upper = [1*y_8,3*y_8,5*y_8,7*y_8]
        lower = [7*y_8,1*y_8,3*y_8,5*y_8]
        idx = {}

        idx['A1'] = np.logical_or(self.positions[:,1]>lower[0],
        self.positions[:,1]<upper[0]).nonzero()
        idx['B1'] = np.logical_and(self.positions[:,1]>lower[1],
        self.positions[:,1]<upper[1]).nonzero()
        idx['A2'] = np.logical_and(self.positions[:,1]>lower[2],
        self.positions[:,1]<upper[2]).nonzero()
        idx['B2'] = np.logical_and(self.positions[:,1]>lower[3],
        self.positions[:,1]<upper[3]).nonzero()
        return idx

    def __getitem__(self, idxs):
        return self.positions[idxs]
    
    def _get_possible_idxs(self):
        """ creates a list of indices of hosting sites in the first three layers

        could be modified such that only specific sites are listed

        Returns
        -------
        list - contains indices of sites in first three layers
        """
        idxs = []
        for layer in self.layer:
            idxs.append(self.layer[layer])
        return list(np.sort(np.concatenate(idxs).flatten()))

    def exhaustive_search(self, config, n=2):
        """takes a configuration and returns all possible configurations with n added polarons

        Parameters
        ----------
        config : numpy.array - contains indices where polarons are located in configuration
        n : int - specifies number of polarons to add to config

        Returns
        -------
        config_idxs : numpy.array - indices of hosting sites in resulting configurations
        positions : numpy.array - cartesian coordinates of resulting configurations
        """
        if config.size == 0:
            idxs = self._get_possible_idxs()
            config_idxs =  list(combinations(idxs, n))
        else:
            idxs = self._get_possible_idxs()
            for i in config:
                idxs.remove(i)
            res = list(product([config],list(combinations(idxs, n))))
            config_idxs = np.array([np.concatenate(i) for i in res])
        return np.array(config_idxs), self.positions[config_idxs]
    
def search_extended_config_TiO2(net,
                           data_train,
                           search_class, 
                           configuration, 
                           num_pols_to_add,
                           ov_pos,
                           site_list,
                           box_size,
                           R_c=12):
    """wrapper that automates the prediction and search of configurations

    Parameters
    ----------
    net : Network - pretrained model
    data_train : Dataset - dataset for kernel evaluation of new data
    search_class : Search - search class that automates the search of new configs
    configuration : numpy.array : indices of polarons of configuraiton to which polarons 
                                  should be added
    num_pols_to_add : int - number of polarons to add to configurations
    """
    idxs_config, configs = search_class.exhaustive_search(
        configuration,num_pols_to_add)
    
    descs = {site:[] for site in site_list}
    idxs = {site:[] for site in site_list}
    
    for i,config in enumerate(configs):
        c = Configuration_TiO2(config, ov_pos, box_size)
        desc, idx = c.full_descriptors(3,i,R_c)
        
        for site in desc:
            descs[site].append(desc[site])
            idxs[site].append(idx[site])
            
    for site in descs:
        descs[site] = np.concatenate(descs[site])
        idxs[site] = np.concatenate(idxs[site])

    d_search = Dataset(descs,
                       idxs,
                       np.zeros(len(configs)),
                       np.zeros(len(configs)))
    d_search.descs = data_train.kernel_test(d_search)
    
    res = []
    for x,_,_ in d_search:
        res.append(net(x).item())
    res = np.array(res)
    sorted_idx = np.argsort(res)
    
    return idxs_config[sorted_idx], res[sorted_idx]

class Search_SrTiO3():
    """class that allows construction of new configurations

    Attributes
    ----------
    path : str - path to a POSCAR file for which the new configurations should be constructed

    Methods
    -------
    _get_possible_idxs():
        gives indices of sites in the first three layers (could be restricted to only specific
        sites or layers)
    exhaustive_search(config, i):
       takes a configuration and adds i polarons to all possible sites and return the
       resulting configurations
    """
    def __init__(self, path, def_num, ):
        box_size = np.diag(np.loadtxt(path,skiprows=2,max_rows=3))
        self.def_num = def_num

        if 'Cartesian' in linecache.getline(path, 9):
            self.positions = np.loadtxt(path,skiprows=9,usecols=(0,1,2))[def_num:120]
        else:
            self.positions = np.loadtxt(path,skiprows=9,usecols=(0,1,2))[def_num:120]*box_size
        self.layer = self._layer_index()
        
    def _layer_index(self):
        idx = {}
        idx['S0'] = (self.positions[:,2]>21).nonzero()[0]
        idx['S1'] = np.logical_and(
                self.positions[:,2]<20,self.positions[:,2]>18).nonzero()[0]
        idx['S2'] = np.logical_and(
                self.positions[:,2]<18,self.positions[:,2]>14).nonzero()[0]
        self.layer = idx
        return idx

    def __getitem__(self, idxs):
        return self.positions[idxs]
    
    def _get_possible_idxs(self):
        """ creates a list of indices of hosting sites in the first three layers

        could be modified such that only specific sites are listed

        Returns
        -------
        list - contains indices of sites in first three layers
        """
        idxs = []
        for layer in self.layer:
            idxs.append(self.layer[layer])
        return list(np.sort(np.concatenate(idxs, axis=0).flatten()))

    def exhaustive_search(self, config, n=2):
        """takes a configuration and returns all possible configurations with n added polarons

        Parameters
        ----------
        config : numpy.array - contains indices where polarons are located in configuration
        n : int - specifies number of polarons to add to config

        Returns
        -------
        config_idxs : numpy.array - indices of hosting sites in resulting configurations
        positions : numpy.array - cartesian coordinates of resulting configurations
        """
        if config.size == 0:
            idxs = self._get_possible_idxs()
            config_idxs =  list(combinations(idxs, n))
        else:
            idxs = self._get_possible_idxs()
            for i in config:
                idxs.remove(i)
            res = list(product([config],list(combinations(idxs, n))))
            config_idxs = np.array([np.concatenate(i) for i in res])
        return np.array(config_idxs), self.positions[config_idxs] 

def search_extended_config_SrTiO3(net,
                           data_train,
                           search_class, 
                           configuration, 
                           num_pols_to_add, 
                           Nb, 
                           cell,
                           site_list,
                           size = 20000, 
                           verbose=True):
    """wrapper that automates the prediction and search of configurations

    Parameters
    ----------
    net : Network - pretrained model
    data_train : Dataset - dataset for kernel evaluation of new data
    search_class : Search - search class that automates the search of new configs
    configuration : numpy.array : indices of polarons of configuraiton to which polarons 
                                  should be added
    num_pols_to_add : int - number of polarons to add to configurations
    num_ov : int - specifies up to which element ov_pos should be used for the descriptor 
    calculation
    """
    res = []
    idxs_kept = []
    idxs_config, configs = search_class.exhaustive_search(
        configuration,num_pols_to_add)
    
    descs = {site:[] for site in site_list}
    idxs = {site:[] for site in site_list}
    tmp_idxs = []
    
    num_Nb = Nb.shape[0]
    running_average = np.zeros(72-num_Nb)
    count_average = np.zeros(72-num_Nb)

    
    for i,(config,idx_config) in enumerate(zip(configs,idxs_config)):
        c = Configuration_SrTiO3(config, Nb, cell)
        desc, idx = c.full_descriptors(4,i%size,13)
        tmp_idxs.append(idx_config)
        
        for site in desc:
            descs[site].append(desc[site])
            idxs[site].append(idx[site])
            

        if i%size == 0:
            if verbose:
                print(i/idxs_config.shape[0])
            for site in descs:
                descs[site] = np.concatenate(descs[site])
                idxs[site] = np.concatenate(idxs[site])
            d_search = Dataset(descs,
                           idxs,
                           np.zeros(len(tmp_idxs)),
                           np.zeros(len(tmp_idxs)))
            d_search.descs = data_train.kernel_test(d_search)

            tmp = []
            for x,_,_ in d_search:
                tmp.append(net(x).item())
            tmp = np.array(tmp)
            tmp_idxs = np.array(tmp_idxs)
            
            
            
            
            for idx in range(72-num_Nb):
                mask = np.any(idx == tmp_idxs, axis=1)
                mask_n = mask.sum()
                count_average[idx] += mask_n
                running_average[idx] += np.sum(tmp[mask])
            
            
            sorted_idx = np.argsort(tmp)
            res.append(tmp[sorted_idx])
            idxs_kept.append(tmp_idxs[sorted_idx])
            
            descs = {site:[] for site in site_list}
            idxs = {site:[] for site in site_list}
            tmp_idxs = []

    if verbose:        
        print(i/idxs_config.shape[0])
    for site in descs:
        descs[site] = np.concatenate(descs[site])
        idxs[site] = np.concatenate(idxs[site])
    d_search = Dataset(descs,
                   idxs,
                   np.zeros(len(tmp_idxs)),
                   np.zeros(len(tmp_idxs)))
    d_search.descs = data_train.kernel_test(d_search)

    tmp = []
    for x,_,_ in d_search:
        tmp.append(net(x).item())
    tmp = np.array(tmp)
    tmp_idxs = np.array(tmp_idxs)

    
    for idx in range(72-num_Nb):
        mask = np.any(idx == tmp_idxs, axis=1)
        mask_n = mask.sum()
        count_average[idx] += mask_n
        running_average[idx] += np.sum(tmp[mask])

    
    
    sorted_idx = np.argsort(tmp)
    res.append(tmp[sorted_idx])
    idxs_kept.append(tmp_idxs[sorted_idx])

    descs = {site:[] for site in site_list}
    idxs = {site:[] for site in site_list}
    tmp_idxs = []
    return np.concatenate(idxs_kept), np.concatenate(res), running_average, count_average