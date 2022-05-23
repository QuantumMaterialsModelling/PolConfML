import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import laplacian_kernel

dtype = torch.float32

class ScalarProd(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = torch.nn.Parameter(data=torch.zeros(n), 
                                    requires_grad=True)
        
    def forward(self, x):
        out = torch.matmul(x,self.w,)
        return out
    
class AddLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, xs):
        out = 0
        for x in xs:
            out += x.sum()
        return out
    
class Network(nn.Module):
    def __init__(self, ns):
        super(Network,self).__init__()
        self.regressors = nn.ModuleList([ScalarProd(n) for n in ns])
        self.adding = AddLayer()
        
    def forward(self, xs):
        out = self.adding([reg(x) for (reg,x) in zip(self.regressors, xs)])
        return out


class Dataset(torch.utils.data.Dataset):
    """Torch dataset class that allows loading all descriptors belonging
    to a specific configuration and transforming data to kernel representation

    Attributes
    ----------
    descs - dict
        dictionary containing descriptors
    idxs - dict
        dictionary containing indices that assign descriptors to configurations
    Y - numpy.array
        energies of polaron configurations
    ov - numpy.array
        number of oxygen vacancies of polaron configurations

    Methods
    -------
    __len__():
        returns the number of configurations in the dataset
    __getitem__():
        returns descriptors, energy and defect concentration of configuration with index i
    get_network_sizes():
        returns a tuple that specifies the number of single polaron descriptors in each
        polaron class
    kernel(gamma):
        transforms data into kernel representation
    kernel_test(data):
        calculates kernel function of a different dataset with configurations in this dataset
    """
    def __init__(self, descs, idxs, Y, ov):
        self.descs = descs.copy()
        self.idxs = idxs.copy()
        self.y = torch.tensor(Y).type(dtype)
        self.ov = torch.tensor(ov).type(dtype)
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return tuple(self.descs[site][self.idxs[site] == i] for site in self.descs), self.y[i], self.ov[i]
    
    def get_network_sizes(self):
        return tuple(self.descs[site].shape[0] for site in self.descs)
    
    def kernel(self,gamma):
        """ calculates the laplacian kernel of polaron descriptors at each hosting site

        Overwrites raw descriptor vectors with kernelized version and stores the raw version
        in descs_raw

        Parameters
        ----------
        gamma : float - kernel parameter for laplacian kernel (see sklearn documentation)

        """
        self.gamma = gamma
        self.descs_raw = self.descs.copy()
        for site in self.descs:
            self.descs[site] = torch.from_numpy(laplacian_kernel(self.descs[site],gamma=gamma))
            self.descs[site] = self.descs[site].type(dtype)
        
    def kernel_test(self, data):
        """ calculates kernel matrices of different training set with descriptors in this
        dataset. Requires you to call the self.kernel(gamma) function beforehand

        Parameters
        ----------
        data : Dataset - the dataset of configurations that should be kernelized

        Returns
        -------
        kernel_test : dict - descriptor dictionary that contains kernelized version of data
        """
        kernel_test = {}
        for site in data.descs:
            if data.descs[site].shape[0]>0:
                kernel_test[site] = torch.from_numpy(
                    laplacian_kernel(
                        self.descs_raw[site],
                        data.descs[site],
                        gamma=self.gamma
                    )
                )
                kernel_test[site] = kernel_test[site].type(dtype)
                kernel_test[site] = torch.transpose(kernel_test[site],1,0)
            else:
                kernel_test[site] = torch.zeros(size=(self.descs[site].shape[0],0)).type(dtype)
                kernel_test[site] = torch.transpose(kernel_test[site],1,0)
        return kernel_test
    
    def kernel_test_single(self, data):
        kernel_test = {}
        for site in data:
            if data[site].shape[0]>0:
                kernel_test[site] = torch.from_numpy(
                    laplacian_kernel(
                        self.descs_raw[site],
                        data[site],
                        gamma=self.gamma
                    )
                )
                kernel_test[site] = kernel_test[site].type(dtype)
                kernel_test[site] = torch.transpose(kernel_test[site],1,0)
            else:
                kernel_test[site] = torch.zeros(size=(self.descs[site].shape[0],0))
                kernel_test[site] = torch.transpose(kernel_test[site],1,0)
        return kernel_test