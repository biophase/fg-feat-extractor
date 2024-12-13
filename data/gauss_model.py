import torch
from torch import nn
from torch.nn.functional import relu
import numpy as np

class GaussModel(nn.Module):
    def __init__(self, num_data_points, signal_length, init_mu=None, init_eps=None):
        super(GaussModel,self).__init__()
        # self.mu = nn.Parameter(torch.rand(num_data_points)[:,None]*signal_length, requires_grad=True)
        # self.eps = nn.Parameter(torch.rand(num_data_points)[:,None]*signal_length, requires_grad=True)        
        if init_mu is None:
            self.mu = nn.Parameter(torch.zeros(num_data_points)[:,None]+signal_length/2, requires_grad=True)
        else:
            self.mu = nn.Parameter(init_mu, requires_grad=True)            
        if init_eps is None:
            self.eps = nn.Parameter(torch.ones(num_data_points)[:,None]*1, requires_grad=True)     
        else:
            self.eps = nn.Parameter(init_eps, requires_grad=True)
            
    @property
    def sigma(self):
        return torch.log(1+torch.exp(self.eps))
    
    def forward(self, x):
        return 1/(self.sigma * np.sqrt(2*np.pi)) * torch.exp(-0.5*((x-self.mu)/self.sigma)**2) #- self.base
    


class GMM(nn.Module):
    def __init__(self, num_data_points, signal_length, num_gaussians):
        super(GMM,self).__init__()
        self.num_gaussians = num_gaussians
        self.gaussians = nn.ModuleList([GaussModel(num_data_points, signal_length) for _ in range(num_gaussians)])
        self.weights = nn.Parameter(torch.ones(num_gaussians)/num_gaussians, requires_grad=True)  # type: ignore
    def forward(self, x):
        return sum([relu(w) * g(x) for w, g in zip(self.weights, self.gaussians)])    



def mse_loss(y_pred, y_true):
    diff = y_pred-y_true
    # diff[torch.eq(y_true,0)] = 0
    loss = torch.mean(torch.sum((diff**2),axis=1),axis=0)
    return loss

