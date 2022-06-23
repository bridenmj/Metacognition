from torchvision import datasets, transforms
import torch
import torch.nn as nn

class DropInput(nn.Module):
    """
    with x chance, select p percent of indices and set to zero
    """
    def __init__(self,p=0.5, x =0.5,device_ = None):
        self.p = p
        self.x = x
        self.device = device_

    def __call__(self, tensor):
        bsz = tensor.size()[0]
        indices = torch.randperm(bsz)[:int(bsz*self.x)].to(self.device)
        mask = torch.ones(indices.size()[0],17,39,11).to(self.device)
        select = torch.rand(indices.size()[0],17,39,11).to(self.device)
        mask[select <= (self.p)] = 0
        tensor[indices,:,:] = tensor[indices,:,:]*mask
        return tensor

class AddPinkNoise(nn.Module):
    """
    Add gaussian white noise to scalograms with probability p
    """
    def __init__(self, mean=0., std=1.0 ,p=0.5,device_ = None):
        self.std = std
        self.mean = mean
        self.p = p
        self.device = device_

    def __call__(self, tensor):
        batch_mask = torch.zeros(tensor.size()[0]).to(self.device)
        select = torch.rand(tensor.size()[0]).to(self.device)
        batch_mask[select <= self.p] = 1
        batch_mask = batch_mask.view(-1,1,1,1)
        f_inv  = torch.tensor(1/torch.unsqueeze(torch.arange(tensor.shape[2],0,-1),1)).to(self.device)
        return tensor + torch.randn(tensor.size()).to(self.device) * self.std * f_inv * batch_mask + self.mean


    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={}, p={})'.format(self.mean, self.std, self.p)

class AddGaussianNoise(nn.Module):
    """
    Add gaussian white noise to scalograms with probability p
    """
    def __init__(self, mean=0., std=1.0 ,p=0.5,device_ = None):
        self.std = std
        self.mean = mean
        self.p = p
        self.device = device_

    def __call__(self, tensor):
        batch_mask = torch.zeros(tensor.size()[0]).to(self.device)
        select = torch.rand(tensor.size()[0]).to(self.device)
        batch_mask[select <= self.p] = 1
        batch_mask = batch_mask.view(-1,1,1,1)

        return tensor + torch.randn(tensor.size()).to(self.device) * self.std * batch_mask + self.mean


    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={}, p={})'.format(self.mean, self.std, self.p)



class GaussianCorruption(nn.Module):
    """
    with x chance, select p percent of indices and add gaussian noise
    """
    def __init__(self, mean=0., std=1.0 ,p=0.5, x =0.5,device_ = None):
        self.std = std
        self.mean = mean
        self.p = p
        self.x = x
        self.device = device_

    def __call__(self, tensor):
        batch_mask = torch.zeros(tensor.size()[0]).to(self.device)
        select = torch.rand(tensor.size()[0]).to(self.device)
        batch_mask[select <= self.x] = 1
        batch_mask = batch_mask.view(-1,1,1,1)

        mask = torch.zeros(tensor.size()).to(self.device)
        select = torch.rand(tensor.size()).to(self.device)
        mask[select<=(self.p)] = 1
        return tensor + mask * batch_mask * torch.randn(tensor.size()).to(self.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={}, p={}, x={})'.format(self.mean, self.std, self.p,self.x)




    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={}, p={}, x={})'.format(self.mean, self.std, self.p,self.x)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, device_ ):
        self.device = device_
        self.transform = transform

    def __call__(self,x):
      augmentation = torch.empty(2, len(x[:,0,0,0]), 17,39,11).to(self.device )
      augmentation[0] = self.transform(x)
      augmentation[1] = self.transform(x)

      return augmentation
