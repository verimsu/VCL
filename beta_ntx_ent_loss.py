import math
import torch
from torch import nn
import torch.nn.functional as F

class vloss(nn.Module):
    def __init__(self):
        super(vloss,self).__init__()

    def l1(self, x, y):
        return x - y

    def l2(self, x, y):
        return (x - y).square()

    def forward(self, mu, logVar):
        lVp = logVar[:,0,:]
        lVq = logVar[:,1,:]
        mp = mu[:,0,:]
        mq = mu[:,1,:]
        sm = 0.5 * (lVp.exp() + lVq.exp())
        lSm = sm.log()/2
        mm = 0.5 * (mp + mq)
        eq_ = (0.5 * (self.l1(lSm, lVp/2) + self.l1(lSm, lVq/2)) + 0.25 * ((self.l2(mp,mm) + self.l2(mq,mm)) / sm)).mean()
        norm_ = - 0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
        return eq_ + norm_

class betaNTXentLoss(torch.nn.Module):
    def __init__(self, beta=0.0005, temperature=0.07):
        super(betaNTXentLoss, self).__init__()
        self.beta = beta
        self.sigma = 0.5
        self.temperature = temperature
        self.eps = 1e-6
        
    def mask_type_transfer(self, mask):
        mask = mask.type(torch.bool)
        return mask

    def get_pos_and_neg_mask(self, bs):
        zeros = torch.zeros((bs, bs), dtype=torch.uint8)
        eye = torch.eye(bs, dtype=torch.uint8)
        pos_mask = torch.cat([
            torch.cat([zeros, eye], dim=0), torch.cat([eye, zeros], dim=0),
        ], dim=1)
        neg_mask = (torch.ones(2*bs, 2*bs, dtype=torch.uint8) - torch.eye(
            2*bs, dtype=torch.uint8))
        pos_mask = self.mask_type_transfer(pos_mask)
        neg_mask = self.mask_type_transfer(neg_mask)
        return pos_mask, neg_mask        
    
    def similarity_matrix(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary');
    
    def GaussianDistance(self, z0 ,z1):
        const1 = ((1 + self.beta) / self.beta)
        const2 = 1 / pow((2 * math.pi * (self.sigma**2)), self.beta)
        return const1 * (const2 * torch.exp(-(self.beta * 128 / (2 * (self.sigma**2))) * self.similarity_matrix(z0, z1)) - 1)
    
    def forward(self, zis, zjs):
        device = zis.device
        batch_size = zis.shape[0]
        pos_mask, neg_mask = self.get_pos_and_neg_mask(batch_size)
        pos_mask, neg_mask = pos_mask.to(device), neg_mask.to(device)
        
        zis, zjs = F.normalize(zis, dim=1), F.normalize(zjs, dim=1)
        z_all = torch.cat([zis, zjs], dim=0)

        sim_mat = self.GaussianDistance(z_all, z_all) / self.temperature

        sim_pos = sim_mat.masked_select(pos_mask).view(2*batch_size).clone()
        sim_neg = sim_mat.masked_select(neg_mask).view(2*batch_size, -1).clone()
        
        loss = (- sim_pos + torch.logsumexp(sim_neg, dim=-1)).mean()
        
        return loss
