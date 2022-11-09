import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


''' 
Class for Deep Sets architecture (as developed in 
https://arxiv.org/abs/1703.06114). This models a permutation invariant scalar 
function as f({x_i}) = \rho(\sum_i \phi(x_i)), where phi and rho are neural 
networks.
'''

class Deep_Sets(nn.Module):
  def __init__(self, input_dim, width, depth_phi, depth_rho):
    '''
    Initializes a Deep Sets architecture

    Parameters:
      input_dim: dimension of input to the deep sets (i.e. dimension of x_i)
      width: width of the networks phi and rho (we've chosen these widths to be
        the same)
      depth_phi, depth_rho: depth of the networks phi and rho, respectively
    '''

    # Initializes as a nn.Module
    super().__init__()

    # Set parameters
    self.width = width
    self.depth_phi = depth_phi
    self.depth_rho = depth_rho
    self.input_dim = input_dim

    # Creates fully connected (fc) layers for phi network
    self.phi_fc0 = nn.Linear(self.input_dim, self.width)
    self.phi_fcs = nn.ModuleList([nn.Linear(self.width, self.width)
                                  for i in range(self.depth_phi-1)])

    # Creates fully connected (fc) layers for rho network
    self.rho_fcs = nn.ModuleList([nn.Linear(self.width, self.width)
                                  for i in range(self.depth_rho-1)])
    self.rho_fc_end = nn.Linear(self.width, 1)


  def forward(self, x_emb, mask):
    '''
    Passes the data x_emb (an array of embedded particle positions) through the
    deep sets network.

    Parameters:
      x_emb: an array of dimension (n_samples, n_particles, input_dim), which
        represents n_samples samples of n_particles particle positions, each
        embedded into a latent space of dimension input_dim. Since particle
        number may fluctuate, x_emb may contain entries corresponding to
        non-existent particles; these take value 0 and are masked out by the
        variable 'mask'.
      mask: a Boolean-valued array of dimension (n_samples, n_particles) used to
        remove entries corresponding to non-existent particles when passed through
        the deep sets network. Takes value 1 if particle exists, 0 if
        non-existent.

    Returns:
      y: an array of dimension (n_samples, 1) containing the output of the deep
        sets network across the n_samples samples
    '''

    # Pass through phi network
    y = torch.tanh(self.phi_fc0(x_emb))
    for l in self.phi_fcs:
      y = torch.tanh(l(y))

    # Pooling; mask is used to remove non-existent particle entries
    y = torch.sum(y*mask, axis=-2)

    # Pass through rho network
    for l in self.rho_fcs:
      y = torch.tanh(l(y))
    y = F.softplus(self.rho_fc_end(y))    # softplus enforces positivity

    return y

