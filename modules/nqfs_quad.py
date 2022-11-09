# Import packages
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.special import gammaln, factorial
from deep_sets import Deep_Sets


'''
Class for the Deep Sets Neural-Network Quantum Field State, building off the
architecture developed in https://arxiv.org/abs/2112.11957. In this ansatz,
each n-particle state is modelled as \varphi_n = q_n*DS1({x_i})*DS2({x_i-x_j})
for discrete distribution q_n, and deep sets networks DS1 and DS2. This ansatz
is specially designed for the quadratic model (hence the subscript '_Quad').
'''


class NQFS_Quad():
    def __init__(self, DS_width, DS_depth_phi, DS_depth_rho, L, periodic):
        '''
        Initializes the NQFS ansatz

        Parameters:
          DS_width, DS_depth_phi, DS_depth_rho: width and depths of the underlying
          neural networks (i.e. rho and phi) of DS1 and DS2
          L: system length
          periodic: system periodicity (True if periodic, False if not)
        '''

        # Initializes deep sets DS1 and DS2
        input_dim1 = 2  # The input to DS1 is a 2d embedding
        input_dim2 = 1  # The input to DS2 is a 1d embedding
        self.DS1 = Deep_Sets(input_dim1, DS_width, DS_depth_phi, DS_depth_rho)
        self.DS2 = Deep_Sets(input_dim2, DS_width, DS_depth_phi, DS_depth_rho)

        # Register and initialize parameters of q_n
        '''q_n is modelled as a smoothed rectangular pulse; its parameters are 
           included in DS1 for optimization. Explicitly, 
           q_n = 1/(1+e^{-s*(n-c_1)})*1/(1+e^{s*(n-c_2)}) for 0 < c_1 < c_2 and s > 0,
           where q_n_mean = (c_1 + c_2)/2, q_n_width = (c_2 - c_1), q_n_slope = s.'''
        self.DS1.register_parameter(name='q_n_mean',
                                    param=torch.nn.Parameter(torch.tensor(2.)))
        # Parametrize the inverse softplus of the width to ensure width is positive
        self.DS1.register_parameter(name='q_n_inv_softplus_width',
                                    param=torch.nn.Parameter(torch.tensor(3.)))
        # Parameterize the inverse softplus of the slope to ensure slope is positive
        self.DS1.register_parameter(name='q_n_inv_softplus_slope',
                                    param=torch.nn.Parameter(torch.tensor(3.)))
        self.DS1.register_parameter(name='q_n_inv_softplus_slope',
                                    param=torch.nn.Parameter(torch.tensor(np.log(3.))))

        # Set system parameters
        self.L = L
        self.periodic = periodic

    def Embedding1(self, x):
        '''
        Constructs the embedding of particle positions x for DS1

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing n_samples
          samples of n_particles particle positions

        Returns:
          embedded_data: a tensor of dimension (n_samples, n_particles, 2)
          containing the embedding of the particle positions
        '''

        # If periodic, embedding = (sin(2*pi*x/L), cos(2*pi*x/L))
        # If not periodic, embedding = (x/L, 1-x/L)
        if self.periodic:
            x_tmp = 2 * np.pi / self.L * (x[None, ...]).permute(1, 2, 0)
            sines = torch.sin(x_tmp)
            cosines = torch.cos(x_tmp)
            embedded_data = torch.cat((sines, cosines), axis=2)
        else:
            x_norm = x[:, :, None] / self.L
            embedded_data = torch.cat((x_norm, 1 - x_norm), axis=2)

        return embedded_data

    def Embedding2(self, x):
        '''
        Constructs the embedding of particle positions x for DS2

        Parameters:
          x: an tensor of dimension (n_samples, n_particles) representing n_samples
          samples of n_particles particle positions

        Returns:
          embedded_data: a tensor of dimension
          (n_samples, n_particles*(n_particles-1)/2, 1) containing the embedding of
          the particle positions
        '''

        # Determines interparticle separations {x_i - x_j} for i < j
        idx = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
        interparticle_seps = (x[:, :, None] - x[:, None, :])[:, idx[0], idx[1]]

        # If periodic, embedding = cos(2*pi*(x_i-x_j)/L)
        # If not periodic, embedding = ((x_i-x_j)/L)^2
        if self.periodic:
            interparticle_seps_tmp = 2 * np.pi / self.L * \
                                     (interparticle_seps[None, ...]).permute(1, 2, 0)
            embedded_data = torch.cos(interparticle_seps_tmp)
        else:
            embedded_data = (interparticle_seps[..., None] / self.L) ** 2

        return embedded_data

    def Psi(self, x):
        '''
        Calculates Psi(x)

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions. x may contain entries
          corresponding to non-existent particles; we denote these by a 'nan' value

        Returns:
          val: a tensor of dimension (n_samples, 1) containing the values of
          Psi(x)
        '''

        # The wave function is enforced to be positive; to consider positive values
        # of lam, you'll want to force the wave function to have alternating sign
        val = torch.exp(self.log_Psi(x))
        return val

    def log_Psi(self, x):
        '''
        Calculates log(Psi(x))

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions. x may contain entries
          corresponding to non-existent particles; we denote these by a 'nan' value

        Returns:
          val: a tensor of dimension (n_samples, 1) containing the values of
          log_Psi(x)
        '''

        # Contribution from DS1
        x_emb1 = self.Embedding1(x)
        # Constructs mask; non-existent particle entries in x are denoted by 'nan'
        mask1 = ((~x.isnan())[..., None]).detach()
        val = (1 - self.periodic) * self.log_DS1(x_emb1.nan_to_num(), mask1)  # Really only need this if not periodic

        # Contribution from DS2
        n = torch.sum(~x.isnan(), axis=1)[:, None]
        x_emb2 = self.Embedding2(x)
        mask2 = (~x_emb2.isnan()).detach()
        val += (n >= 2) * self.log_DS2(x_emb2.nan_to_num(), mask2)

        # Contribution from q_n
        val += 1 / 2 * self.log_q_n(n)

        # Contribution from cutoff factor if system is not periodic
        if not self.periodic:
            val += self.log_cutoff_factor(x, n)

        return val

    def log_DS1(self, x_emb1, mask1):
        '''
        Returns log(DS1(x_emb1, mask1))

        Parameters:
          x_emb1: a tensor of dimension (n_samples, n_particles, 2)
          containing the embedding of the particle positions for DS1
          mask1: a Boolean tensor of dimension (n_samples, n_particles),
          denoting existent particle positions

        Returns:
          val: a tensor of dimension (n_samples, 1) containing the values of
          log(DS1(x_emb1, mask1))
        '''

        val = torch.log(self.DS1.forward(x_emb1, mask1))
        return val

    def log_DS2(self, x_emb2, mask2):
        '''
        Returns log(DS2(x_emb2, mask2))

        Parameters:
          x_emb2: a tensor of dimension (n_samples, n_particles, 1)
          containing the embedding of the particle positions for DS2
          mask2: a Boolean tensor of dimension
          (n_samples, n_particles*(n_particles-1)/2) denoting existent particle
          positions

        Returns:
          val: a tensor of dimension (n_samples, 1) containing the values of
          log(DS2(x_emb2, mask2))
        '''

        val = torch.log(self.DS2.forward(x_emb2, mask2))
        return val

    def log_q_n(self, n):
        '''
        Calculates log(q_n) where q_n = 1/(1+e^{-s*(n-c_1)})*1/(1+e^{s*(n-c_2)})
        for 0 < c_1 < c_2 and s > 0, and q_n_mean = (c_1 + c_2)/2,
        q_n_width = (c_2 - c_1), and  q_n_slope = s

        Parameters:
          n: a tensor of dimension (n_samples, 1) containing the number of
          (existent) particles for each sample

        Returns:
          val: a tensor of dimension (n_samples, 1) containing the values of
          log(q_n)
        '''

        # q_n_width = torch.log(1 + torch.exp(self.DS1.q_n_inv_softplus_width))
        q_n_width = self.DS1.q_n_inv_softplus_width
        c_1 = 1 / 2 * (2 * self.DS1.q_n_mean - q_n_width)
        c_2 = 1 / 2 * (2 * self.DS1.q_n_mean + q_n_width)
        # q_n_slope = torch.log(1 + torch.exp(self.DS1.q_n_inv_softplus_slope))
        q_n_slope = torch.exp(self.DS1.q_n_inv_softplus_slope)
        s = q_n_slope

        val = -torch.log(1 + torch.exp(-s * (n - c_1)))
        val += -torch.log(1 + torch.exp(s * (n - c_2)))

        # Optional extra factor of L^(-n/2) to reduce dependence on system size
        val += -n * np.log(self.L)

        return val

    def log_cutoff_factor(self, x, n):
        '''
        Calculates logarithm of the cutoff factor used to implement closed boundary
        conditions. Explicitly, cutoff factor = 1/N*\prod_i{x_i/L*(1-x_i/L)},
        where 1/N = (30/L)^(n/2) is a normalization factor.

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          n: a tensor of dimension (n_samples, 1) containing the number of
          (existent) particles of each sample

        Returns:
          val: a tensor of dimension (n_samples, 1) that contains the values of
          log(cutoff factor)
        '''

        val = (x % self.L) / self.L * (1 - (x % self.L) / self.L)
        val = torch.sum(torch.log(val + 1e-16).nan_to_num(), axis=1)[:, None]
        val += -n * 1 / 2 * np.log(self.L / 30)
        return val

    def MetropolisHastings_FockSpace_Quad(self, n_samples, n_chains, p_pm, n_0,
                                          GPU_device):
        '''
        Performs the Metropolis-Hastings algorithm in Fock space to produce
        samples of particle configurations drawn from |Psi|^2. This runs multiple
        MCMC chains on GPU to parallelize data generation. This algorithm
        specializes to the quadratic model and only allows the particle number to
        change in increments/decrements of 2.

        Parameters:
          n_samples: number of samples drawn for each MCMC chain
          n_chains: number of separate MCMC chains
          p_pm: probability of increasing or decreasing particle number at each
          configuration proposal
          n_0: initial particle number of each MCMC chain
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          x_list: a list of n_samples tensors, each containing the sampled particle
          configurations across the MCMC chains at the specified iteration. Each
          such tensor is of dimension (n_chains, n), where n is the maximum particle
          number of the sampled configurations at the corresponding iteration.
          n_list: a list of n_samples tensors, each containing the number of
          particles of the sampled particle configurations across the MCMC
          chains at the specified iteration. Each such tensor is of dimension
          (n_chains, 1), and takes integer values.
        '''

        # Specialize to particle number being even
        if (n_0 % 2) != 0:
            raise Exception("n_0 must be even!")

        # Sets width of the uniform distribution used for configuration proposals
        w = 0.3 * self.L

        # Initializes n_list as the number of particles, each confiuration
        # containing n_0 particles at the beginning
        n_list = [n_0 * torch.ones(n_chains, dtype=torch.long, device=GPU_device)]

        # Initializes x_list as paticle positions; positions are chosen randomly; a
        # value 'nan' is used to denote the absence of a particle
        if n_0 == 0:
            x_0 = torch.cuda.FloatTensor(n_chains, 1).fill_(float('nan'))
        else:
            x_0 = self.L * torch.rand(n_chains, n_0, device=GPU_device)
        x_list = [x_0]

        # Sets current particle positions, particle number, and log(Psi)
        x = x_list[0].detach().clone()
        n = n_list[0].detach().clone()
        log_Psi_val = self.log_Psi(x)

        # Runs MCMC
        for i in range(n_samples - 1):
            # Generates random numbers to choose which chains will be proposed to have
            # a particle added or removed; if neither, the proposal perturbs the
            # particle positions
            u = torch.cuda.FloatTensor(2, n_chains).uniform_()
            add_ind = torch.where(u[0, :] < p_pm)[0]
            remove_ind = torch.where(torch.logical_and(u[0, :] > p_pm,
                                                       u[0, :] < 2 * p_pm))[0]
            perturb_ind = (u[0, :] > 2 * p_pm)[:, None]

            # Generates proposed configurations

            # Adds two particles at a randomly chosen positions
            x_proposed = torch.cat((x, \
                                    torch.cuda.FloatTensor(n_chains, 2).fill_(float('nan'))), axis=1)
            x_proposed[add_ind, n[add_ind]] = \
                self.L * torch.cuda.FloatTensor(add_ind.shape[0]).uniform_()
            x_proposed[add_ind, n[add_ind] + 1] = \
                self.L * torch.cuda.FloatTensor(add_ind.shape[0]).uniform_()

            # Removes two particles at random (n_cutoff is used to ensure particle
            # number doesn't become negative)
            n_cutoff = torch.maximum(torch.cuda.FloatTensor(1).fill_(2.),
                                     n[remove_ind]).long()
            x_remove_ind = torch.floor( \
                torch.cuda.FloatTensor(remove_ind.shape[0]).uniform_() * n_cutoff).long()
            x_proposed[remove_ind, x_remove_ind] = x_proposed[remove_ind, n_cutoff - 1]
            x_proposed[remove_ind, n_cutoff - 1] = float('nan')
            n_cutoff = torch.maximum(torch.cuda.FloatTensor(1).fill_(1.),
                                     n[remove_ind] - 1).long()
            x_remove_ind = torch.floor( \
                torch.cuda.FloatTensor(remove_ind.shape[0]).uniform_() * n_cutoff).long()
            x_proposed[remove_ind, x_remove_ind] = x_proposed[remove_ind, n_cutoff - 1]
            x_proposed[remove_ind, n_cutoff - 1] = float('nan')

            # Perturbs particle positions by a uniform random variable of mean 0 and
            # width w
            perturbation = torch.cuda.FloatTensor(x_proposed.shape).uniform_(-w / 2, w / 2)
            x_proposed += perturb_ind * perturbation
            x_proposed %= self.L

            # Calculates particle numbers of proposed configurations
            n_proposed = n.detach().clone()
            n_proposed[add_ind] += 2
            n_proposed[remove_ind] = torch.maximum(n_proposed[remove_ind] - 2, \
                                                   torch.cuda.FloatTensor(1).fill_(0.).long())

            # Computes acceptance ratio
            log_Psi_val_proposed = self.log_Psi(x_proposed)
            L_factor = torch.cuda.FloatTensor(n_chains, 1).fill_(1.)
            L_factor[add_ind, 0] *= self.L ** 2
            L_factor[remove_ind, 0] *= 1 / self.L ** 2
            A = torch.minimum(torch.cuda.FloatTensor(n_chains, 1).fill_(1.),
                              L_factor * torch.exp(2 * (log_Psi_val_proposed - log_Psi_val)))

            # Accepts or rejects configuration proposals (using the random numbers u
            # generated earlier), and then updates x, n, and log_Psi_val
            accept_ind = torch.where(u[1, :] < A[:, 0])[0]
            x = torch.cat((x, \
                           torch.cuda.FloatTensor(n_chains, 2).fill_(float('nan'))), axis=1)
            x[accept_ind, :] = x_proposed[accept_ind, :]
            while torch.all(x[:, -1].isnan()) and x.shape[1] > 1:
                x = x[:, :-1]
            log_Psi_val[accept_ind, :] = log_Psi_val_proposed[accept_ind, :]
            n[accept_ind] = n_proposed[accept_ind]

            # Records particle configurations and particle number
            n_list.append(n.detach().clone())
            x_list.append(x.detach().clone())

        return x_list, n_list

    def GenerateMCMCSamples_FockSpace(self, n_samples, n_chains, p_pm, n_0,
                                      GPU_device):
        '''
        Generates particle configurations drawn from |Psi|^2 using the algorithm for
        Metropolis-Hastings in Fock space, and subsequently removes MCMC burn-in
        phase and sorts the configurations by particle number.

        Parameters:
          n_samples: number of samples drawn for each MCMC chain
          n_chains: number of separate MCMC chains
          p_pm: probability of increasing or decreasing particle number at each
          configuration proposal
          n_0: initial particle number of each MCMC chain
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          x_sorted: a list of tensors, the nth of which contains the sampled
          particle configurations containing n particles and is of dimension (#, n),
          where # denotes the number of drawn samples of particle number n
          n: a tensor containing the particle numbers of the configurations drawn
          from |Psi|^2
          chain_idx_sorted: a list of tensors, each of the same dimension as those
          of x_sorted. An entry of such tensor denotes the MCMC chain from
          which the corresponding particle configuration came.
        '''

        # Generate samples
        x_list, n_list = self.MetropolisHastings_FockSpace_Quad(n_samples, n_chains, p_pm,
                                                                n_0, GPU_device)

        # Discard values from the MCMC burn-in phase
        discard_fraction = 0.3
        n_discard = min(500, int(discard_fraction * n_samples))
        n = torch.flatten(torch.stack(n_list[n_discard:]))
        n_max = torch.max(n)
        n_min = torch.min(n)
        x = torch.cuda.FloatTensor((n_samples - n_discard) * n_chains,
                                   max(1, n_max)).fill_(float('nan'))
        for i in range(n_samples - n_discard):
            x[i * n_chains:(i + 1) * n_chains, :x_list[i + n_discard].shape[1]] = \
                x_list[i + n_discard]

            # Sort by particle number and determine chain of origin
        x_sorted = []
        chain_idx = torch.arange(n_chains, \
                                 device=GPU_device)[:, None].repeat(n_samples - n_discard, 1)
        chain_idx_sorted = []
        for n_val in range(n_min, n_max + 1):
            idx = torch.where(n == n_val)[0]
            x_sorted.append(x[idx, :n_val])
            chain_idx_sorted.append(chain_idx[idx, 0])

        return x_sorted, n, chain_idx_sorted

    def lambda_Local_Energy(self, x, lam, N_int, batch_size, GPU_device):
        '''
        Calculates the local energy of the lambda term in the Hamiltonian, for the
        configurations in x

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          lam_local: a tensor of dimension (n_samples, 1) containing the local
          energies of the lambda term for the configurations in x
        '''

        x_int_points = torch.linspace(0, self.L, N_int, device=GPU_device)

        lam_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)
        for k in range(np.ceil(x.shape[0] / batch_size).astype(int)):
            x_tmp = x[k * batch_size:(k + 1) * batch_size, :]
            Psi_denominator = self.Psi(x_tmp)[:, 0].detach()

            x_int = torch.cat((x_tmp[None, :, :].repeat(N_int, 1, 1),
                               x_int_points[:, None, None].repeat(1, x_tmp.shape[0], 2)), axis=2)
            x_int = x_int.reshape(N_int * x_tmp.shape[0], x_tmp.shape[1] + 2)
            integrand_vals = self.Psi(x_int).detach()
            integrand_vals = integrand_vals.reshape(N_int, x_tmp.shape[0], 1)[:, :, 0]

            integrals = torch.trapz(integrand_vals, dx=self.L / (N_int - 1), dim=0)
            lam_local[k * batch_size:(k + 1) * batch_size, 0] = 2 * lam * integrals / Psi_denominator
            del x_tmp, Psi_denominator, x_int, integrand_vals, integrals

        n = x.shape[1]
        lam_local *= ((n + 1) * (n + 2)) ** 0.5
        return lam_local

    def Local_Energy(self, x, v, lam, N_int, batch_size, GPU_device):
        '''
        Calculates the local energies of the configurations in x

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          E_local: a tensor of dimension (n_samples, 1) containing the local
          energies of the configurations in x
        '''

        # Computes local energies
        if torch.numel(x) > 0:
            # Creates temporary variable to compute gradients
            x_tmp = x.detach().clone()
            x_tmp.requires_grad = True
            x_tmp.grad = None

            # Compute gradient
            tmp = self.log_Psi(x_tmp)
            tmp2 = torch.autograd.grad(outputs=tmp, inputs=x_tmp, \
                                       grad_outputs=torch.ones_like(tmp), create_graph=True)[0]
            gradient = tmp2.detach()
            del tmp, tmp2

            KE_local = torch.sum(gradient ** 2, axis=1)[:, None]
            del gradient

            PE_local = torch.sum(0 * x + v, axis=1)[:, None]
        else:
            KE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)
            PE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)

        lam_local = self.lambda_Local_Energy(x, lam, N_int, batch_size, GPU_device)
        E_local = KE_local + PE_local + lam_local
        del KE_local, PE_local, lam_local
        return E_local

    def Local_Energy_lap(self, x, v, lam, N_int, batch_size, GPU_device):
        '''
        Calculates the local energies of the configurations in x; uses Laplacian
        formula for kinetic energy

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          E_local: a tensor of dimension (n_samples, 1) containing the local
          energies of the configurations in x
        '''

        # Computes local energies
        if torch.numel(x) > 0:
            # Creates temporary variable to compute gradients
            x_tmp = x.detach().clone()
            x_tmp.requires_grad = True
            x_tmp.grad = None

            # Compute gradient
            tmp = self.log_Psi(x_tmp)
            tmp2 = torch.autograd.grad(outputs=tmp, inputs=x_tmp, \
                                       grad_outputs=torch.ones_like(tmp), create_graph=True)[0]
            gradient = tmp2.detach()

            # Compute second derivatives
            tmp3 = [torch.autograd.grad(outputs=tmp2[:, i], inputs=x_tmp,
                                        grad_outputs=torch.ones_like(tmp2[:, i]),
                                        retain_graph=True)[0][:, i].detach()
                    for i in range(x_tmp.shape[1])]
            second_ders = torch.stack(tmp3, axis=1)
            del tmp, tmp2, tmp3, x_tmp

            KE_local = torch.sum(-gradient ** 2 - second_ders, axis=1)[:, None]
            del gradient, second_ders

            PE_local = torch.sum(0 * x + v, axis=1)[:, None]
        else:
            KE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)
            PE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)

        lam_local = self.lambda_Local_Energy(x, lam, N_int, batch_size, GPU_device)
        E_local = KE_local + PE_local + lam_local
        del KE_local, PE_local, lam_local
        return E_local

    def Energy_Estimate(self, x_sorted, chain_idx_sorted, n_chains, GPU_device,
                        v, lam, N_int, batch_size):
        '''
        Estimates the energies of the configurations in x_sorted

        Parameters:
          x_sorted: a list of tensors, the nth of which contains the sampled
          particle configurations of n particles and is of dimension (#, n),
          where # denotes the number of samples drawn of particle number n
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
          which the corresponding particle configuration in x_sorted came.
          n_chains: number of MCMC chains
          GPU_device: the GPU device under use (i.e. cuda)
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory

        Returns:
          E: The estimated energy
          E_std: The estimated standard deviation of E
        '''

        # Calculates local energies and the corresponding estimated energy
        Local_Energies = []
        for x in x_sorted:
            Local_Energies.append(self.Local_Energy(x, v, lam, N_int, batch_size, GPU_device))
        Local_Energies = torch.cat(Local_Energies)
        E = torch.mean(Local_Energies)

        # Calculates standard deviation of E by binning across the MCMC chains
        chain_idx_sorted_flat = torch.cat(chain_idx_sorted)
        E_means_chains = \
            torch.tensor([torch.sum(Local_Energies[:, 0] * (chain_idx_sorted_flat == i)) / \
                          torch.sum(chain_idx_sorted_flat == i) \
                          for i in range(n_chains)], device=GPU_device)
        E_std = torch.std(E_means_chains) / ((n_chains) ** 0.5)

        return E, E_std

    def lambda_integral_grad_sum(self, x, lam, N_int, batch_size, GPU_device):
        '''
        Calculates the gradient of the integral part of the lambda term in the
        Hamiltonian, for the configurations in x

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          integral_grad_sum1, integral_grad_sum2: a tensor of dimension
          (n_samples, 1) containing the sum of the integral part of the lambda term
          w.r.t. the parameters of DS1, DS2, respectively, for the configurations
          in x
        '''

        x_int_points = torch.linspace(0, self.L, N_int, device=GPU_device)
        n = x.shape[1]

        integral_grad_sum1 = []
        integral_grad_sum2 = []
        for param in self.DS1.parameters():
            integral_grad_sum1.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
        for param in self.DS2.parameters():
            integral_grad_sum2.append(torch.cuda.FloatTensor(param.shape).fill_(0.))

        for k in range(np.ceil(x.shape[0] / batch_size).astype(int)):
            x_tmp = x[k * batch_size:(k + 1) * batch_size, :]
            Psi_denominator = self.Psi(x_tmp)[:, 0].detach()
            x_int = torch.cat((x_tmp[None, :, :].repeat(N_int, 1, 1),
                               x_int_points[:, None, None].repeat(1, x_tmp.shape[0], 2)), axis=2)
            x_int = x_int.reshape(N_int * x_tmp.shape[0], x_tmp.shape[1] + 2)

            integrand_vals = self.Psi(x_int)
            integrand_vals = integrand_vals.reshape(N_int, x_tmp.shape[0], 1)[:, :, 0]
            integrals = torch.trapz(integrand_vals, dx=self.L / (N_int - 1), dim=0)
            tmp = lam * ((n + 1) * (n + 2)) ** 0.5 * integrals / Psi_denominator
            grad_sum_tmp1 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS1.parameters(),
                                                grad_outputs=torch.ones_like(tmp))
            for i in range(len(list(self.DS1.parameters()))):
                integral_grad_sum1[i] += grad_sum_tmp1[i]
            del grad_sum_tmp1, tmp

            integrand_vals = self.Psi(x_int)
            integrand_vals = integrand_vals.reshape(N_int, x_tmp.shape[0], 1)[:, :, 0]
            integrals = torch.trapz(integrand_vals, dx=self.L / (N_int - 1), dim=0)
            tmp = lam * ((n + 1) * (n + 2)) ** 0.5 * integrals / Psi_denominator
            grad_sum_tmp2 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS2.parameters(),
                                                grad_outputs=torch.ones_like(tmp))
            for i in range(len(list(self.DS2.parameters()))):
                integral_grad_sum2[i] += grad_sum_tmp2[i]
            del grad_sum_tmp2, tmp

            del x_tmp, Psi_denominator, x_int, integrand_vals, integrals

        return integral_grad_sum1, integral_grad_sum2

    def lambda_Local_Energy_and_grad_sum(self, x, lam, N_int, batch_size, GPU_device):
        '''
        Calculates the local energy of the lambda term in the Hamiltonian and its
        gradient, for the configurations in x

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          lam_local: a tensor of dimension (n_samples, 1) containing the local
          energies of the lambda term for the configurations in x
          lam_log_Psi_grad_sum1, lam_log_Psi_grad_sum2: a tensor of dimension
          (n_samples, 1) containing the sum of the local lambda energy times the
          gradient of the logarithm of Psi w.r.t. the parameters of DS1, DS2, respectively,
          for the configurations in x
        '''

        x_tmp = x.clone()
        x_tmp.requires_grad = True
        x_tmp.grad = None

        lam_local = self.lambda_Local_Energy(x, lam, N_int, batch_size, GPU_device).detach()

        log_Psi_val = self.log_Psi(x_tmp)
        tmp = 1 / 2 * lam_local * log_Psi_val
        lam_log_Psi_grad_sum1 = torch.autograd.grad(outputs=tmp,
                                                    inputs=self.DS1.parameters(),
                                                    grad_outputs=torch.ones_like(tmp))

        log_Psi_val = self.log_Psi(x_tmp)
        tmp = 1 / 2 * lam_local * log_Psi_val
        lam_log_Psi_grad_sum2 = torch.autograd.grad(outputs=tmp,
                                                    inputs=self.DS2.parameters(),
                                                    grad_outputs=torch.ones_like(tmp))

        lam_log_Psi_grad_sum1 = list(lam_log_Psi_grad_sum1)
        lam_log_Psi_grad_sum2 = list(lam_log_Psi_grad_sum2)

        lam_integral_grad_sum1, lam_integral_grad_sum2 = \
            self.lambda_integral_grad_sum(x, lam, N_int, batch_size, GPU_device)
        for i in range(len(list(self.DS1.parameters()))):
            lam_log_Psi_grad_sum1[i] += lam_integral_grad_sum1[i]
        for i in range(len(list(self.DS2.parameters()))):
            lam_log_Psi_grad_sum2[i] += lam_integral_grad_sum2[i]

        del x_tmp, log_Psi_val, tmp, lam_integral_grad_sum1, lam_integral_grad_sum2
        return lam_local, lam_log_Psi_grad_sum1, lam_log_Psi_grad_sum2

    def Local_Gradient_Energy(self, x, v, lam, N_int, batch_size, GPU_device):
        '''
        Calculates the local energies and the local gradient of the energy w.r.t.
        neural network parameters

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          E_local: a tensor of dimension (n_samples, 1) containing the local
          energies of the configurations in x
          log_Psi_grad_sum1, log_Psi_grad_sum2: a tensor of dimension
          (n_samples, 1) containing the sum of the gradient of the logarithm of Psi
          w.r.t. the parameters of DS1, DS2, respectively, for configurations in x
          E_log_Psi_grad_sum1, E_log_Psi_der_sum2: a tensor of dimension
          (n_samples, 1) containing the sum of the local energy times the gradient
          of the logarithm of Psi w.r.t. the parameters of DS1, DS2, respectively,
          for the configurations in x
        '''

        # Computes local energies
        if torch.numel(x) > 0:
            # Creates temporary variable to compute gradients
            x_tmp = x.detach().clone()
            x_tmp.requires_grad = True
            x_tmp.grad = None

            # Compute gradient
            tmp = self.log_Psi(x_tmp)
            tmp2 = torch.autograd.grad(outputs=tmp, inputs=x_tmp, \
                                       grad_outputs=torch.ones_like(tmp))[0]
            gradient = tmp2.detach()
            del tmp, tmp2
            KE_local = torch.sum(gradient ** 2, axis=1)[:, None]
            del gradient

            PE_local = torch.sum(0 * x + v, axis=1)[:, None]
        else:
            KE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)
            PE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)

        lam_local, lam_log_Psi_grad_sum1, lam_log_Psi_grad_sum2 = \
            self.lambda_Local_Energy_and_grad_sum(x, lam, N_int, batch_size, GPU_device)
        E_local = KE_local + PE_local + lam_local
        # del KE_local, PE_local

        # Computes the sum of the gradient of log_Psi wrt neural network parameters
        x.grad = None
        tmp = self.log_Psi(x)
        # Accumulated gradient is naturally the sum
        log_Psi_grad_sum1 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS1.parameters(),
                                                grad_outputs=torch.ones_like(tmp))
        tmp = self.log_Psi(x)
        log_Psi_grad_sum2 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS2.parameters(),
                                                grad_outputs=torch.ones_like(tmp))

        # Computes the sum of the local energy times the gradient of the logarithm
        # of Psi w.r.t. neural network parameters
        x_tmp = x.detach().clone()
        x_tmp.requires_grad = True
        x_tmp.grad = None
        log_Psi_val = self.log_Psi(x_tmp)
        gradient = torch.autograd.grad(outputs=log_Psi_val, inputs=x_tmp, \
                                       grad_outputs=torch.ones_like(log_Psi_val),
                                       create_graph=True)[0]
        KE_local_tmp = torch.sum(gradient ** 2, axis=1)[:, None]
        tmp = 1 / 2 * KE_local_tmp + (KE_local_tmp.detach() + PE_local) * log_Psi_val
        # Accumulated gradient is naturally the sum
        KE_PE_log_Psi_grad_sum1 = torch.autograd.grad(outputs=tmp,
                                                      inputs=self.DS1.parameters(),
                                                      grad_outputs=torch.ones_like(tmp))
        del log_Psi_val, gradient, KE_local_tmp, tmp

        x_tmp = x.detach().clone()
        x_tmp.requires_grad = True
        x_tmp.grad = None
        log_Psi_val = self.log_Psi(x_tmp)
        gradient = torch.autograd.grad(outputs=log_Psi_val, inputs=x_tmp, \
                                       grad_outputs=torch.ones_like(log_Psi_val),
                                       create_graph=True)[0]
        KE_local_tmp = torch.sum(gradient ** 2, axis=1)[:, None]
        tmp = 1 / 2 * KE_local_tmp + (KE_local_tmp.detach() + PE_local) * log_Psi_val
        # Accumulated gradient is naturally the sum
        KE_PE_log_Psi_grad_sum2 = torch.autograd.grad(outputs=tmp,
                                                      inputs=self.DS2.parameters(),
                                                      grad_outputs=torch.ones_like(tmp))
        del log_Psi_val, gradient, KE_local_tmp, tmp

        E_log_Psi_grad_sum1 = []
        E_log_Psi_grad_sum2 = []
        for k, param in enumerate(self.DS1.parameters()):
            E_log_Psi_grad_sum1.append((KE_PE_log_Psi_grad_sum1[k] + lam_log_Psi_grad_sum1[k]).detach())
        for k, param in enumerate(self.DS2.parameters()):
            E_log_Psi_grad_sum2.append((KE_PE_log_Psi_grad_sum2[k] + lam_log_Psi_grad_sum2[k]).detach())

        return E_local, log_Psi_grad_sum1, log_Psi_grad_sum2, E_log_Psi_grad_sum1, E_log_Psi_grad_sum2

    def Local_Gradient_Energy_lap(self, x, v, lam, N_int, batch_size, GPU_device):
        '''
        Calculates the local energies and the local gradient of the energy w.r.t.
        neural network parameters; uses laplacian formula for kinetic energy

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
          n_samples samples of n_particles particle positions
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          E_local: a tensor of dimension (n_samples, 1) containing the local
          energies of the configurations in x
          log_Psi_grad_sum1, log_Psi_grad_sum2 : a tensor of dimension
          (n_samples, 1) containing the sum of the gradient of the logarithm of Psi
          w.r.t. the parameters of DS1, DS2, respectively, for configurations in x
          E_log_Psi_grad_sum1, E_log_Psi_der_sum2: a tensor of dimension
          (n_samples, 1) containing the sum of the local energy times the gradient
          of the logarithm of Psi w.r.t. the parameters of DS1, DS2, respectively,
          for the configurations in x
        '''

        # Computes local energies
        if torch.numel(x) > 0:
            # Creates temporary variable to compute gradients
            x_tmp = x.detach().clone()
            x_tmp.requires_grad = True
            x_tmp.grad = None

            # Compute gradient
            tmp = self.log_Psi(x_tmp)
            tmp2 = torch.autograd.grad(outputs=tmp, inputs=x_tmp, \
                                       grad_outputs=torch.ones_like(tmp), create_graph=True)[0]
            gradient = tmp2.detach()

            # Compute second derivatives
            tmp3 = [torch.autograd.grad(outputs=tmp2[:, i], inputs=x_tmp,
                                        grad_outputs=torch.ones_like(tmp2[:, i]),
                                        retain_graph=True)[0][:, i].detach()
                    for i in range(x_tmp.shape[1])]
            second_ders = torch.stack(tmp3, axis=1)
            del tmp, tmp2, tmp3, x_tmp

            KE_local = torch.sum(-gradient ** 2 - second_ders, axis=1)[:, None]
            del gradient, second_ders

            PE_local = torch.sum(0 * x + v, axis=1)[:, None]
        else:
            KE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)
            PE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)

        lam_local, lam_log_Psi_grad_sum1, lam_log_Psi_grad_sum2 = \
            self.lambda_Local_Energy_and_grad_sum(x, lam, N_int, batch_size, GPU_device)
        E_local = KE_local + PE_local + lam_local
        # del KE_local, PE_local

        # Computes the sum of the gradient of log_Psi wrt neural network parameters
        x.grad = None
        tmp = self.log_Psi(x)
        # Accumulated gradient is naturally the sum
        log_Psi_grad_sum1 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS1.parameters(),
                                                grad_outputs=torch.ones_like(tmp))
        tmp = self.log_Psi(x)
        log_Psi_grad_sum2 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS2.parameters(),
                                                grad_outputs=torch.ones_like(tmp))

        # Computes the sum of the local energy times the gradient of the logarithm
        # of Psi w.r.t. neural network parameters
        x_tmp = x.detach().clone()
        x_tmp.requires_grad = True
        x_tmp.grad = None
        tmp = self.log_Psi(x_tmp)
        tmp2 = torch.autograd.grad(outputs=tmp, inputs=x_tmp, \
                                   grad_outputs=torch.ones_like(tmp))[0]
        gradient = tmp2
        KE_local_tmp = torch.sum(gradient ** 2, axis=1)[:, None]

        x.grad = None
        tmp = self.log_Psi(x)
        tmp = 1 / 2 * KE_local_tmp + (KE_local + PE_local) * tmp
        # tmp = (KE_local + PE_local)*tmp
        KE_PE_log_Psi_grad_sum1 = torch.autograd.grad(outputs=tmp,
                                                      inputs=self.DS1.parameters(),
                                                      grad_outputs=torch.ones_like(tmp))

        x_tmp = x.detach().clone()
        x_tmp.requires_grad = True
        x_tmp.grad = None
        tmp = self.log_Psi(x_tmp)
        tmp2 = torch.autograd.grad(outputs=tmp, inputs=x_tmp, \
                                   grad_outputs=torch.ones_like(tmp))[0]
        gradient = tmp2
        KE_local_tmp = torch.sum(gradient ** 2, axis=1)[:, None]

        x.grad = None
        tmp = self.log_Psi(x)
        tmp = 1 / 2 * KE_local_tmp + (KE_local + PE_local) * tmp
        # tmp = (KE_local + PE_local)*tmp
        KE_PE_log_Psi_grad_sum2 = torch.autograd.grad(outputs=tmp,
                                                      inputs=self.DS2.parameters(),
                                                      grad_outputs=torch.ones_like(tmp))

        E_log_Psi_grad_sum1 = []
        E_log_Psi_grad_sum2 = []
        for k, param in enumerate(self.DS1.parameters()):
            E_log_Psi_grad_sum1.append((KE_PE_log_Psi_grad_sum1[k] + lam_log_Psi_grad_sum1[k]).detach())
        for k, param in enumerate(self.DS2.parameters()):
            E_log_Psi_grad_sum2.append((KE_PE_log_Psi_grad_sum2[k] + lam_log_Psi_grad_sum2[k]).detach())

        return E_local, log_Psi_grad_sum1, log_Psi_grad_sum2, E_log_Psi_grad_sum1, E_log_Psi_grad_sum2

    def Energy_Grad_Estimate(self, x_sorted, n, chain_idx_sorted, n_chains,
                             GPU_device, v, lam, N_int, batch_size):
        '''
        Estimates the gradient of the energy w.r.t. neural network parameters. Also
        estimates the energy and its standard deviation.

        Parameters:
          x_sorted: a list of tensors, the nth of which contains the sampled
          particle configurations containing n particles and is of dimension (#, n),
          where # denotes the number of samples drawn of particle number n
          n: a tensor containing the particle numbers of configurations in x_sorted
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
          which the corresponding particle configuration in x_sorted came.
          n_chains: number of chains
          GPU_device: the GPU device under use (i.e. cuda)
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          N_int: number of integration points used to compute the numerical integral
          for the lambda term
          batch_size: the size of batches over which the integrals are calculated;
          performing the numerical integral is memory intensive, and hence we batch
          over entries to save memory
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          E: the estimated energy
          E_std: the estimated standard deviation of E
          E_grad1, E_grad2: gradient of E wrt parameters of DS1, DS2, respectively
        '''

        # Initializes variables
        Local_Energies = []
        log_Psi_grad1 = []
        log_Psi_grad2 = []
        E_log_Psi_grad1 = []
        E_log_Psi_grad2 = []
        for param in self.DS1.parameters():
            log_Psi_grad1.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
            E_log_Psi_grad1.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
        for param in self.DS2.parameters():
            log_Psi_grad2.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
            E_log_Psi_grad2.append(torch.cuda.FloatTensor(param.shape).fill_(0.))

            # Calculates local energies and local energy gradients
        for x in x_sorted:
            Local_Energies_tmp, log_Psi_grad_sum1, log_Psi_grad_sum2, \
            E_log_Psi_grad_sum1, E_log_Psi_grad_sum2 = \
                self.Local_Gradient_Energy(x, v, lam, N_int, batch_size, GPU_device)
            Local_Energies.append(Local_Energies_tmp)
            for i in range(len(list(self.DS1.parameters()))):
                log_Psi_grad1[i] += log_Psi_grad_sum1[i]
                E_log_Psi_grad1[i] += E_log_Psi_grad_sum1[i]
            for i in range(len(list(self.DS2.parameters()))):
                log_Psi_grad2[i] += log_Psi_grad_sum2[i]
                E_log_Psi_grad2[i] += E_log_Psi_grad_sum2[i]

        for i in range(len(list(self.DS1.parameters()))):
            log_Psi_grad1[i] *= 1 / n.shape[0]
            E_log_Psi_grad1[i] *= 1 / n.shape[0]
        for i in range(len(list(self.DS2.parameters()))):
            log_Psi_grad2[i] *= 1 / n.shape[0]
            E_log_Psi_grad2[i] *= 1 / n.shape[0]

        # Estimates energy and its standard deviations
        Local_Energies = torch.cat(Local_Energies)
        E = torch.mean(Local_Energies)
        chain_idx_sorted_flat = torch.cat(chain_idx_sorted)
        E_means_chains = \
            torch.tensor([torch.sum(Local_Energies[:, 0] * (chain_idx_sorted_flat == i)) / \
                          torch.sum(chain_idx_sorted_flat == i) \
                          for i in range(n_chains)], device=GPU_device)
        E_std = torch.std(E_means_chains) / ((n_chains) ** 0.5)

        # Estimates gradients of energy wrt neural network parameters
        E_grad1 = []
        E_grad2 = []
        for i in range(len(list(self.DS1.parameters()))):
            tmp = 2 * (E_log_Psi_grad1[i] - E * log_Psi_grad1[i])
            E_grad1.append(tmp)
        for i in range(len(list(self.DS2.parameters()))):
            tmp = 2 * (E_log_Psi_grad2[i] - E * log_Psi_grad2[i])
            E_grad2.append(tmp)

        return E, E_std, E_grad1, E_grad2

    def ADAM_update(self, x_sorted, n, chain_idx_sorted, n_chains, GPU_device, v,
                    lam, t, m1, v1, m2, v2, beta_1, beta_2, lr, lr_q, N_int, batch_size):
        '''
        Performs a single ADAM update

        Parameters:
          x_sorted: a list of tensors, the nth of which contains the sampled
          particle configurations containing n particles and is of dimension (#, n),
          where # denotes the number of samples drawn of particle number n
          n: a tensor containing the particle numbers of configurations in x_sorted
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
          which the corresponding particle configuration in x_sorted came.
          n_chains: number of separate MCMC chains
          GPU_device: the GPU device under use (i.e. cuda)
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          t: current ADAM iteration
          m1, m2: first moment estimate for DS1, DS2, respectively
          v1, v2: second moment estimate for DS1, DS2, respectively
          beta_1, beta_2: ADAM parameters
          lr: learning rate for parameters of DS1, DS
          lr: learning rate for parameters of q_n (have the option to use a
          different rate)

        Returns:
          E: the estimated energy
          E_std: the estimated standard deviation of E
          m1, m2: updated first moment estimate for DS1, DS2, respectively
          v1, v2: updated second moment estimate for DS1, DS2, respectively
        '''

        eps = 1e-8  # ADAM parameter

        # Estimates gradients
        E, E_std, E_grad1, E_grad2 = self.Energy_Grad_Estimate(x_sorted, n, \
                                                               chain_idx_sorted, n_chains, GPU_device, v, lam, N_int,
                                                               batch_size)

        # Performs ADAM update on DS1
        for i, param in enumerate(self.DS1.parameters()):
            m1[i] = beta_1 * m1[i] + (1 - beta_1) * E_grad1[i]
            v1[i] = beta_2 * v1[i] + (1 - beta_2) * E_grad1[i] ** 2
            m1_hat = m1[i] / (1 - beta_1 ** t)
            v1_hat = v1[i] / (1 - beta_2 ** t)
            if i < 3:
                param.data -= lr_q * m1_hat / (torch.sqrt(v1_hat) + eps)
            else:
                param.data -= lr * m1_hat / (torch.sqrt(v1_hat) + eps)

        # Performs ADAM update on DS2
        for i, param in enumerate(self.DS2.parameters()):
            m2[i] = beta_1 * m2[i] + (1 - beta_1) * E_grad2[i]
            v2[i] = beta_2 * v2[i] + (1 - beta_2) * E_grad2[i] ** 2
            m2_hat = m2[i] / (1 - beta_1 ** t)
            v2_hat = v2[i] / (1 - beta_2 ** t)
            param.data -= lr * m2_hat / (torch.sqrt(v2_hat) + eps)

        return E, E_std, m1, v1, m2, v2

    def minimize_energy_ADAM(self, n_samples, n_chains, p_pm, GPU_device,
                             v, lam, beta_1, beta_2, lr, lr_q, n_iters,
                             N_int, batch_size):
        '''
        Minimizes energy with ADAM, and outputs the estimated energy and particle
        number at each iteration.

        Parameters:
          n_samples: number of samples drawn for each MCMC chain
          n_chains: number of separate MCMC chains
          p_pm: probability of increasing or decreasing particle number at each
          configuration proposal
          GPU_device: the GPU device under use (i.e. cuda)
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          beta_1, beta_2: ADAM parameters
          lr: learning rate for parameters of DS1, DS
          lr: learning rate for parameters of q_n (have the option to use a
          different rate)
          n_iters: number of iterations of ADAM to be performed

        Returns:
          Es: an array containing the estimated energies at each iteration
          E_stds: an array containing the estimated standard deviations of energy
          at each iteration
          n_means: an array containing the estimated mean particle number at each
          iteration
          n_stds: an array containing the estimated standard deviation of particle
          number at each iteration
        '''

        # Initializes variables
        Es = torch.cuda.FloatTensor(n_iters).fill_(0.)
        E_stds = torch.cuda.FloatTensor(n_iters).fill_(0.)
        n_means = torch.cuda.FloatTensor(n_iters).fill_(0.)
        n_stds = torch.cuda.FloatTensor(n_iters).fill_(0.)
        m1, v1 = [], []
        for param in self.DS1.parameters():
            m1.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
            v1.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
        m2, v2 = [], []
        for param in self.DS2.parameters():
            m2.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
            v2.append(torch.cuda.FloatTensor(param.shape).fill_(0.))
        n_means[-1] = 2 * round(self.DS1.q_n_mean.data.item() / 2)

        # Performs ADAM for n_iters number of iterations
        for t in range(1, n_iters + 1):
            # Generates MCMC data
            # n_0 = int(n_means[t-2].item())
            n_0 = 2 * round(n_means[t - 2].item() / 2)
            x_sorted, n, chain_idx_sorted = \
                self.GenerateMCMCSamples_FockSpace(n_samples, n_chains, p_pm, n_0,
                                                   GPU_device)

            # Performs ADAM update
            E, E_std, m1, v1, m2, v2 = \
                self.ADAM_update(x_sorted, n, chain_idx_sorted, n_chains, GPU_device, v,
                                 lam, t, m1, v1, m2, v2, beta_1, beta_2, lr, lr_q,
                                 N_int, batch_size)

            # Stores energy, particle number, and the standard deviations thereof
            Es[t - 1] = E.detach().clone()
            E_stds[t - 1] = E_std.detach().clone()
            n_mean = torch.mean(n.to(torch.float))
            n_std = torch.std(n.to(torch.float))
            n_means[t - 1] = n_mean.detach().clone()
            n_stds[t - 1] = n_std.detach().clone()

            # Prints results of current iteration
            print("Iteration: " + str(t) + "/" + str(n_iters))
            print("Energy: " + str(round(E.item(), 3)) + " +- " \
                  + str(round(E_std.item(), 3)))
            print("Number of particles: " + str(round(n_mean.item(), 3)) + \
                  " +- " + str(round(n_std.item(), 3)))
            '''print("q_n mean: " + str(round(self.DS1.q_n_mean.data.item(), 3)))
            w = torch.log(1 + torch.exp(self.DS1.q_n_inv_softplus_width))
            print("q_n width: " + str(round(w.item(), 3)))
            s = torch.log(1 + torch.exp(self.DS1.q_n_inv_softplus_slope))
            print("q_n slope: " + str(round(s.item(), 3)))'''
            print("\n")

        return Es, E_stds, n_means, n_stds


    def plot_energy(self, Es, E_stds, E_exact, start, end):
        '''
        Plots energy vs. iteration from start to end.

        Parameters:
          Es: an array containing the estimated energies at each iteration
          E_stds: an array containing the estimated standard deviations of energy
          E_exact: the exact ground state energy
          start: the starting iteration from which data is plotted
          end: the last iteration from which data is plotted
        '''

        iterations = np.arange(start, end)

        # Plots energy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=200)
        ax1.plot(iterations, Es.cpu()[start:end], linewidth=.7, label='Energy')
        ax1.fill_between(iterations, (Es - E_stds).cpu()[start:end],
                         (Es + E_stds).cpu()[start:end], facecolor='blue',
                         alpha=0.25, interpolate=True)
        ax1.plot(iterations, E_exact + 0 * Es.cpu()[start:end], linewidth=1,
                 label='Exact Energy')
        ax1.set_title('Energy vs. Iteration', fontsize=10)
        ax1.set_ylabel('Energy', fontsize=8)
        ax1.set_xlabel('Iteration', fontsize=8)
        ax1.legend(prop={'size': 7})
        ax1.set_xticklabels(ax1.get_xticks().astype(int), fontsize=6)
        ax1.set_yticklabels(np.round(ax1.get_yticks(), 2), fontsize=6)

        # Plots error in energy and relative standard deviation
        Energy_errors = (torch.abs((Es.cpu() - E_exact) / E_exact))[start:end]
        ax2.plot(iterations, Energy_errors, linewidth=.7, label='Relative Error')
        ax2.plot(iterations, (torch.abs(E_stds / Es)).cpu()[start:end],
                 linewidth=.7, label='Relative St. Dev.')
        ax2.set_yscale('log')
        ax2.set_title('Energy Discrepancies vs. Iteration', fontsize=10)
        ax2.set_ylabel('Value', fontsize=8)
        ax2.set_xlabel('Iteration', fontsize=8)
        ax2.legend(prop={'size': 7})
        ax2.set_xticklabels(ax2.get_xticks().astype(int), fontsize=6)
        ax2.set_yticklabels(np.round(ax2.get_yticks(), 2), fontsize=6)

        fig.tight_layout()
        fig.show()


    def Exact_energy_density_Quad(self, v, lam, L, N_max):
        '''
        Computes exact ground state energy density. Requires |lam| <= v/2 in order to
        be well-defined

        Parameters:
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          L: system length
          N_max: maximum integer over which the sum extends (the expression for
            energy density extends from N=-\infty to \infty, but we restrict this from
            -N_max to N_max)

          Returns:
            epsilon: Exact energy density
        '''

        ps = 2 * np.pi / L * np.arange(-1 * N_max, N_max + 1)
        integrand_vals = -4 * lam ** 2 / (ps ** 2 + v) / (1 + np.sqrt(1 - 4 * lam ** 2 / (ps ** 2 + v) ** 2))
        integral = 1 / L * np.sum(integrand_vals)
        epsilon = 1 / 2 * integral
        return epsilon


    def v_Over_u_p(self, p, v, lam):
        '''
        Computes ratio of Bogoliubov coefficients v_p/u_p

        Parameters:
          p: momentum
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)

        Returns:
          val: Value of the ratio v_p/u_p
        '''

        numer = -1 * (p ** 2 + v - ((p ** 2 + v) ** 2 - 4 * lam ** 2) ** 0.5) ** (0.5)
        denom = (p ** 2 + v + ((p ** 2 + v) ** 2 - 4 * lam ** 2) ** 0.5) ** (0.5)
        val = numer / denom
        return val


    def Exact_P_n(self, v, lam, L, n_max, N_max):
        '''
        Computes exact n-particle probability distribution

        Parameters:
          v: coefficient of particle number term
          lam: coefficient of (psi(x)*psi(x) + h.c.)
          L: system length
          n_max: largest particle number whose probability is to be computed
          N_max: maximum integer over which the sum extends (the expression for
            energy density extends from N=-\infty to \infty, but we restrict this from
            -N_max to N_max)

          Returns:
            P_n_exact: Exact n-particle probability distribution
        '''

        n_range = np.arange(0, n_max + 1, 2)  # Range of particle numbers to analyze
        P_n_exact = np.zeros(n_range.shape)

        # Computes probabilities
        for i, n in enumerate(n_range):
            if n == 0:
                P_n_val = 1
            else:
                # Computes all possible combinations of momentum modes that add up to n/2
                combs = itertools.combinations_with_replacement(np.identity(N_max + 1, dtype=int), n // 2)
                ls = np.array(list(sum(c) for c in combs))

                # Computes contribution of each combination to P_n
                ps = 2 * np.pi / L * np.arange(0, N_max + 1)
                v_Over_u_p_vals = np.tile(self.v_Over_u_p(ps, v, lam), (ls.shape[0], 1))
                v_Over_u_p_vals[:, 0] *= 1 / 2
                v_Over_u_p_product = np.prod(v_Over_u_p_vals ** (2 * ls), axis=1)
                combinatorial_term = np.exp(gammaln(2 * ls[:, 0] + 1) - 2 * gammaln(ls[:, 0] + 1))
                vals = v_Over_u_p_product * combinatorial_term
                P_n_val = np.sum(vals)

            P_n_exact[i] = P_n_val

        P_n_exact *= 1 / np.sum(P_n_exact)

        return P_n_exact


    def plot_P_n(self, P_n_approx, P_n_exact):
        '''
        Plots approximate and exact particle number probability distributions

        Parameters:
          P_n_approx: Approximate particle number distribution
          P_n_exact: Exact particle number distribution
        '''
        n_range = np.arange(0, 2 * P_n_exact.shape[0], 2)

        fig, ax = plt.subplots(figsize=(3.25, 2.8), dpi=200)
        ax.bar(n_range - 0.4, P_n_approx, width=0.8, label='Approximate', align='center', edgecolor='black', linewidth=0.8)
        ax.bar(n_range + 0.4, P_n_exact, width=0.8, label='Exact', align='center', edgecolor='black', linewidth=0.8)
        ax.set_title('Particle Number Distributions', fontsize=10)
        ax.set_ylabel('P_n', fontsize=8)
        ax.set_xlabel('n', fontsize=8)
        ax.legend(prop={'size': 7})
        ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=6)
        ax.set_yticklabels(np.round(ax.get_yticks(), 1), fontsize=6)

        fig.tight_layout()
        fig.show()