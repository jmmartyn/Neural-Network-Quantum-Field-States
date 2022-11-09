import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from deep_sets import Deep_Sets


'''
Class for the Deep Sets Neural-Network Quantum Field State, building off the 
architecture developed in https://arxiv.org/abs/2112.11957. In this ansatz, 
each n-particle state is modelled as \varphi_n = DS1({x_i})*DS2({x_i-x_j}),
for deep sets networks DS1 and DS2. This ansatz is specially designed for the 
Lieb-Liniger model (hence the subscript '_LL').
'''


class NQFS_LL():
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
        self.DS1 = Deep_Sets(input_dim1, DS_width, DS_depth_phi, DS_depth_rho)
        input_dim2 = 1  # The input to DS2 is a 1d embedding
        self.DS2 = Deep_Sets(input_dim2, DS_width, DS_depth_phi, DS_depth_rho)

        # Register and initialize parameters of q_n
        '''q_n is modelled as a smoothed rectangular pulse; its parameters are 
           included in DS1 for optimization. Explicitly, 
           q_n = 1/(1+e^{-s*(n-c_1)})*1/(1+e^{s*(n-c_2)}) for 0 < c_1 < c_2 and s > 0,
           where q_n_mean = (c_1 + c_2)/2, q_n_width = (c_2 - c_1), q_n_slope = s.'''
        self.DS1.register_parameter(name='q_n_mean',
                                    param=torch.nn.Parameter(torch.tensor(3.)))
        # Parametrize the inverse softplus of the width to ensure width is positive
        self.DS1.register_parameter(name='q_n_inv_softplus_width',
                                    param=torch.nn.Parameter(torch.tensor(6.)))
        # Parameterize the inverse softplus of the slope to ensure slope is positive
        self.DS1.register_parameter(name='q_n_inv_softplus_slope',
                                    param=torch.nn.Parameter(torch.tensor(1.)))

        # Set system parameters
        self.L = L
        self.periodic = periodic

    def Embedding1(self, x):
        '''
        Constructs the feature embedding of particle positions x for DS1

        Parameters:
          x: an tensor of dimension (n_samples, n_particles) representing n_samples
            samples of n_particles particle positions

        Returns:
          embedded_data: a tensor of dimension (n_samples, n_particles, 2)
            containing the embedding of the particle positions
        '''

        # If periodic, embedding = (sin(2*pi*x/L), cos(2*pi*x/L))
        # If not periodic, embedding = (x/L, 1-x/L)
        x_norm = x[:, :, None] / self.L
        if self.periodic:
            sines = torch.sin(2 * np.pi * x_norm)
            cosines = torch.cos(2 * np.pi * x_norm)
            embedded_data = torch.cat((sines, cosines), axis=2)
        else:
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
            (n_samples, n_particles*(n_particles-1)/2, 1) containing the embedding
            of the particle positions
        '''

        # Determines interparticle seperations {x_i - x_j} for i < j
        idx = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
        interparticle_seps = (x[:, :, None] - x[:, None, :])[:, idx[0], idx[1]]

        # If periodic, embedding = cos(2*pi*(x_i-x_j)/L)
        # If not periodic, embedding = ((x_i-x_j)/L)^2
        if self.periodic:
            embedded_data = torch.cos(2 * np.pi / self.L * interparticle_seps[..., None])
        else:
            embedded_data = (interparticle_seps[..., None] / self.L) ** 2

        return embedded_data

    def log_Psi(self, x, m, g):
        '''
        Calculates log(Psi(x))

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
            n_samples samples of n_particles particle positions. x may contain
            entries corresponding to non-existent particles; we denote these by a
            'nan' value
          m: particle mass
          g: Lieb-Liniger interaction strength

        Returns:
          val: a tensor of dimension (n_samples, 1) containing the values of
            log_Psi(x)
        '''

        # Contribution from DS1
        x_emb1 = self.Embedding1(x)
        # Constructs mask; non-existent particle entries in x are denoted by 'nan'
        mask1 = ((~x.isnan())[..., None]).detach()
        val = self.log_DS1(x_emb1.nan_to_num(), mask1)

        # Contribution from DS2
        n = torch.sum(~x.isnan(), axis=1)[:, None]
        x_emb2 = self.Embedding2(x)
        mask2 = (~x_emb2.isnan()).detach()
        val += (n >= 2) * self.log_DS2(x_emb2.nan_to_num(), mask2)

        # Contribution from L^(-n/2)
        val += -1 / 2 * (n * np.log(self.L))

        # Contribution from Jastrow factor
        val += (n >= 2) * self.log_jastrow(x, m, g, n, mask1)

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

    def log_jastrow(self, x, m, g, n, mask1):
        '''
        Returns log(Jastrow factor) = log(\prod_{i<j}(|x_i-x_j| + 1/(m*g)))

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) containing particle
            positions
          m: particle mass
          g: Lieb-Liniger interaction strength

        Returns:
          val: a tensor of dimension (n_samples, 1) containing the values of
            log(Jastrow factor)
        '''

        # Determines interparticle seperations {x_i - x_j} for i < j
        idx = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
        interparticle_seps = (x[:, :, None] - x[:, None, :])[:, idx[0], idx[1]]

        # Calculates Jastrow factor
        jastrows = torch.abs(interparticle_seps) + 1 / (m * g)
        val = torch.sum(torch.log(jastrows).nan_to_num(), axis=1)[:, None]

        # Accounts for normalization factor
        if x.shape[1] > 0:
            log_Selberg_vals = [(2 * torch.special.gammaln(0 * n + 1 + j) + \
                                 torch.special.gammaln(0 * n + 2 + j) - \
                                 torch.special.gammaln(1 + n + j)) \
                                for j in range(0, x.shape[1])]
            log_Selberg_vals = torch.stack(log_Selberg_vals, axis=1)
            log_Selberg = torch.sum(log_Selberg_vals * mask1, axis=1)
            val += -1 / 2 * (log_Selberg)

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

        q_n_width = torch.log(1 + torch.exp(self.DS1.q_n_inv_softplus_width))
        c_1 = 1 / 2 * (2 * self.DS1.q_n_mean - q_n_width)
        c_2 = 1 / 2 * (2 * self.DS1.q_n_mean + q_n_width)
        q_n_slope = torch.log(1 + torch.exp(self.DS1.q_n_inv_softplus_slope))
        s = q_n_slope

        val = -torch.log(1 + torch.exp(-s * (n - c_1)))
        val += -torch.log(1 + torch.exp(s * (n - c_2)))

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

    def MetropolisHastings_FockSpace(self, n_samples, n_chains, p_pm, n_0,
                                     GPU_device, m, g):
        '''
        Performs the Metropolis-Hastings algorithm in Fock space to produce
        samples of particle configurations drawn from P_n and |\varphi_n|^2. This
        runs multiple MCMC chains on GPU to parallelize data generation.

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
            such tensor is of dimension (n_chains, n), where n is the maximum
            particle number of the sampled configurations at the corresponding
            iteration.
          n_list: a list of n_samples tensors, each containing the number of
            particles of the sampled particle configurations across the MCMC
            chains at the specified iteration. Each such tensor is of dimension
            (n_chains, 1), and takes integer values.
        '''

        # Sets width of the uniform distribution used for configuration proposals
        w = 0.3 * self.L

        # Initializes n_list as the number of particles, each configuration
        # containing n_0 particles at the beginning
        n_list = [n_0 * torch.ones(n_chains, dtype=torch.long, device=GPU_device)]

        # Initializes x_list as particle positions; positions are chosen randomly; a
        # value 'nan' is used to denote the absence of a particle
        if n_0 == 0:
            x_0 = torch.cuda.FloatTensor(n_chains, 1).fill_(float('nan'))
        else:
            x_0 = self.L * torch.rand(n_chains, n_0, device=GPU_device)
        x_list = [x_0]

        # Sets current particle positions, particle number, and log(Psi)
        x = x_list[0].detach().clone()
        n = n_list[0].detach().clone()
        log_Psi_val = self.log_Psi(x, m, g)

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

            # Adds a particle at a randomly chosen position
            x_proposed = torch.cat((x, \
                                    torch.cuda.FloatTensor(n_chains, 1).fill_(float('nan'))), axis=1)
            x_proposed[add_ind, n[add_ind]] = \
                self.L * torch.cuda.FloatTensor(add_ind.shape[0]).uniform_()

            # Removes a particle at random (n_cutoff is used to ensure particle number
            # doesn't become negative)
            n_cutoff = torch.maximum(torch.cuda.FloatTensor(1).fill_(1.),
                                     n[remove_ind]).long()
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
            n_proposed[add_ind] += 1
            n_proposed[remove_ind] = torch.maximum(n_proposed[remove_ind] - 1, \
                                                   torch.cuda.FloatTensor(1).fill_(0.).long())

            # Computes acceptance ratio
            log_Psi_val_proposed = self.log_Psi(x_proposed, m, g)
            L_factor = torch.cuda.FloatTensor(n_chains, 1).fill_(1.)
            L_factor[add_ind, 0] *= self.L
            L_factor[remove_ind, 0] *= 1 / self.L
            A = torch.minimum(torch.cuda.FloatTensor(n_chains, 1).fill_(1.),
                              L_factor * torch.exp(2 * (log_Psi_val_proposed - log_Psi_val)))

            # Accepts or rejects configuration proposals (using the random numbers u
            # generated earlier), and then updates x, n, and log_Psi_val
            accept_ind = torch.where(u[1, :] < A[:, 0])[0]
            x = torch.cat((x, \
                           torch.cuda.FloatTensor(n_chains, 1).fill_(float('nan'))), axis=1)
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
                                      GPU_device, m, g):
        '''
        Generates particle configurations drawn from P_n and |\varphi_n|^2 using the
        algorithm for Metropolis-Hastings in Fock space, and subsequently removes
        MCMC burn-in phase and sorts the configurations by particle number.

        Parameters:
          n_samples: number of samples drawn for each MCMC chain
          n_chains: number of separate MCMC chains
          p_pm: probability of increasing or decreasing particle number at each
            configuration proposal
          n_0: initial particle number of each MCMC chain
          GPU_device: the GPU device under use (i.e. cuda)

        Returns:
          x_sorted: a list of tensors, the nth of which contains the sampled
            particle configurations containing n particles and is of dimension
            (#, n), where # denotes the number of drawn samples of particle number n
          n: a tensor containing the particle numbers of the configurations drawn
            from P_n and |\varphi_n|^2
          chain_idx_sorted: a list of tensors, each of the same dimension as those
            of x_sorted. An entry of such tensor denotes the MCMC chain from
            which the corresponding particle configuration came.
        '''

        # Generate samples
        x_list, n_list = self.MetropolisHastings_FockSpace(n_samples, n_chains, p_pm,
                                                           n_0, GPU_device, m, g)

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

    def Local_Energy(self, x, m, mu, V, g):
        '''
        Calculates the local energies of the configurations in x

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
            n_samples samples of n_particles particle positions
          m: particle mass
          mu: chemical potential
          V: External potential function
          g: Lieb-Liniger interaction strength

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
            tmp = self.log_Psi(x_tmp, m, g)
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

            KE_local = 1 / (2 * m) * torch.sum(-gradient ** 2 - second_ders, axis=1)[:, None]
            del gradient, second_ders

            PE_local = torch.sum(V(x) - mu, axis=1)[:, None]
        else:
            KE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)
            PE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)

        E_local = KE_local + PE_local
        del KE_local, PE_local
        return E_local

    def Energy_Estimate(self, x_sorted, chain_idx_sorted, n_chains, GPU_device,
                        m, mu, V, g):
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
          m: particle mass
          mu: chemical potential
          V: External potential function
          g: Lieb-Liniger interaction strength

        Returns:
          E: The estimated energy
          E_std: The estimated standard deviation of E
        '''

        # Calculates local energies and the corresponding estimated energy
        Local_Energies = []
        for x in x_sorted:
            Local_Energies.append(self.Local_Energy(x, m, mu, V, g))
        Local_Energies = torch.cat(Local_Energies)
        E = torch.mean(Local_Energies)

        # Calculates standard deviation of E by binning across the MCMC chains
        chain_idx_sorted_flat = torch.cat(chain_idx_sorted)
        E_means_chains = \
            torch.tensor([torch.sum(Local_Energies[:, 0] * (chain_idx_sorted_flat == i)) / \
                          torch.sum(chain_idx_sorted_flat == i) \
                          for i in range(n_chains)], device=GPU_device)
        E_std = torch.std(E_means_chains) / (n_chains ** 0.5)

        return E, E_std

    def Local_Gradient_Energy(self, x, m, mu, V, g):
        '''
        Calculates the local energies and the local gradient of the energy w.r.t.
        neural network parameters

        Parameters:
          x: a tensor of dimension (n_samples, n_particles) representing
            n_samples samples of n_particles particle positions
          m: particle mass
          mu: chemical potential
          V: External potential function
          g: Lieb-Liniger interaction strength

        Returns:
          E_local: a tensor of dimension (n_samples, 1) containing the local
            energies of the configurations in x
          log_Psi_grad_sum1, log_Psi_grad_sum2 : a tensor of dimension
            (n_samples, 1) containing the sum of the gradient of the logarithm of
            Psi w.r.t. the parameters of DS1, DS2, respectively, for configurations
            in x
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
            tmp = self.log_Psi(x_tmp, m, g)
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

            KE_local = 1 / (2 * m) * torch.sum(-gradient ** 2 - second_ders, axis=1)[:, None]
            del gradient, second_ders

            PE_local = torch.sum(V(x) - mu, axis=1)[:, None]
        else:
            KE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)
            PE_local = torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.)

        E_local = KE_local + PE_local
        del KE_local, PE_local

        # Computes the sum of the gradient of log_Psi wrt neural network parameters
        x.grad = None
        tmp = self.log_Psi(x, m, g)
        # Accumulated gradient is naturally the sum
        log_Psi_grad_sum1 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS1.parameters(),
                                                grad_outputs=torch.ones_like(tmp))
        tmp = self.log_Psi(x, m, g)
        log_Psi_grad_sum2 = torch.autograd.grad(outputs=tmp,
                                                inputs=self.DS2.parameters(),
                                                grad_outputs=torch.ones_like(tmp))

        # Comptues the sum of the local energy times the gradient of the logarithm
        # of Psi w.r.t. neural network parameters
        x.grad = None
        tmp = self.log_Psi(x, m, g)
        tmp = E_local * tmp
        E_log_Psi_grad_sum1 = torch.autograd.grad(outputs=tmp,
                                                  inputs=self.DS1.parameters(),
                                                  grad_outputs=torch.ones_like(tmp))
        x.grad = None
        tmp = self.log_Psi(x, m, g)
        tmp = E_local * tmp
        E_log_Psi_grad_sum2 = torch.autograd.grad(outputs=tmp,
                                                  inputs=self.DS2.parameters(),
                                                  grad_outputs=torch.ones_like(tmp))

        return E_local, log_Psi_grad_sum1, log_Psi_grad_sum2, E_log_Psi_grad_sum1, E_log_Psi_grad_sum2

    def Energy_Grad_Estimate(self, x_sorted, n, chain_idx_sorted, n_chains,
                             GPU_device, m, mu, V, g):
        '''
        Estimates the gradient of the energy w.r.t. neural network parameters. Also
        estimates the energy and its standard deviation.

        Parameters:
          x_sorted: a list of tensors, the nth of which contains the sampled
            particle configurations containing n particles and is of dimension
            (#, n), where # denotes the number of samples drawn of particle number n
          n: a tensor containing the particle numbers of configurations in x_sorted
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
            which the corresponding particle configuration in x_sorted came.
          n_chains: number of chains
          GPU_device: the GPU device under use (i.e. cuda)
          m: particle mass
          mu: chemical potential
          V: external potential function
          g: Lieb-Liniger interaction strength

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
                self.Local_Gradient_Energy(x, m, mu, V, g)
            Local_Energies.append(Local_Energies_tmp)
            for i in range(len(list(self.DS1.parameters()))):
                log_Psi_grad1[i] += log_Psi_grad_sum1[i]
                E_log_Psi_grad1[i] += E_log_Psi_grad_sum1[i]
            for i in range(len(list(self.DS2.parameters()))):
                log_Psi_grad1[i] += log_Psi_grad_sum1[i]
                E_log_Psi_grad1[i] += E_log_Psi_grad_sum1[i]

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
        E_std = torch.std(E_means_chains) / (n_chains ** 0.5)

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

    def ADAM_update(self, x_sorted, n, chain_idx_sorted, n_chains, GPU_device, m, mu, V, g, t, m1, v1,
                    m2, v2, beta_1, beta_2, lr, lr_q):
        '''
        Performs a single ADAM update

        Parameters:
          x_sorted: a list of tensors, the nth of which contains the sampled
            particle configurations containing n particles and is of dimension
            (#, n), where # denotes the number of samples drawn of particle number n
          n: a tensor containing the particle numbers of configurations in x_sorted
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
            which the corresponding particle configuration in x_sorted came.
          chain_idx_sorted: a list of tensors, each of the same dimension as those
            of x_sorted. An entry of such tensor denotes the MCMC chain from
            which the corresponding particle configuration came.
          n_chains: number of separate MCMC chains
          GPU_device: the GPU device under use (i.e. cuda)
          m: particle mass
          mu: chemical potential
          V: external potential function
          g: Lieb-Liniger interaction strength
          t: current ADAM iteration
          m1, m2: first moment estimate for DS1, DS2, respectively
          v1, v2: second moment estimate for DS1, DS2, respectively
          beta_1, beta_2: ADAM parameters
          lr: learning rate for parameters of DS1, DS
          lr_q: learning rate for parameters of q_n

        Returns:
          E: the estimated energy
          E_std: the estimated standard deviation of E
          m1, m2: updated first moment estimate for DS1, DS2, respectively
          v1, v2: updated second moment estimate for DS1, DS2, respectively
        '''

        eps = 1e-8  # ADAM parameter

        # Estimates gradients
        E, E_std, E_grad1, E_grad2 = self.Energy_Grad_Estimate(x_sorted, n, \
                                                               chain_idx_sorted, n_chains, GPU_device, m, mu, V, g)

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
                             m, mu, V, g, beta_1, beta_2, lr, lr_q, n_iters):
        '''
        Minimizes energy with ADAM, and outputs the estimated energy and particle
        number at each iteration.

        Parameters:
          n_samples: number of samples drawn for each MCMC chain
          n_chains: number of separate MCMC chains
          p_pm: probability of increasing or decreasing particle number at each
            configuration proposal
          GPU_device: the GPU device under use (i.e. cuda)
          m: particle mass
          mu: chemical potential
          V: external potential function
          g: Lieb-Liniger interaction strength
          beta_1, beta_2: ADAM parameters
          lr: learning rate for parameters of DS1, DS
          lr_q: learning rate for parameters of q_n
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
        n_means[-1] = int(round(self.DS1.q_n_mean.data.item()))

        # Performs ADAM for n_iters number of iterations
        for t in range(1, n_iters + 1):
            # Generates MCMC data
            n_0 = int(n_means[t - 2].item())
            x_sorted, n, chain_idx_sorted = \
                self.GenerateMCMCSamples_FockSpace(n_samples, n_chains, p_pm, n_0,
                                                   GPU_device, m, g)

            # Performs ADAM update
            E, E_std, m1, v1, m2, v2 = \
                self.ADAM_update(x_sorted, n, chain_idx_sorted, n_chains, GPU_device, m, mu, V, g, t, m1, v1,
                                 m2, v2, beta_1, beta_2, lr, lr_q)

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

    def plot_n(self, n_means, n_stds, n_exact, start, end):
        '''
        Plots particle number vs. iteration from start to end

        Parameters:
          n_means: an array containing the estimated mean particle number at each
            iteration
          n_stds: an array containing the estimated standard deviation of particle
            number at each iteration
          n_exact: the exact ground state energy
          start: the starting iteration from which data is plotted
          end: the last iteration from which data is plotted
        '''

        iterations = np.arange(start, end)

        # Plots particle number
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=200)
        ax1.plot(iterations, n_means.cpu()[start:end], linewidth=1, label='n')
        ax1.fill_between(iterations, (n_means - n_stds).cpu()[start:end],
                         (n_means + n_stds).cpu()[start:end], facecolor='blue',
                         alpha=0.25, interpolate=True)
        ax1.plot(iterations, n_exact + 0 * n_means.cpu()[start:end],
                 linewidth=1, label='Exact n')
        ax1.set_title('Particle Number vs. Iteration', fontsize=10)
        ax1.set_ylabel('n', fontsize=8)
        ax1.set_xlabel('Iteration', fontsize=8)
        ax1.legend(prop={'size': 7})
        ax1.set_xticklabels(ax1.get_xticks().astype(int), fontsize=6)
        ax1.set_yticklabels(np.round(ax1.get_yticks(), 2), fontsize=6)

        # Plots error in particle number and relative standard deviation
        n_errors = (torch.abs((n_means.cpu() - n_exact) / n_exact))[start:end]
        ax2.plot(iterations, n_errors, linewidth=.7, label='Relative Error')
        ax2.plot(iterations, (n_stds / n_means).cpu()[start:end],
                 linewidth=.7, label='Relative St. Dev.')
        ax2.set_yscale('log')
        ax2.set_title('Particle Number Discrepancies vs. Iteration', fontsize=10)
        ax2.set_ylabel('Value', fontsize=8)
        ax2.set_xlabel('Iteration', fontsize=8)
        ax2.legend(prop={'size': 7})
        ax2.set_xticklabels(ax2.get_xticks().astype(int), fontsize=6)
        ax2.set_yticklabels(np.round(ax2.get_yticks(), 2), fontsize=6)

        fig.tight_layout()
        fig.show()

    def Number_KE_density_function(self, N_pts, x_sorted, chain_idx_sorted, n_chains,
                                   GPU_device, m, g, N_int, batch_size):
        '''
        Estimates the number density function and kinetic energy density function
        across the range [0,L], and the standard deviations thereof

        Parameters:
          N_pts: number of points at which the number density is estimated over
            the range [0,L]
          x_sorted: a list of tensors, the nth of which contains the sampled
            particle configurations of n particles and is of dimension (#, n),
            where # denotes the number of samples drawn of particle number n
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
            which the corresponding particle configuration in x_sorted came.
          n_chains: number of MCMC chains
          GPU_device: the GPU device under use (i.e. cuda)
          N_int: number of integration points used to compute the numerical integral
            needed for local number density calculation
          batch_size: the size of batches over which the local number density is
            estimated; performing the numerical integral is memory intensive, and
            hence we batch over entries to save memory

        Returns:
          num_densities: the estimated number densities
          num_densities_std: the standard deviation of the estimated
            number densities
          KE_densities: the estimated kinetic energy densities
          KE_densities_std: the standard deviation of the estimated kinetic
            energy densities
        '''

        # Evaluates number density and kinetic energy density across N_pts points
        xs = np.linspace(0, self.L, N_pts)
        num_densities = np.zeros(N_pts)
        num_densities_std = np.zeros(N_pts)
        KE_densities = np.zeros(N_pts)
        KE_densities_std = np.zeros(N_pts)
        for i in range(N_pts):
            print('Evaluating number density and KE density at point ' + str(i + 1) + \
                  '/' + str(N_pts))
            x_eval = xs[i]
            num_density, num_density_std, KE_density, KE_density_std = \
                self.Number_KE_density(x_eval, x_sorted, chain_idx_sorted, n_chains, \
                                       GPU_device, m, g, N_int, batch_size)
            num_densities[i] = num_density
            num_densities_std[i] = num_density_std
            KE_densities[i] = KE_density
            KE_densities_std[i] = KE_density_std

        return num_densities, num_densities_std, KE_densities, KE_densities_std

    def Number_KE_density(self, x_eval, x_sorted, chain_idx_sorted, n_chains,
                          GPU_device, m, g, N_int, batch_size):
        '''
        Estimates the number density and kinetic energy density at x_eval and their
        standard deviations

        Parameters:
          x_eval: a scalar value of the point at which the number density is to be
            estimated
          x_sorted: a list of tensors, the nth of which contains the sampled
            particle configurations of n particles and is of dimension (#, n),
            where # denotes the number of samples drawn of particle number n
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
            which the corresponding particle configuration in x_sorted came.
          n_chains: number of MCMC chains
          GPU_device: the GPU device under use (i.e. cuda)
          N_int: number of integration points used to compute the numerical integral
            needed for local number density calculation
          batch_size: the size of batches over which the local number density is
            estimated; performing the numerical integral is memory intensive, and
            hence we batch over entries to save memory

        Returns:
          num_density: the estimated number density
          num_density_std: the standard deviation of the estimated number density
          KE_density: the estimated kinetic energy density
          KE_density_std: the standard deviation of the estimated kinetic
            energy density
        '''

        # Calculates local number densities and local KE densities
        Local_num_densities = []
        Local_KE_densities = []
        for x in x_sorted:
            num_density_local, KE_density_local = self.Local_number_KE_density(x_eval, \
                                                                               x, GPU_device, m, g, N_int, batch_size)
            Local_num_densities.append(num_density_local)
            Local_KE_densities.append(KE_density_local)

        # Estimated number density and KE density
        Local_num_densities = torch.cat(Local_num_densities)
        num_density = torch.mean(Local_num_densities)
        Local_KE_densities = torch.cat(Local_KE_densities)
        KE_density = torch.mean(Local_KE_densities)

        # Calculates standard deviations by binning across the MCMC chains
        chain_idx_sorted_flat = torch.cat(chain_idx_sorted)
        num_density_means_chains = \
            torch.tensor([torch.sum(Local_num_densities[:, 0] * (chain_idx_sorted_flat == i)) / \
                          torch.sum(chain_idx_sorted_flat == i) \
                          for i in range(n_chains)], device=GPU_device)
        num_density_std = torch.std(num_density_means_chains) / (n_chains ** 0.5)
        KE_density_means_chains = \
            torch.tensor([torch.sum(Local_KE_densities[:, 0] * (chain_idx_sorted_flat == i)) / \
                          torch.sum(chain_idx_sorted_flat == i) \
                          for i in range(n_chains)], device=GPU_device)
        KE_density_std = torch.std(KE_density_means_chains) / (n_chains ** 0.5)

        return num_density, num_density_std, KE_density, KE_density_std

    def Local_number_KE_density(self, x_eval, x, GPU_device, m, g,
                                N_int, batch_size):
        '''
        Calculates the local number densities and kinetic energy densities of the
        configurations in x

        Parameters:
          x_eval: a scalar value of the point at which the number density is to be
            estimated
          x: a tensor of dimension (n_samples, n_particles) representing
            n_samples samples of n_particles particle positions
          GPU_device: the GPU device under use (i.e. cuda)
          m: particle mass
          g: Lieb-Liniger interaction strength
          N_int: number of integration points used to compute the numerical integral
            \int dx_i |\varphi_n|^2 and evaluate the marginal distribution
          batch_size: the size of batches over which the local number density is
            estimated; performing the numerical integral is memory intensive, and
            hence we batch over entries to save memory

        Returns:
          num_density_local: a tensor of dimension (n_samples, 1) containing the
            local number densities at x_eval of the configurations in x
          KE_density_local: a tensor of dimension (n_samples, 1) containing the
            local kinetic energy densities at x_eval of the configurations in x
        '''

        x_eval_tensor = torch.cuda.FloatTensor(1).fill_(x_eval)
        # The points across which the numerical integral is calculated
        x_int_points = torch.linspace(0, self.L, N_int, device=GPU_device)

        num_density_local = []
        KE_density_local = []
        for i in range(np.ceil(x.shape[0] / batch_size).astype(int)):
            # Computes log_Psi for numerator
            x_tmp = x[i * batch_size:(i + 1) * batch_size, :]
            x_tmp_numerator = torch.cat((x_eval_tensor.repeat(x_tmp.shape[0], 1),
                                         x_tmp[:, 1:]), axis=1)
            log_Psi_numerator_val = self.log_Psi(x_tmp_numerator, m, g)[:, 0].detach()

            x_tmp_numerator2 = x_tmp_numerator.detach().clone()
            x_tmp_numerator2.requires_grad = True
            x_tmp_numerator2.grad = None

            tmp = torch.exp(self.log_Psi(x_tmp_numerator2, m, g))
            tmp2 = torch.autograd.grad(outputs=tmp, inputs=x_tmp_numerator2,
                                       grad_outputs=torch.ones_like(tmp),
                                       create_graph=True)[0]
            gradient = tmp2.detach()
            del tmp, tmp2

            # Computes log_Psi for integral over distribution
            x_int = torch.cat((x_int_points[:, None, None].repeat(1, x_tmp.shape[0], 1),
                               x_tmp[None, :, 1:].repeat(N_int, 1, 1)), axis=2)
            x_int = x_int.reshape(N_int * x_tmp.shape[0], -1)
            log_Psi_integrand_vals = self.log_Psi(x_int, m, g).detach()
            log_Psi_integrand_vals = log_Psi_integrand_vals.reshape(N_int,
                                                                    x_tmp.shape[0], 1)

            # Calculates local number density
            integrand_vals = torch.exp(2 * log_Psi_integrand_vals[:, :, 0])
            integrals = torch.trapz(integrand_vals, dx=self.L / (N_int - 1), dim=0)
            num_density_local.append(x.shape[1] * torch.exp(2 * log_Psi_numerator_val) \
                                     / integrals)
            # KE_density_local.append(-1/(2*m)*x.shape[1]*torch.exp(log_Psi_numerator_val)*\
            # second_ders[:,0]/integrals)
            KE_density_local.append(1 / (2 * m) * x.shape[1] * gradient[:, 0] ** 2 / integrals)
            del x_tmp, x_tmp_numerator, log_Psi_numerator_val, x_int
            # del log_Psi_integrand_vals, integrand_vals, integrals, second_ders
            del log_Psi_integrand_vals, integrand_vals, integrals, gradient

        num_density_local = torch.cat(num_density_local)[:, None]
        KE_density_local = torch.cat(KE_density_local)[:, None]
        return num_density_local, KE_density_local

    def IE_density_function(self, N_pts, x_sorted, chain_idx_sorted, n_chains,
                            GPU_device, m, g, N_int, batch_size):
        '''
        Estimates the Lieb-Liniger interaction energy density function
        across the range [0,L], and the standard deviation thereof

        Parameters:
          N_pts: number of points at which the density is estimated over
            the range [0,L]
          x_sorted: a list of tensors, the nth of which contains the sampled
            particle configurations of n particles and is of dimension (#, n),
            where # denotes the number of samples drawn of particle number n
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
            which the corresponding particle configuration in x_sorted came.
          n_chains: number of MCMC chains
          GPU_device: the GPU device under use (i.e. cuda)
          N_int: number of integration points used to compute the numerical integral
            needed for local density calculation
          batch_size: the size of batches over which the local density is
            estimated; performing the numerical integral is memory intensive, and
            hence we batch over entries to save memory

        Returns:
          IE_densities: the estimated Lieb-Liniger interaction energy densities
          IE_densities_std: the standard deviation of the estimated Lieb-Liniger
            interaction energy densities
        '''

        # Evaluates interaction energy density across N_pts points
        xs = np.linspace(0, self.L, N_pts)
        IE_densities = np.zeros(N_pts)
        IE_densities_std = np.zeros(N_pts)
        for i in range(N_pts):
            print('Evaluating interaction energy density at point ' + str(i + 1) + \
                  '/' + str(N_pts))
            x_eval = xs[i]
            IE_density, IE_density_std = \
                self.IE_density(x_eval, x_sorted, chain_idx_sorted, n_chains, \
                                GPU_device, m, g, N_int, batch_size)
            IE_densities[i] = IE_density
            IE_densities_std[i] = IE_density_std

        return IE_densities, IE_densities_std

    def IE_density(self, x_eval, x_sorted, chain_idx_sorted, n_chains,
                   GPU_device, m, g, N_int, batch_size):
        '''
        Estimates the interaction energy density at x_eval and their standard
        deviations

        Parameters:
          x_eval: a scalar value of the point at which the interaction energy
            density is to be estimated
          x_sorted: a list of tensors, the nth of which contains the sampled
            particle configurations of n particles and is of dimension (#, n),
            where # denotes the number of samples drawn of particle number n
          chain_idx_sorted: a list of tensors denoting the MCMC chain from
            which the corresponding particle configuration in x_sorted came.
          n_chains: number of MCMC chains
          GPU_device: the GPU device under use (i.e. cuda)
          N_int: number of integration points used to compute the numerical integral
            needed for local density calculation
          batch_size: the size of batches over which the local density is
            estimated; performing the numerical integral is memory intensive, and
            hence we batch over entries to save memory

        Returns:
          IE_density: the estimated interaction energy density
          IE_density_std: the standard deviation of the estimated interaction
            energy density
        '''

        # Calculates local interaction energy densities
        Local_IE_densities = []
        for x in x_sorted:
            IE_density_local = self.Local_IE_density(x_eval, x, GPU_device, m, g,
                                                     N_int, batch_size)
            Local_IE_densities.append(IE_density_local)

        # Estimated interaction energy density
        Local_IE_densities = torch.cat(Local_IE_densities)
        IE_density = torch.mean(Local_IE_densities)

        # Calculates standard deviations by binning across the MCMC chains
        chain_idx_sorted_flat = torch.cat(chain_idx_sorted)
        IE_density_means_chains = \
            torch.tensor([torch.sum(Local_IE_densities[:, 0] * (chain_idx_sorted_flat == i)) / \
                          torch.sum(chain_idx_sorted_flat == i) \
                          for i in range(n_chains)], device=GPU_device)
        IE_density_std = torch.std(IE_density_means_chains) / (n_chains ** 0.5)

        return IE_density, IE_density_std

    def Local_IE_density(self, x_eval, x, GPU_device, m, g, N_int, batch_size):
        '''
        Calculates the local interaction energy densities of the configurations in x

        Parameters:
          x_eval: a scalar value of the point at which the density is to be estimated
          x: a tensor of dimension (n_samples, n_particles) representing
            n_samples samples of n_particles particle positions
          GPU_device: the GPU device under use (i.e. cuda)
          m: particle mass
          g: Lieb-Liniger interaction strength
          N_int: number of integration points used to compute the numerical integral
            for the marginal distribution
          batch_size: the size of batches over which the local density is estimated;
            performing the numerical integral is memory intensive, and hence we batch
            over entries to save memory

        Returns:
          IE_density_local: a tensor of dimension (n_samples, 1) containing the
          local interaction energy densities at x_eval of the configurations in x
        '''

        x_eval_tensor = torch.cuda.FloatTensor(1).fill_(x_eval)
        # The points across which the numerical integral is calculated
        x_int_points = torch.linspace(0, self.L, N_int, device=GPU_device)

        IE_density_local = []
        for i in range(np.ceil(x.shape[0] / batch_size).astype(int)):
            # Computes log_Psi for numerator
            x_tmp = x[i * batch_size:(i + 1) * batch_size, :]
            x_tmp_numerator = torch.cat((x_eval_tensor.repeat(x_tmp.shape[0], 1),
                                         x_eval_tensor.repeat(x_tmp.shape[0], 1),
                                         x_tmp[:, 2:]), axis=1)
            log_Psi_numerator_val = self.log_Psi(x_tmp_numerator, m, g)[:, 0].detach()

            # Computes log_Psi for integral over distribution
            x_int = torch.cat((x_int_points[:, None, None, None].repeat(1, N_int,
                                                                        x_tmp.shape[0], 1),
                               x_int_points[None, :, None, None].repeat(N_int, 1,
                                                                        x_tmp.shape[0], 1),
                               x_tmp[None, None, :, 2:].repeat(N_int, N_int, 1, 1)), axis=3)
            x_int = x_int.reshape(N_int ** 2 * x_tmp.shape[0], -1)
            log_Psi_integrand_vals = self.log_Psi(x_int, m, g).detach()
            log_Psi_integrand_vals = log_Psi_integrand_vals.reshape(N_int, N_int,
                                                                    x_tmp.shape[0], 1)

            # Calculates local interaction energy density
            integrand_vals = torch.exp(2 * log_Psi_integrand_vals[:, :, :, 0])
            integrals = torch.trapz(torch.trapz(integrand_vals,
                                                dx=self.L / (N_int - 1), dim=0),
                                    dx=self.L / (N_int - 1), dim=0)
            IE_density_local.append(g * x.shape[1] * (x.shape[1] - 1) * \
                                    torch.exp(2 * log_Psi_numerator_val) / integrals)
            del x_tmp, x_tmp_numerator, log_Psi_numerator_val, x_int
            del log_Psi_integrand_vals, integrand_vals, integrals

        IE_density_local = torch.cat(IE_density_local)[:, None]
        return IE_density_local

    def TG_n(self, L, m, mu):
        '''
        Calculates the particle number of the Tonks-Girardeau ground state

        Parameters:
          L: system size
          m: particle mass
          mu: chemical potential

        Returns:
          n: particle number of the Tonks-Girardeau ground state
        '''

        b = np.pi ** 2 / (2 * m * L ** 2)
        c = np.pi ** 2 / (12 * m * L ** 2) - mu
        n = (-b + (b ** 2 - 4 * b * c) ** 0.5) / (2 * b)
        n = int(np.round(n))
        return n

    def TG_energy(self, L, m, mu):
        '''
        Calculates the energy of the Tonks-Girardeau ground state

        Parameters:
          L: system size
          m: particle mass
          mu: chemical potential

        Returns:
          E: energy of the Tonks-Girardeau ground state
        '''

        n = self.TG_n(L, m, mu)
        b = np.pi ** 2 / (2 * m * L ** 2)
        c = np.pi ** 2 / (12 * m * L ** 2) - mu
        E = b * (n ** 3 / 3 + n ** 2 / 2 + n / 6) - mu * n
        return E

    def TG_number_density(self, x_eval, n, L):
        '''
        Calculates the number density of the Tonks-Girardeau ground state at the
        point x_eval

        Parameters:
          x_eval: a scalar value at which the number density is to be evaluated
          n: particle number
          L: system size

        Returns:
          num_density_TG: an array of the number density
        '''

        num_density_TG = np.sum(np.array([2 / L * (np.sin(j * np.pi * x_eval / L)) ** 2 \
                                          for j in range(1, n + 1)]))
        return num_density_TG

    def TG_KE_density(self, x_eval, n, L, m):
        '''
        Calculates the kinetic energy density of the Tonks-Girardeau ground state
        at the point x_eval

        Parameters:
          x_eval: a scalar value at which the KE density is to be evaluated
          n: particle number
          L: system size
          m: particle mass

        Returns:
          KE_density_TG: an array of the kinetic energy density
        '''

        KE_density_TG = 1 / (2 * m) * np.sum(np.array([2 / L * (j * np.pi / L * np.cos(j * np.pi * x_eval / L)) ** 2 \
                                                       for j in range(1, n + 1)]))
        return KE_density_TG

    def TG_density_function(self, N_pts, n, L, m):
        '''
        Calculates the number density and kinetic energy density of the
        Tonks-Girardeau ground state across the range [0,L]

        Parameters:
          N_pts: number of points at which the number density is estimated over
            the range [0,L]
          n: particle number
          L: system size
          m: particle mass

        Returns:
          num_density_TG: an array of the kinetic energy density
          KE_density_TG: an array of the kinetic energy density
        '''

        xs = np.linspace(0, L, N_pts)
        num_densities_TG = np.array([self.TG_number_density(x, n, L) for x in xs])
        KE_densities_TG = np.array([self.TG_KE_density(x, n, L, m) for x in xs])
        return num_densities_TG, KE_densities_TG

    def plot_number_density(self, num_densities, num_densities_std,
                            num_densities_exact):
        '''
        Plots particle number density vs position

        Parameters:
          num_densities: an array of the approximate particle number densities
          num_densities_std: an array of the standard deviation of the approximate
            particle number densities
          num_densities_exact: an array of the exact particle number densities
        '''

        # Plots particle number density
        xs = np.linspace(0, self.L, num_densities.shape[0])
        fig, ax = plt.subplots(figsize=(3.25, 2.8), dpi=200)
        ax.plot(xs, num_densities, linewidth=1, label='Approximate')
        ax.fill_between(xs, num_densities - num_densities_std,
                        num_densities + num_densities_std, facecolor='blue',
                        alpha=0.25, interpolate=True)
        ax.plot(xs, num_densities_exact, linewidth=1, label='Exact')
        ax.set_title('Particle Number Density vs. x', fontsize=10)
        ax.set_ylabel('Number Density', fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.legend(prop={'size': 7})
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=6)
        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=6)

        fig.tight_layout()
        fig.show()

    def plot_KE_density(self, KE_densities, KE_densities_std,
                        KE_densities_exact):
        '''
        Plots kinetic energy density vs position

        Parameters:
          KE_densities: an array of the approximate kinetic energy densities
          KE_densities_std: an array of the standard deviation of the approximate
            kinetic energy densities
          KE_densities_exact: an array of the exact kinetic energy densities
        '''

        # Plots kinetic energy density
        xs = np.linspace(0, self.L, KE_densities.shape[0])
        fig, ax = plt.subplots(figsize=(3.25, 2.8), dpi=200)
        ax.plot(xs, KE_densities, linewidth=1, label='Approximate')
        ax.fill_between(xs, KE_densities - KE_densities_std,
                        KE_densities + KE_densities_std, facecolor='blue',
                        alpha=0.25, interpolate=True)
        ax.plot(xs, KE_densities_exact, linewidth=1, label='Exact')
        ax.set_title('Kinetic Energy Density vs. x', fontsize=10)
        ax.set_ylabel('KE Density', fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.legend(prop={'size': 7})
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=6)
        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=6)

        fig.tight_layout()
        fig.show()

    def plot_IE_density(self, IE_densities, IE_densities_std):
        '''
        Plots interaction energy density vs position

        Parameters:
          IE_densities: an array of the approximate interaction energy densities
          IE_densities_std: an array of the standard deviation of the approximate
            interaction energy densities
        '''

        # Plots interaction energy density
        xs = np.linspace(0, self.L, IE_densities.shape[0])
        fig, ax = plt.subplots(figsize=(3.25, 2.8), dpi=200)
        ax.plot(xs, IE_densities, linewidth=1, label='Approximate')
        ax.fill_between(xs, IE_densities - IE_densities_std,
                        IE_densities + IE_densities_std, facecolor='blue',
                        alpha=0.25, interpolate=True)
        ax.plot(xs, 0 * xs, linewidth=1, label='Exact')  # Exact IE density is 0
        ax.set_title('Interaction Energy Density vs. x', fontsize=10)
        ax.set_ylabel('Interaction Energy Density', fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.legend(prop={'size': 7})
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=6)
        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=6)

        fig.tight_layout()
        fig.show()

    def plot_E_density(self, E_densities, E_densities_std, E_densities_exact):
        '''
        Plots interaction energy density vs. position

        Parameters:
          E_densities: an array of the approximate total energy densities
          E_densities_std: an array of the standard deviation of the approximate
            total energy densities
          E_densities_exact: an array of the exact kinetic energy densities
        '''

        # Plots energy density
        xs = np.linspace(0, self.L, E_densities.shape[0])
        fig, ax = plt.subplots(figsize=(3.25, 2.8), dpi=200)
        ax.plot(xs, E_densities, linewidth=1, label='Approximate')
        ax.fill_between(xs, E_densities - E_densities_std,
                        E_densities + E_densities_std, facecolor='blue',
                        alpha=0.25, interpolate=True)
        ax.plot(xs, E_densities_exact, linewidth=1, label='Exact')
        ax.set_title('Energy Density vs. x', fontsize=10)
        ax.set_ylabel('Energy Density', fontsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.legend(prop={'size': 7})
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=6)
        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=6)

        fig.tight_layout()
        fig.show()

