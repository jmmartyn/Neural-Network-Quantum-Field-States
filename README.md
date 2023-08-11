# Neural-Network Quantum Field States

In recent years, machine learning has inspired the application of neural networks to problems in quantum physics, leading to a new variational ansatz known as a [neural-network quantum state](https://www.science.org/doi/10.1126/science.aag2302) (NQS). Predicated on the variational principle, this ansatz parameterizes a wave function by neural networks, which are optimized over to approximate the ground state of a system. NQSs have seen successful applications to [spin chains](https://www.nature.com/articles/s41567-019-0545-1), [quantum chemistry](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429), and beyond.

In [Variational Neural-Network Ansatz for Continuum Quantum Field Theory](https://arxiv.org/abs/2212.00782), we develop neural-network quantum field states (NQFSs) as a variational ansatz for quantum field theory, thus extending the range of NQSs to quantum field theories directly in the continuum. In our work, we study non-relativstic bosonic field theories. We model the corresponding NQFS by its Fock space representation as a superposition of $n$-particle wave functions, $\varphi_n^\text{NQFS}(\textbf{x}_n)$:

$$ |\Psi^{\text{NQFS}}\rangle = \sum_{n=0}^\infty \int d^nx \ \varphi_n^\text{NQFS}(\textbf{x}_n) |\textbf{x}_n\rangle, $$

where each $n$-particle wave function is a product of two [Deep Sets](https://arxiv.org/abs/1703.06114) neural network architectures -- one that accounts for particle positions $\\{x_i\\} _{i=1}^n$, and the other for particle separations $\\{x_i-x_j\\} _{i < j}$:

$$ \varphi_n^\text{NQFS}(\textbf{x}_n) = \frac{1}{L^{n/2}} \cdot f_1\big( \\{x_i\\} _{i=1}^n \big) \cdot f_2\big( \\{x_i-x_j\\} _{i < j} \big). $$

This architecture is permutation invariant and able to accept an arbitrary number of inputs, which crucially allows us to parameterize the infinitely many $n$-particle wave functions comprising a bosonic quantum field state, with a finite number of neural-networks. The NQFS ansatz is fundementally inspired by [recent](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.023138) [works](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.022502) that apply Deep Sets in the context of neural-network quantum states.

We employ an algorithm for variational Monte Carlo in Fock space to estimate and minimize the energy of a NQFS, and ultimately approximate the ground state. We demonstrate the applicability of NQFS to a variety of field theories by benchmarking on the Lieb-Liniger model, the Calogero-Sutherland model, and a regularized Klein-Gordon model.


# Code Description
This repository contains the code for NQFS. The folder ./modules contains the following classes:
* `deep_sets.py` - Class for the Deep Sets architecture
* `nqfs_ll.py` - Class for NQFS, specially developed for the Lieb-Liniger model
* `nqfs_cs.py` - Class for NQFS, specially developed for the Calogero-Sutherland model
* `nqfs_quad.py` - Class for NQFS, specially developed for a quadratic model that is equivalent to a regularized Klein-Gordon model
* `nqfs.py` - Class for NQFS, applied to a generic QFT Hamiltonian


We also include notebooks to run this code: 
* `NQFS_LiebLiniger.ipynb` - NQFS applied to the Lieb-Liniger model 
* `NQFS_CalogeroSutherland.ipynb` - NQFS applied to the Calogero-Sutherland model
* `NQFS_Quadratic.ipynb` - NQFS applied to a quadratic model, equivalent to a regularized Klein-Gordon model
* `NQFS_Generic.ipynb` - NQFS applied to a generic QFT Hamiltonian

Upon downloading the ./modules folder, these notebooks can be run on Google Colab with access to GPU.
