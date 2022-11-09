# Neural Network Quantum Field States

In recent years, machine learning has inspired the application of neural networks to problems in quantum physics, leading to a new variational ansatz known as a [neural-network quantum state](https://www.science.org/doi/10.1126/science.aag2302). Predicated on the varitaional principle, this ansatz parameterizes a wave function by neural networks, which are optimized over to approximate a ground state. NQSs has seen succcessful applications to [spin chains](https://www.nature.com/articles/s41567-019-0545-1), [quantum chemsitry](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429), and beyond.

In [Neural-Network Quantum States for Continuum Quantum Field Theory](link), we develop neural-network quantum field states (NQFSs) as a variational ansatz for quantum fidle theory, thus extending the range of NQS to quantum field theories directly in the continuum. In this work, we study non-relativstic bosonic field theories. We model the corresponding NQFS by its Fock space representation as a superposition of $n$-particle wave functions, $\varphi_n^\text{NQFS}(\textbf{x}_n)$:

$$ |\Psi^{\text{NQFS}}\rangle = \sum_{n=0}^\infty \int d^nx \ \varphi_n^\text{NQFS}(\textbf{x}_n) |\textbf{x}_n\rangle, $$

where each $n$-particle wave function is a product of two [Deep Sets](https://arxiv.org/abs/1703.06114) neural- network architectures -- one that accounts for particle positions ${x_i}_{i=1}^n}, and the other for particle separations:

$$ \varphi_n^\text{NQFS}(\textbf{x}_n) = \frac{1}{L^{n/2}} \cdot f_1\big( \\{x_i\\}_{i=1}^n \big) \cdot f_2\big( \{x_i-x_j\}_{i < j} \big), $$

This architecture is permutation invariant and able to accept an arbitrary number of inputs, which crucially allows us to parameterize the infinitely many $n$-particle wave functions comprising a bosninc quantum field state. It is ultimately inspired by [recent](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.023138) [works](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.022502) that apply Deep Sets in the context of neural-network quantum states.

We employ an algorithm for variational Monte Carlo in Fock space to estimate and minimize the energy of a NQFS, and ultimately approximate the ground state. We demonstrate the applicability of NQFS to a variety of field thoeries by benchmarking on the Lieb-Liniger model, the Calogero-Sutherland model, and a regularized Klein-Gordon model.


# Code Description
This repository contains the classes for NQFS

We also icnlude ntoebooks to run theis code. These can be run on Google Colab with access to GPU
