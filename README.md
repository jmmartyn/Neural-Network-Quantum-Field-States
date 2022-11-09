# Neural Network Quantum Field States

In recent years, machine learning has inspired the application of neural networks to problems in quantum physics, leading to a new variational ansatz known as a neural-network quantum state (NQS)~\cite{Carleo2017}. Predicated on the varitaional principle, this ansatz parameterizes a wave function by neural networks, which are optimized over to approximate a ground state.

In Neural-Network Quantum States for Continuum Quantum Field Theory, we developed neural-network quantum field states (NQFSs), thus extending the range of NQS to quantum field theories directly in the continuum. In this work, we studied non-relativstic bosonic field theories. We model the corresponding NQFS by its Fock space representation as a superposition of $n$-particle wave functions, $\varphi_n^\text{NQFS}(\textbf{x}_n)$:
\begin{equation}
\begin{aligned}
    &|\Psi^{\text{NQFS}}\rangle = \sum_{n=0}^\infty \int d^nx \ \varphi_n^\text{NQFS}(\textbf{x}_n) |\textbf{x}_n\rangle,
\end{aligned}
\end{equation}
where each $n$-particle wave function is a product of two Deep Sets architectures -- one that accounts for particle positions $\{x_i\}_{i=1}^n$, and the other for particle separations $\{x_i - x_j\}_{i<j}$:
\begin{equation}
\begin{aligned}
    &\varphi_n^\text{NQFS}(\textbf{x}_n) = \frac{1}{L^{n/2}} \cdot f_1\big( \{x_i\}_{i=1}^n \big) \cdot f_2\big( \{x_i-x_j\}_{i < j} \big),
\end{aligned}
\end{equation}
This architecture is permutation invariantand able to accept an arbitrary number of inputs, which crucially allowa us to parameterize the infinitely many $n$-particle wave functions comprising a bosninc quantum field state. We employ an algorithm for variational Monte Carlo in Fock space to optimize a NQFS, and benchmark our ansatz by approximating the ground state of the Lieb-Liniger model, the Calogero-Sutherland model, and a regularized Klein-Gordon model.


# Code Description



# Code Usage
