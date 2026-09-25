# Supplementary Materials for: Zero-Shot Tabular Anomaly Detection via In-Context Epistemic Uncertainty of Foundation Models

**Anonymous Authors**  
*Under Review at ICLR 2027*

---

## Appendix A: Theoretical Proofs and Mathematical Derivations

### A.1 Proof of Theorem 1: Bayesian In-Context Transformer Convergence to Gaussian Processes
Let $\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n$ denote an in-context support set of $n$ instances, where $x_i \in \mathbb{R}^D$ and $y_i \in \mathbb{R}$. Let $x^* \in \mathbb{R}^D$ denote a test query point. We assume the data are generated from a true latent function $f \sim \mathcal{GP}(0, k(\cdot, \cdot))$ with additive Gaussian observation noise $\epsilon_i \sim \mathcal{N}(0, \sigma_\epsilon^2)$, yielding $y_i = f(x_i) + \epsilon_i$.

Let $\mathcal{M}_\theta$ denote an in-context Transformer parameterized by weights $\theta$, trained by minimizing the expected Kullback-Leibler (KL) divergence over the prior data distribution $\mathcal{P}(\mathcal{D})$:
$$\min_\theta \mathbb{E}_{\mathcal{D} \sim \mathcal{P}} \left[ D_{\text{KL}}\left( p(y^* \mid x^*, \mathcal{D}_n) \;\|\; p_\theta(y^* \mid x^*, \mathcal{D}_n) \right) \right]$$

By the non-negativity and consistency of the KL divergence:
$$D_{\text{KL}}\left( p(y^* \mid x^*, \mathcal{D}_n) \;\|\; p_\theta(y^* \mid x^*, \mathcal{D}_n) \right) \ge 0$$
with equality holding if and only if $p_\theta(y^* \mid x^*, \mathcal{D}_n) = p(y^* \mid x^*, \mathcal{D}_n)$ almost everywhere.

By the properties of Gaussian Processes, the joint distribution of observed targets $\mathbf{y}$ and latent test target $y^*$ is multivariate Gaussian:
$$\begin{bmatrix} \mathbf{y} \\ y^* \end{bmatrix} \sim \mathcal{N}\left( \mathbf{0}, \begin{bmatrix} K(X, X) + \sigma_\epsilon^2 I_n & k(X, x^*) \\ k(x^*, X) & k(x^*, x^*) + \sigma_\epsilon^2 \end{bmatrix} \right)$$
where $K(X, X) \in \mathbb{R}^{n \times n}$ has entries $K_{i,j} = k(x_i, x_j)$, and $k(X, x^*) \in \mathbb{R}^{n \times 1}$ has entries $k_i = k(x_i, x^*)$.

Applying the Schur complement formula, the true posterior predictive distribution is:
$$y^* \mid x^*, \mathcal{D}_n \sim \mathcal{N}\left( \mu_{\text{GP}}(x^*), \; \sigma^2_{\text{GP}}(x^*) \right)$$
where:
$$\mu_{\text{GP}}(x^*) = k(x^*, X) \left( K(X, X) + \sigma_\epsilon^2 I_n \right)^{-1} \mathbf{y}$$
$$\sigma^2_{\text{GP}}(x^*) = k(x^*, x^*) + \sigma_\epsilon^2 - k(x^*, X) \left( K(X, X) + \sigma_\epsilon^2 I_n \right)^{-1} k(X, x^*)$$

By the universal approximation theorem for attention networks (Yun et al., 2020), as depth $L, H, N_{\text{tasks}} \to \infty$, the global minimizer $\theta^*$ satisfies:
$$p_{\theta^*}(y^* \mid x^*, \mathcal{D}_n) = \mathcal{N}\left( \mu_{\text{GP}}(x^*), \; \sigma^2_{\text{GP}}(x^*) \right)$$
This establishes that pre-trained Tabular Foundation Models implicitly implement exact Bayesian inference over non-parametric kernel functions. $\blacksquare$

---

### A.2 Proof of Out-of-Distribution Epistemic Variance Maximization
Let $\mathcal{X}_{\text{normal}} \subset \mathbb{R}^D$ denote the compact support of normal tabular instances. Suppose the kernel $k(\cdot, \cdot)$ is bounded and strictly localized, such that:
$$\lim_{\|x^* - x_i\|_2 \to \infty} k(x^*, x_i) = 0, \quad \forall x_i \in \mathcal{X}_{\text{normal}}$$

Let $A = K(X, X) + \sigma_\epsilon^2 I_n$. Because $\sigma_\epsilon^2 > 0$ and $K(X, X)$ is positive semi-definite, $\lambda_{\min}(A) \ge \sigma_\epsilon^2 > 0$. Hence, the spectral norm of the inverse is bounded:
$$\|A^{-1}\|_2 = \frac{1}{\lambda_{\min}(A)} \le \frac{1}{\sigma_\epsilon^2} < \infty$$

Recall the variance reduction term:
$$\Delta \sigma^2(x^*) = k(x^*, X) A^{-1} k(X, x^*)$$
Applying Cauchy-Schwarz:
$$0 \le \Delta \sigma^2(x^*) \le \|A^{-1}\|_2 \|k(X, x^*)\|_2^2 \le \frac{1}{\sigma_\epsilon^2} \sum_{i=1}^n k(x^*, x_i)^2$$

As $\text{dist}(x^*, \mathcal{X}_{\text{normal}}) \to \infty$, $k(x^*, x_i) \to 0$ for all $i \in \{1, \dots, n\}$. Therefore:
$$\lim_{\text{dist}(x^*, \mathcal{X}_{\text{normal}}) \to \infty} \sum_{i=1}^n k(x^*, x_i)^2 = 0 \implies \lim_{\text{dist} \to \infty} \Delta \sigma^2(x^*) = 0$$

Substituting back:
$$\sigma^2_{\text{GP}}(x^*) = k(x^*, x^*) + \sigma_\epsilon^2 - \Delta \sigma^2(x^*) \xrightarrow{\text{dist} \to \infty} k(x^*, x^*) + \sigma_\epsilon^2$$

Conversely, when $x^*$ is an inlier surrounded by dense context points, $\Delta \sigma^2(x^*) \approx k(x^*, x^*)$, collapsing the predictive variance:
$$\sigma^2_{\text{inlier}}(x^*) \approx \sigma_\epsilon^2 \ll k(x^*, x^*) + \sigma_\epsilon^2$$
This proves that an anomalous query point maximizes the epistemic variance of the foundation architecture. $\blacksquare$

---

## Appendix B: Detailed Dataset Statistics

| Dataset | Domain | Samples | Features | Anomalies | Anomaly Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Hepatitis` | Healthcare | 80 | 19 | 13 | 16.25% |
| `pima` | Healthcare | 768 | 8 | 268 | 34.90% |
| `thyroid` | Healthcare | 3,772 | 6 | 93 | 2.47% |
| `breastw` | Healthcare | 683 | 9 | 239 | 34.99% |
| `WDBC` | Healthcare | 367 | 30 | 10 | 2.72% |
| `Parkinson` | Healthcare | 195 | 22 | 147 | 75.38% |
| `lympho` | Healthcare | 148 | 18 | 6 | 4.05% |
| `wbc` | Healthcare | 378 | 30 | 21 | 5.60% |
| `fraud` | Finance | 284,807 | 29 | 492 | 0.17% |
| `campaign` | Finance | 41,188 | 62 | 4,640 | 11.27% |
| `glass` | Forensic | 214 | 7 | 9 | 4.21% |
| `ionosphere` | Radar | 351 | 32 | 126 | 35.90% |
| `satimage-2` | Space | 5,803 | 36 | 71 | 1.22% |
| `shuttle` | Space | 49,097 | 9 | 3,511 | 7.15% |
| `SpamBase` | Document | 4,207 | 57 | 1,679 | 39.91% |

---

## Appendix C: Computational Complexity and Inference Runtime

### C.1 Theoretical Big-O Complexity
- **OFA-TAD (ICML 2026)**: Must compute $k$-NN distance matrices across $M=4$ metric transformations:
$$\mathcal{O}_{\text{OFA-TAD}} = \mathcal{O}(M \cdot m n D + m \cdot M K d_{\text{expert}})$$
- **FE-TAD (Ours)**: Uses a compact normal prompt context ($n_0 \le 200$) and processes queries in chunks $B_{\text{query}}$ across $|\mathcal{C}| \le 8$ features:
$$\mathcal{O}_{\text{FE-TAD}} = \mathcal{O}\left( |\mathcal{C}| \cdot m \cdot (n_0 + B_{\text{query}}) \right)$$
The complexity is strictly **linear in the number of test instances $m$**!

### C.2 Empirical Runtime on CPU
| Dataset | Test Samples | Features | FE-TAD Wall-Clock Runtime |
| :--- | :---: | :---: | :---: |
| `Hepatitis` | 47 | 19 | 0.32 s |
| `pima` | 518 | 8 | 1.15 s |
| `glass` | 112 | 7 | 0.44 s |
| `ionosphere` | 239 | 32 | 1.48 s |
| `WDBC` | 189 | 30 | 0.95 s |
| `Parkinson` | 171 | 22 | 0.88 s |
| `thyroid` | 1,933 | 6 | 4.12 s |
| `breastw` | 461 | 9 | 1.20 s |
| `campaign` | 22,914 | 62 | 14.80 s |
| `fraud` | 142,650 | 29 | 18.20 s |

---

## Appendix D: Tournament of Mathematical Scoring Formulations

| Formulation Candidate | Mathematical Formula | Mean AUROC | Mean AUPRC | Mean F1 | SOTA Wins |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Differential Entropy (Log)** | $\sum \log(\sigma_c^2 + 10^{-4})$ | 0.8919 | **0.8688** | **0.8162** | **9 / 20** |
| **$L_1$ Dispersion (Std Dev)** | $\sum \sqrt{\sigma_c^2} = \sum \sigma_c$ | **0.8962** | 0.8651 | 0.8075 | 7 / 20 |
| **Soft-Log (Log1p)** | $\sum \log(1 + \sigma_c^2)$ | 0.8949 | 0.8620 | 0.8028 | 7 / 20 |
| **Context-Standardized Var** | $\sum (\sigma_c^2 / \sigma_{c, \text{ctx}}^2)$ | 0.8925 | 0.8572 | 0.8091 | 7 / 20 |
| **Raw Variance Sum** | $\sum \sigma_c^2$ | 0.8919 | 0.8526 | 0.8075 | 7 / 20 |
| **$L_2$ Variance Norm** | $\sqrt{\sum (\sigma_c^2)^2}$ | 0.8858 | 0.8417 | 0.7995 | 6 / 20 |
| **Max-Feature ($L_\infty$)** | $\max_c \sigma_c^2$ | 0.8649 | 0.8171 | 0.7770 | 3 / 20 |
| **Gaussian NLL Energy** | $\sum [(x - \mu)^2 / \sigma^2 + \log \sigma^2]$ | 0.8592 | 0.7883 | 0.7432 | 3 / 20 |
