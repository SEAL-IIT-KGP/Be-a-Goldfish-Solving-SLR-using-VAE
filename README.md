# Be a Goldfish: Forgetting Bad Conditioning in Sparse Linear Regression via Variational Autoencoders

This repository contains the code to reproduce the results of our **ICML 2025** paper (Submission Number: 15780).

## Authors
- Kuheli Pratihar  
- Debdeep Mukhopadhyay  

Department of Computer Science and Engineering, Indian Institute of Technology Kharagpur, India  

---

## TL;DR
We use **Variational Autoencoders (VAEs)** to smooth out bad local minima in the NP-hard problem of Sparse Linear Regression (SLR). Our method outperforms conventional approaches, particularly under ill-conditioned design matrices with correlated features.

---

## Abstract
Variational Autoencoders (VAEs), a class of latent-variable generative models, have seen extensive use in high-fidelity synthesis tasks, yet their loss landscape remains poorly understood. Prior theoretical works on VAE loss analysis have focused on their latent-space representational capabilities, both in the optimal and limiting cases. Although these insights have guided better VAE designs, they also often restrict VAEs to problem settings where classical algorithms, such as Principal Component Analysis (PCA), can trivially guarantee globally optimal solutions.  

In this work, we extend the understanding of VAEs to **NP-hard sparse inverse problems**, specifically the **Sparse Linear Regression (SLR)** problem of recovering optimal sparse inputs under ill-conditioned design matrices. We prove that under a linear encoder and a decoder incorporating the product of the SLR design matrix with a trainable sparsity-promoting diagonal matrix, **any minimum of the VAE loss corresponds to an optimal solution**.  

This property enables the identification of:  
1. A preconditioning factor that reduces eigenvalue spread.  
2. The corresponding optimal sparse representation.  

Our empirical results validate these findings across various design matrices, showing a higher recovery rate even in low-sparsity regimes where traditional algorithms fail. Overall, this highlights the adaptability of VAEs to efficiently solve computationally hard problems under structured constraints.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/SEAL-IIT-KGP/Be-a-Goldfish-Solving-SLR-using-VAE.git
```

### Using Conda (Recommended)
```bash
conda create -n goldfish-env python=3.10 -y
conda activate goldfish-env
pip install numpy scikit-learn matplotlib pandas seaborn
```
---

## Reproducing Results

The scripts below generate the plots for **Figure 1** in the paper:

1. **Gaussian Identity Matrix**  
   ```bash
   python Code/Test_Gaussian_Identity.py
   ```

2. **Gaussian Random Walk Matrix**  
   ```bash
   python Code/Test_Gaussian_RandomWalk.py
   ```

3. **Riboflavin Dataset Matrix (Biomedical)**  
   ```bash
   python Code/Test_Biomedical_Matrix.py
   ```

The output plots will replicate the corresponding subfigures in the paper.  

---

## Repository Structure
```
.
├── Code/Test_Gaussian_Identity.py      # Code for Figure 1(a)
├── Code/Test_Gaussian_RandomWalk.py    # Code for Figure 1(b)
├── Code/Test_Biomedical_Matrix.py      # Code for Figure 1(c)
└── README.md                      # Project documentation
```

---

## Citation
If you use this code in your research, please cite our ICML 2025 paper:  

```bibtex
@inproceedings{pratihar2025goldfish,
  title     = {Be a Goldfish: Forgetting Bad Conditioning in Sparse Linear Regression via Variational Autoencoders},
  author    = {Pratihar, Kuheli and Mukhopadhyay, Debdeep},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
