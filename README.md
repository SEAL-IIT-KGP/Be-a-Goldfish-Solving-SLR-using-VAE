# Be a Goldfish: Forgetting Bad Conditioning in Sparse Linear Regression via Variational Autoencoders
Code to replicate the results of the our ICML 2025 paper (Submission Number: 15780).

## Authors
Kuheli Pratihar, Debdeep Mukhopadhyay 

Department of Computer Science and Engineering, Indian Institute of Technology Kharagpur, Kharagpur, India

### TL;DR: 
We use Variational Autoencoders that smoothen out bad local minima to solve the NP-hard inverse problem of Sparse linear regression and perform better than conventional methods.

# Abstract:
Variational Autoencoders (VAEs), a class of latent-variable generative models, have seen extensive use in high-fidelity synthesis tasks, yet their loss landscape remains poorly understood. Prior theoretical works on VAE loss analysis have focused on their latent-space representational capabilities both in the optimal and limiting cases. Although these insights have guided better VAE designs, they also often restrict VAEs to problem settings where classical algorithms, such as Principal Component Analysis (PCA), can trivially guarantee globally optimal solutions. In this work, we push the boundaries of our understanding of VAEs beyond these traditional regimes to tackle NP-hard sparse inverse problems, for which no classical algorithms exist. Specifically, we examine the nontrivial Sparse Linear Regression (SLR) problem of recovering optimal sparse inputs in presence of an ill-conditioned design matrix having correlated features. We provably show that, under a linear encoder and a decoder architecture incorporating the product of the SLR design matrix with a trainable, sparsity-promoting diagonal matrix, any minimum of VAE loss is guaranteed to be an optimal solution. This property is especially useful for identifying (a) a preconditioning factor that reduces the eigenvalue spread, and (b) the corresponding optimal sparse representation, albeit in settings where the design matrix can be preconditioned in polynomial time. Lastly, our empirical analysis with different types of design matrices validate these findings, and even demonstrate a higher recovery rate at low-sparsity where traditional algorithms fail. Overall, this work highlights the flexible nature of the VAE loss, which can be adapted to efficiently solve computationally hard problems under specific constraints.

# Libraries reuqired:
numpy, sklearn, matplotlib, pandas, seaborn

# Please run the following python files to generate the plots for Fig. 1 in the manuscript.
1. Figure 1(a) using Gaussian Identitiy Matrix - Test_Gaussian_Identity.py
2. Figure 1(b) using Gaussian Random Walk Matrix - Test_Gaussian_RandomWalk.py
3. Figure 1(c) using Riboflavin Dataset Matrix - Test_Biomedical_Matrix.py
   
The codes for the SLR will be updated with more experimental details once the camera ready version becomes online.
