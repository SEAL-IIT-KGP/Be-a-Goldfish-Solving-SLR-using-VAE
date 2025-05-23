#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from VAEforSLR import recover_from_vae
import pandas
import seaborn

#%%
def generate_dictionary(d, n):
    """
    Generate a d x n matrix Phi with i.i.d. Gaussian entries.
    Then normalize each column to have unit L2 norm.
    
    :param d: Number of rows (measurement dimension).
    :param n: Number of columns (dictionary size).
    :return: (d x n) numpy array representing the dictionary Phi.
    """
    Num = 10000
    Phi_full = np.random.randn(Num, n)
    
    Sigma = (1/(Num-1))*Phi_full.T@Phi_full - (np.mean(Phi_full.T,axis=1))**2
    
    Phi = Phi_full[:d,:]
    # Normalize columns to have unit L2 norm
    for j in range(n):
        col_norm = norm(Phi[:, j])
        if col_norm > 1e-12:
            Phi[:, j] /= col_norm

    return Phi, Sigma

def generate_sparse_vector(n, r):
    """
    Generate an n-dimensional sparse vector u_0 with exactly r nonzero entries.
    The nonzero entries are chosen at random, each from a standard normal distribution.
    Then the vector is optionally normalized (optional step).
    
    :param n: Dimension of the coefficient vector.
    :param r: Number of nonzero entries.
    :return: An n-dimensional numpy array representing the ground-truth sparse vector u_0.
    """
    u0 = np.zeros(n)
    
    # Randomly pick which indices are "active" (nonzero)
    active_indices = np.random.choice(n, size=r, replace=False)
    
    # Fill those entries with random Gaussian
    u0[active_indices] = np.random.randn(r)
    
    return u0


def recover_with_lasso(Phi, x, alpha=0.01):
    """
    Solve a sparse linear regression problem using Lasso from scikit-learn:
      minimize  0.5 * ||x - Phi @ u||^2_2 + alpha * ||u||_1
    
    :param Phi: (d x n) dictionary matrix
    :param x: (d,) data/measurement vector
    :param alpha: Regularization parameter for Lasso
    :return: (n,) recovered coefficient vector u_hat
    """
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000)
    # Lasso expects X.shape = (n_samples, n_features), y.shape = (n_samples,)
    # Here, "n_samples" = d, "n_features" = n
    # So we pass Phi as shape (d, n) and x as shape (d,).
    model.fit(Phi, x)
    u_hat = model.coef_
    return u_hat

def compute_metrics(u_true, u_hat, Sigma, threshold = 1e-3):
    # Compute the L2 error (Euclidean norm of the difference)
    true_support = set(np.where(np.abs(u_true) > threshold)[0])
    est_support  = set(np.where(np.abs(u_hat)  > threshold)[0])
    support_match = (true_support == est_support)
    mse_error = np.linalg.norm(u_true - u_hat)/len(u_hat)
    excess_risk = min(u_true@Sigma@u_true, (u_true-u_hat) @ Sigma @ (u_true-u_hat))
    return support_match, excess_risk, mse_error


d=100        # measurement dimension
n=200         # dictionary dimension
r_values = [2, 10, 20, 30, 40, 50, 60] # array of r (number of nonzero entries)
#r_values = [2] # array of r (number of nonzero entries)
num_trials = 10 # how many random trials per r
alpha=0.00001
dyn_lr = 1e-2

# Generate a single dictionary Phi once (or regenerate each time if desired)
Phi, Sigma = generate_dictionary(d, n)

lasso_loss = []
vae_loss = []

lasso_support = []
vae_support = []

lasso_risk = []
vae_risk = []

#%%
for iter_r in range(len(r_values)):
    
    r = r_values[iter_r]
    lasso_loss_sub = []
    vae_loss_sub = []
    
    lasso_support_sub = []
    vae_support_sub = []
    
    lasso_risk_sub = []
    vae_risk_sub = []
    
    for _ in range(num_trials):
        # Generate ground-truth sparse vector and data
        u0 = generate_sparse_vector(n, r)  # (n,)
        x  = Phi @ u0                      # (d,)
        
        # Recover using Lasso
        u_hat_lasso = recover_with_lasso(Phi, x, alpha=alpha)
        
        support_lasso, risk_lasso, error_lasso = compute_metrics(u0, u_hat_lasso, Sigma)
        lasso_loss_sub.append(error_lasso)
        lasso_support_sub.append(support_lasso)
        lasso_risk_sub.append(risk_lasso)
        
    
        _, u_hat_vae = recover_from_vae(Phi, x, n, init_gamma=1e-1, num_steps=50000, lr=dyn_lr)
        #u_hat_vae = np.zeros(n)
        # Check if we recovered the same support
        support_vae, risk_vae, error_vae = compute_metrics(u0, u_hat_vae, Sigma)
        vae_loss_sub.append(error_vae)
        vae_support_sub.append(support_vae)
        vae_risk_sub.append(risk_vae)
        
    plt.figure(figsize=(6, 4))
    plt.plot(u0, marker='o', color='red')
    plt.plot(u_hat_lasso, marker='o', color='blue')
    plt.plot(u_hat_vae, marker='o', color='green')
    plt.grid(True)
    plt.show()
    
    lasso_loss.append(lasso_loss_sub)
    vae_loss.append(vae_loss_sub)
    
    lasso_support.append(lasso_support_sub)
    vae_support.append(vae_support_sub)
    
    lasso_risk.append(lasso_risk_sub)
    vae_risk.append(vae_risk_sub)
    
    print('Loss',lasso_loss[-1],vae_loss[-1])
    print('Support',lasso_support[-1], vae_support[-1])
    print('Risk',lasso_risk[-1], vae_risk[-1])

#%%    
np.savez('Lasso-Gaussian-Identity-10-trials.npz',np.array(lasso_loss),np.array(lasso_support),np.array(lasso_risk))
np.savez('VAE-Gaussian-Identity-10-trials.npz',np.array(vae_loss),np.array(vae_support),np.array(vae_risk))

#%%
df_loss = pandas.DataFrame(columns=["r","lasso","vae"])
for i in range(len(vae_loss)):
    for j in range(len(vae_loss[0])):
        df_loss = df_loss._append({"r": r_values[i], "lasso": lasso_loss[i][j], "vae": vae_loss[i][j]}, ignore_index=True)

df_support = pandas.DataFrame(columns=["r","lasso","vae"])
for i in range(len(vae_support)):
    for j in range(len(vae_support[0])):
        df_support = df_support._append({"r": r_values[i], "lasso": lasso_support[i][j], "vae": vae_support[i][j]}, ignore_index=True)

df_risk = pandas.DataFrame(columns=["r","lasso","vae"])
for i in range(len(vae_risk)):
    for j in range(len(vae_risk[0])):
        df_risk = df_risk._append({"r": r_values[i], "lasso": lasso_risk[i][j], "vae": vae_risk[i][j]}, ignore_index=True)


# Plot the success rates
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

fig, ax = plt.subplots(figsize=(6,4))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
plt.ylabel('Support Recovery Rate',fontsize=18,fontdict=font)
plt.xlabel(r'Number of nonzero entries ($\kappa$)',fontsize=18,fontdict=font)

ax0 = seaborn.lineplot(data=df_support, x="r", y="lasso", errorbar="sd", label="Lasso", color="red")
ax1 = seaborn.lineplot(data=df_support, x="r", y="vae", errorbar="sd", label="VAE", color="green")

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(loc='upper center', ncol=3,bbox_to_anchor=(0.51, 1.2),prop={'size': 16, 'family':'Arial', 'weight':'regular'}                 
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

plt.grid(True)
plt.savefig('Support_Recovery_Gaussian_Identity.png', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(6,4))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
plt.ylabel('Mean Square Loss',fontsize=18,fontdict=font)
plt.xlabel(r'Number of nonzero entries ($\kappa$)',fontsize=18,fontdict=font)

ax0 = seaborn.lineplot(data=df_loss, x="r", y="lasso", errorbar="sd", label="Lasso", color="red")
ax1 = seaborn.lineplot(data=df_loss, x="r", y="vae", errorbar="sd", label="VAE", color="green")

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(loc='upper center', ncol=3,bbox_to_anchor=(0.51, 1.2),prop={'size': 16, 'family':'Arial', 'weight':'regular'}                 
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

plt.grid(True)
plt.savefig('MSE_Loss_Gaussian_Identity.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6,4))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
plt.ylabel('Excess Risk',fontsize=18,fontdict=font)
plt.xlabel(r'Number of nonzero entries ($\kappa$)',fontsize=18,fontdict=font)

ax0 = seaborn.lineplot(data=df_risk, x="r", y="lasso", errorbar="sd", label="Lasso", color="red")
ax1 = seaborn.lineplot(data=df_risk, x="r", y="vae", errorbar="sd", label="VAE", color="green")

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(loc='upper center', ncol=3,bbox_to_anchor=(0.51, 1.2),prop={'size': 16, 'family':'Arial', 'weight':'regular'}                 
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

plt.grid(True)
plt.savefig('Excess_Risk_Gaussian_Identity.png', bbox_inches='tight')
plt.show()
#%%

df_loss_flat = pandas.DataFrame(columns=["r","lasso","vae"])
for i in range(len(vae_loss)):
    df_loss_flat = df_loss_flat._append({"r": r_values[i], "lasso":  np.mean(np.array(lasso_loss[i][:])), "vae":  np.mean(np.array(vae_loss[i][:]))}, ignore_index=True)

df_support_flat = pandas.DataFrame(columns=["r","lasso","vae"])
for i in range(len(vae_support)):
    df_support_flat = df_support_flat._append({"r": r_values[i], "lasso": np.mean(np.array(lasso_support[i][:])),  "vae": np.mean(np.array(vae_support[i][:]))}, ignore_index=True)

df_risk_flat = pandas.DataFrame(columns=["r","lasso","vae"])
for i in range(len(vae_risk)):
    df_risk_flat = df_risk_flat._append({"r": r_values[i], "lasso": np.mean(np.array(lasso_risk[i][:])),  "vae": np.mean(np.array(vae_risk[i][:]))}, ignore_index=True)



# Plot the success rates
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

fig, ax = plt.subplots(figsize=(6,4))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
plt.ylabel('Support Recovery Rate',fontsize=18,fontdict=font)
plt.xlabel(r'Number of nonzero entries ($\kappa$)',fontsize=18,fontdict=font)

ax0 = seaborn.lineplot(data=df_support_flat, x="r", y="lasso", errorbar="sd", label="Lasso", color="red")
ax1 = seaborn.lineplot(data=df_support_flat, x="r", y="vae", errorbar="sd", label="VAE", color="green")

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(loc='upper center', ncol=3,bbox_to_anchor=(0.51, 1.2),prop={'size': 16, 'family':'Arial', 'weight':'regular'}                 
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

plt.grid(True)
plt.savefig('Support_Recovery_Gaussian_Identity_flat.png', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(6,4))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
plt.ylabel('Mean Square Loss',fontsize=18,fontdict=font)
plt.xlabel(r'Number of nonzero entries ($\kappa$)',fontsize=18,fontdict=font)

ax0 = seaborn.lineplot(data=df_loss_flat, x="r", y="lasso", errorbar="sd", label="Lasso", color="red")
ax1 = seaborn.lineplot(data=df_loss_flat, x="r", y="vae", errorbar="sd", label="VAE", color="green")

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.legend(loc='upper center', ncol=3,bbox_to_anchor=(0.51, 1.2),prop={'size': 16, 'family':'Arial', 'weight':'regular'}                 
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)

plt.grid(True)
plt.savefig('MSE_Loss_Gaussian_Identity_flat.png', bbox_inches='tight')
plt.show()
