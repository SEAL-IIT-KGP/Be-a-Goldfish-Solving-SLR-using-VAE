import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

large_loss_threshold = 1e6

def reinitialize_model(model, init_gamma):
    """
    Re-randomize the model's parameters in-place, similar to __init__.
    """
    with torch.no_grad():
        # Use the same shapes and scaling as in the constructor.
        model.w_x.data = torch.randn(model.n, model.n) * 0.1
        model.gamma.data = torch.tensor(init_gamma, dtype=torch.float32)
        model.W_z.data = torch.randn(model.n, model.d) * 0.01
        model.S.data = torch.randn(model.n, model.n) * 0.01
        
class VAESLR(nn.Module):
    """
    A Variational Autoencoder for Sparse Linear Regression (SLR) with:
       - Observed data x in R^d
       - Latent variable z in R^n
       - Dictionary Phi in R^(d x n)
       - Decoder:  x ~ N( Phi * diag(w_x) * z,  gamma * I_d )
       - Encoder:  z ~ N( W_z * x,  S S^T )
       - Prior on z: p(z) = N(0, I_n)
    """

    def __init__(self, d, n, Phi, init_gamma=1e-10):
        """
        :param d: dimension of observed data x (i.e. x in R^d).
        :param n: dimension of latent variable z (i.e. z in R^n).
        :param Phi: (d x n) dictionary matrix.
        :param init_gamma: initial guess for gamma (noise variance).
        """
        super().__init__()

        self.d = d  # dimension of x
        self.n = n  # dimension of z

        # Store Phi as a non-trainable buffer.
        self.register_buffer("Phi", torch.tensor(Phi, dtype=torch.float32))

        # -------------------- DECODER PARAMETERS --------------------
        # w_x in R^n => diag(w_x) is (n x n). 
        self.w_x = nn.Parameter(torch.randn(n) * 0.1)

        # gamma is the scalar variance for N(..., gamma*I_d).
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

        # -------------------- ENCODER PARAMETERS --------------------
        # W_z in R^(n x d). Then mu_z(x) = W_z * x => shape n.
        self.W_z = nn.Parameter(torch.randn(n, d) * 0.01)

        # S is an (n x n) factor for the covariance S S^T.
        # We'll store it as a full matrix (you can use a lower-triangular factor for stability).
        self.S = nn.Parameter(torch.randn(n, n) * 0.01)

    def encode(self, x):
        """
        Encoder: q(z | x) = N( z; mu_z(Px),  S S^T )
        :param x: shape (batch_size, d)
        :return: mu_z, L  => shapes (batch_size, n) and (n, n)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # => (1, d)
        d, n = self.Phi.shape
        
        # mu_z = x @ W_z^T => (batch_size, n)
        mu_z = x @ torch.linalg.inv(self.Phi @ self.Phi.T + self.gamma*torch.eye(d)).T @ self.W_z.T  
        L = self.S  # (n, n) shared across batch
        return mu_z, L

    def reparameterize(self, mu, L):
        """
        z = mu + eps @ L^T,  eps ~ N(0, I_n).
        :param mu: (batch_size, n)
        :param L:  (n, n)
        :return: z, (batch_size, n)
        """
        batch_size, n_dim = mu.shape
        eps = torch.randn(batch_size, n_dim, device=mu.device)
        z = mu + eps @ L.T
        return z

    def decode(self, z):
        """
        Decoder: x ~ N(Phi diag(w_x) z, gamma I_d).
        :param z: shape (batch_size, n)
        :return: mu_x in shape (batch_size, d)
        """
        d, n = self.Phi.shape
        # diag(w_x)*z is elementwise => shape (batch_size, n)
        z_scaled = z * self.w_x

        # Multiply by Phi => shape (batch_size, d)
        # Phi is (d,n), z_scaled is (batch_size, n) => we do a matrix multiply
        mu_x = z_scaled @ self.Phi.T @ torch.linalg.inv(self.Phi @ self.Phi.T + self.gamma*torch.eye(d)).T # => (batch_size, d)
        return mu_x

    def forward(self, x):
        """
        Full forward pass (single-sample).
        :param x: shape (batch_size, d)
        :return: z, mu_z, L, mu_x
        """
        mu_z, L = self.encode(x)           # (batch_size, n), (n, n)
        z       = self.reparameterize(mu_z, L)  # (batch_size, n)
        mu_x    = self.decode(z)               # (batch_size, d)
        return z, mu_z, L, mu_x

    def elbo(self, x):
        """
        Returns the mean *negative* ELBO over the batch:
          = E_{q(z|x)}[ -log p(x|z) ] + KL(q(z|x)||p(z)).
        """
        # Forward pass to get z ~ q(z|x)
        z, mu_z, L, mu_x = self.forward(x)
        d, n = self.Phi.shape
        
        # -------------------- 1) Negative log-likelihood ---------------------
        # p(x|z) = N( P*mu_x, gamma I )
        # -log p(x|z) = 0.5*(1/gamma)*||P*x - mu_x||^2 + 0.5*d*log(gamma) + const
        diff = x @ torch.linalg.inv(self.Phi @ self.Phi.T + self.gamma*torch.eye(d)).T - mu_x
        mse = torch.sum(diff * diff, dim=1)  # (batch_size,)
        nll = 0.5 * (mse / self.gamma) + 0.5 * self.d * torch.log(self.gamma)

        # -------------------- 2) KL Divergence -------------------------------
        # q(z|x) = N(mu_z, L L^T) vs p(z)=N(0,I)
        # KL = 0.5 [ trace(L L^T) + mu_z^T mu_z - n - log det(L L^T) ]
        #     = 0.5 [ sum(L^2) + sum(mu_z^2) - n - 2 * log|det(L)| ]
        batch_size, n_dim = mu_z.shape
        sigma_trace = torch.sum(L * L)             # scalar
        mu_sq = torch.sum(mu_z * mu_z, dim=1)      # (batch_size,)
        sign_L, logdet_L = torch.linalg.slogdet(L) # typically scalar if L is shared
        log_det_sigma = 2.0 * logdet_L             # scalar
        kl = 0.5 * (sigma_trace + mu_sq - n_dim - log_det_sigma)  
        # kl is per sample (same trace and det part, different mu_z^2 part)

        # -------------------- 3) Negative ELBO ------------------------------
        negative_elbo_per_sample = nll + kl
        return negative_elbo_per_sample.mean()


def recover_from_vae_with_precon(
    Phi, x, latent_dim, init_gamma=1e-10, 
    num_steps=1000, lr=5e-3, 
    reduce_factor=0.01,  # factor to reduce lr
    no_change_patience=10,  # number of steps to check for "no change"
    no_change_tol=1e-6      # how close must losses be to count as "no change"
):
    """
    Train the VAE-SLR model on a single measurement x in R^d, 
    then recover the coefficient vector.
    """
    d = Phi.shape[0]  # dimension of x
    n = latent_dim    # dimension of z

    # Convert x => Torch
    x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape (1, d)

    # Initialize model
    model = VAESLR(d, n, Phi, init_gamma=init_gamma)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Keep track of recent losses for "no change" detection
    recent_losses = []
    # Flags for schedule
    has_reduced_once = False
    lr_frozen = False

    for step in range(num_steps):

        # Standard VAE training step
        optimizer.zero_grad()
        loss = model.elbo(x_torch)   # single function for negative ELBO
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        recent_losses.append(current_loss)
        
        # Check if the loss is exploding
        if current_loss > large_loss_threshold:
            print(f"Step {step+1}: Loss = {current_loss:.2e} > {large_loss_threshold}, re-initializing model.")
            # Re-sample parameters
            reinitialize_model(model, init_gamma=1e-10)
            # Reset the optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)
            has_reduced_once = False
            lr_frozen = False
            # Skip this update
            continue
        
        # -----------------------------------------------------
        # 1) If loss < 0 and we haven't reduced LR yet, reduce it
        # -----------------------------------------------------
        if (not lr_frozen) and (not has_reduced_once) and (current_loss < 0.0):
        #if (not lr_frozen) and (not has_reduced_once) and (step > 1500):
            # Reduce LR by a factor (e.g. 0.1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * reduce_factor
            #model.gamma.data = torch.tensor(1e-10, dtype=torch.float32)
            has_reduced_once = True
            print(f"Step {step+1}: negative ELBO < 0, reducing LR to {param_group['lr']:.2e}, gamma = {model.gamma.item():.2e}")

        # (Optional) printing
        if (step+1) % 1000 == 0:
            print(f"Step {step+1}, negative ELBO = {current_loss:.6f}, lr = {optimizer.param_groups[0]['lr']}, gamma = {model.gamma.item():.2e}")

    # After training, form final estimate: u_hat = diag(w_x)*mu_z
    with torch.no_grad():
        mu_z, _ = model.encode(x_torch)  # shape (1, n)
        mu_z = mu_z.squeeze(0)          # => (n,)
        w_x = model.w_x                 # => (n,)
        u_hat_torch = w_x * mu_z        # elementwise => (n,)
        trained_gamma = model.gamma.data
        trained_gamma = trained_gamma.cpu().numpy()
        u_hat = u_hat_torch.cpu().numpy()

    return trained_gamma, u_hat
