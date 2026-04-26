import numpy as np
import torch
import tqdm

def accept_filter(log_ratio, x, x_init):

    log_u = torch.log(torch.rand_like(log_ratio))
    acc = log_u < log_ratio
    x[~acc] = x_init[~acc]

    return x, acc


def get_grad_U(U):
    """
    Function to get grad of input function via autodiff
   
    U: function with single tensor input
    """
    @torch.enable_grad()
    def grad_U(x):
        x = x.clone().detach().requires_grad_()
        return torch.autograd.grad(U(x).sum(), x)[0]
    return grad_U

def run_mala(target_U, grad_U, x_init, n_steps, dt, beta_eff=1, with_tqdm=False):
    """
    target_U: function - target potential we will run the Langevin on (negative log prob)
    gard_U: function - grad of target_U
    x (tensor): init points for the chains to update (batch_dim, dim)
    dt : time step
    beta_eff: additional control to change temperature of target U
    """
    xs = []
    accs = []

    range_ = tqdm.tqdm(range(n_steps)) if with_tqdm else range(n_steps)

    for t in range_:
        x = x_init.clone()
        x = x_init - dt * grad_U(x_init)
        if dt > 0:
            x += dt * np.sqrt(2 / (dt * beta_eff)) * torch.randn_like(x_init)

        log_ratio = -target_U(x)
        log_ratio -= ((x_init - x + dt * grad_U(x)) ** 2 / (4 * dt)).sum(-1)
        log_ratio += target_U(x_init)
        log_ratio += ((x - x_init + dt * grad_U(x_init)) ** 2 / (4 * dt)).sum(-1)
        log_ratio = beta_eff * log_ratio

        x, acc = accept_filter(log_ratio, x, x_init)

        accs.append(acc)
        xs.append(x.clone())
        x_init = x.clone().detach()

    return torch.stack(xs), torch.stack(accs)

def run_mcmc(x_init, proposal, target, n_steps, with_tqdm=False):
    """
    Independent Metropolis-Hastings using `proposal` as an independence proposal.
    Works for a single chain `(dim,)` or batched chains `(n_chains, dim)`.
    """

    single_chain = x_init.dim() == 1
    
    if single_chain:
        x_init = x_init.unsqueeze(0)

    n_chains, dim = x_init.shape
    samples = torch.zeros(
        (n_steps + 1, n_chains, dim), dtype=x_init.dtype, device=x_init.device
    )
    samples[0] = x_init.clone()

    range_ = tqdm.tqdm(range(1, n_steps + 1)) if with_tqdm else range(1, n_steps + 1)
    for i in range_:
        x_prev = samples[i - 1]
        try:
            candidate = proposal.sample(n_chains)
        except TypeError:
            candidate = proposal.sample((n_chains,))
        if candidate.dim() == 1:
            candidate = candidate.unsqueeze(0)

        log_diff_target = target.log_prob(candidate) - target.log_prob(x_prev)
        log_diff_prop = proposal.log_prob(x_prev) - proposal.log_prob(candidate)
        log_accept_ratio = log_diff_target + log_diff_prop

        log_u = torch.log(torch.rand_like(log_accept_ratio))
        acc = log_u < log_accept_ratio

        x_new = x_prev.clone()
        x_new[acc] = candidate[acc]
        samples[i] = x_new

    samples = samples[1:]
    if single_chain:
        return samples.squeeze(1)
    return samples