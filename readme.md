## Sampling with the Help of a Generative Model

This practical notebook introduces how normalizing flows can be used to improve Monte Carlo estimation for a multi-modal target distribution.

The tutorial combines:
- flow-based density modeling,
- importance sampling,
- and Markov Chain Monte Carlo (MCMC).

## Learning Goals

By the end of the notebook, you should be able to:
- build and inspect a simple 2D Mixture of Gaussians (MoG) target,
- instantiate and validate a Real-NVP style normalizing flow,
- train the flow with a Monte Carlo approximation of KL divergence,
- compute and interpret importance weights,
- identify failure modes of importance sampling,
- use a trained flow as a proposal in MCMC.

## Notebook Content

The notebook is organized in five main parts:

1. **Toy target distribution (2D MoG)**  
	Defines a two-component Gaussian mixture with imbalanced weights.

2. **Normalizing flow basics**  
	Instantiates a minimal flow and checks invertibility (`forward`/`backward`).

3. **Training the flow**  
	Uses an empirical KL objective,
	```math
	\mathcal{L}(\theta) \approx \frac{1}{B}\sum_{k=1}^B \left[\log \rho_\theta(x^k)-\log \rho_*(x^k)\right],\quad x^k\sim \rho_\theta
	```
	and monitors convergence through loss and density plots.

4. **Importance sampling with the trained flow**  
	Computes importance weights,
	```math
	w_i = \frac{\rho_*(x_i)}{\rho_\theta(x_i)}
	```
	and uses normalized weights to estimate target-region probabilities and discuss when IS can fail.

5. **Flow-based MCMC proposal**  
	Demonstrates MALA and independent Metropolis-Hastings style sampling with multiple parallel chains.

## Files in This Practical

- `sampling_gen_model_tutorial.ipynb`: main tutorial notebook.
- `models.py`: MoG and normalizing flow implementations.
- `utils_plot.py`: plotting helpers.
- `utils_mcmc.py`: MCMC utilities (MALA and MH helpers).

## How to Run

1. Open the notebook `sampling_gen_model_tutorial.ipynb`.
2. Run cells from top to bottom.
3. If you modify Python modules (`models.py`, `utils_mcmc.py`), restart the kernel or reload modules before re-running dependent cells.

Suggested environment:
- Python 3.10+
- PyTorch
- Matplotlib
- NumPy
- tqdm

## Credit and Attribution

This tutorial is based on course material by **Miss Marylou Gabrié**, teacher at **École Polytechnique, CMAP**.

Please keep this attribution if you reuse, adapt, or share this notebook.