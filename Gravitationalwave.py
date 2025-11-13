import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import time
import os


data_file = "gw_data.csv"     
n_steps = 400_000              
burn_in = 100_000              
step_sizes = np.array([5.0e-4, 0.0010, 0.007])  

start_pos = [1.0, 2.0, 10.0]  
random_seed = 42              


alpha_bounds = (0.0, 2.0)
beta_bounds  = (1.0, 10.0)
gamma_bounds = (1.0, 20.0)


def h_model(t, Alpha, Beta, Gamma):
    return Alpha * np.exp(t) * (1.0 - np.tanh(2.0 * (t - Beta))) * np.sin(Gamma * t)

def log_prior(theta):
    a, b, g = theta
    if (alpha_bounds[0] < a < alpha_bounds[1]) and (beta_bounds[0] < b < beta_bounds[1]) and (gamma_bounds[0] < g < gamma_bounds[1]):
        return 0.0
    return -np.inf

def log_likelihood(theta, t, y, y_err):
    a, b, g = theta
    ymod = h_model(t, a, b, g)
    chi2 = np.sum(((y - ymod) / y_err) ** 2)
    return -0.5 * chi2

def log_posterior(theta, t, y, y_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y, y_err)


def autocorr_fft(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - np.mean(x)
    # next power of two length for FFT
    nfft = 1 << ((2*n - 1).bit_length())
    s = np.fft.fft(x, n=nfft)
    acf = np.fft.ifft(s * np.conjugate(s)).real[:n]
    acf /= acf[0]
    return acf

def integrated_time(x, max_lag=None):
    acf = autocorr_fft(x)
    if max_lag is None:
        max_lag = min(len(acf)-1, 10000)
    tau = 1.0
    for lag in range(1, max_lag):
        if acf[lag] <= 0:
            break
        tau += 2.0 * acf[lag]
    return tau


if not os.path.exists(data_file):
    raise FileNotFoundError(f"'{data_file}' not found in cwd: {os.getcwd()}")

# prefer columns 0 and 2 (observed format has blank middle column)
try:
    df = pd.read_csv(data_file, header=0, comment='#', skipinitialspace=True, encoding='utf-8-sig', engine='python', usecols=[0, 2])
except Exception:
    df_all = pd.read_csv(data_file, header=0, comment='#', skipinitialspace=True, encoding='utf-8-sig', engine='python')
    df = df_all.iloc[:, [0, -1]]

df.columns = ['t', 'h']
df = df.apply(pd.to_numeric, errors='coerce').dropna()
if df.shape[0] == 0:
    raise ValueError("No numeric rows after CSV cleaning — check gw_data.csv format.")
t_data = df['t'].values
y_data = df['h'].values
print(f"Loaded {len(t_data)} data points.")

data_std_dev = np.std(y_data)
y_err = 0.20 * data_std_dev

print(f"Using constant (homoscedastic) error (20% of data std dev): sigma = {y_err:.4f}")


rng = np.random.default_rng(random_seed)
theta = np.array(start_pos, dtype=float)
current_lp = log_posterior(theta, t_data, y_data, y_err)

chain = np.zeros((n_steps, 3))
n_accept = 0

print(f"Starting single-chain MCMC: n_steps={n_steps}, burn_in={burn_in}")
t0 = time.time()
for i in range(n_steps):
    proposal = theta + rng.normal(scale=step_sizes, size=3)
    prop_lp = log_posterior(proposal, t_data, y_data, y_err)
    if prop_lp > current_lp or rng.random() < np.exp(prop_lp - current_lp):
        theta = proposal
        current_lp = prop_lp
        n_accept += 1
    chain[i] = theta
    # occasional status
    if (i+1) % 50000 == 0:
        print(f"  step {i+1}/{n_steps}, accept_rate ~ {n_accept/(i+1):.3f}")
t1 = time.time()
print(f"Single-chain MCMC finished in {t1 - t0:.1f} s")

acceptance_rate = n_accept / n_steps
print(f"Acceptance rate: {acceptance_rate:.3f}")



if burn_in >= n_steps:
    raise ValueError("burn_in must be < n_steps")

chain_post = chain[burn_in:, :]
n_post = chain_post.shape[0]
print(f"Using {n_post} post-burn-in samples for diagnostics and plotting")

# Estimate autocorrelation time and ESS for each parameter
taus = []
ess = []
for p in range(3):
    tau_p = integrated_time(chain_post[:, p])
    taus.append(tau_p)
    ess_p = n_post / max(1.0, 2.0 * tau_p)   # single chain ESS approx
    ess.append(ess_p)

print("Estimated integrated autocorrelation times (tau):", [f"{t:.1f}" for t in taus])
print("Approx. effective sample sizes (ESS):", [f"{e:.0f}" for e in ess])

# Posterior summaries
means = np.mean(chain_post, axis=0)
medians = np.median(chain_post, axis=0)
stds = np.std(chain_post, axis=0)
print("Posterior means (alpha, beta, gamma):", means)
print("Posterior medians:", medians)
print("Posterior stds:", stds)


labels = ["Alpha (α)", "Beta (β)", "Gamma (γ)"]
colors = ["red", "red", "red"]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for p in range(3):
    axes[p].plot(chain_post[:, p], color=colors[p], alpha=0.7, lw=0.5)
    axes[p].set_ylabel(labels[p])
    axes[p].set_title(f"Trace Plot for {labels[p]}")
axes[-1].set_xlabel("Post-burn-in step")
fig.suptitle("MCMC Trace Plots", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("trace_fit.png")
plt.show()



thin = max(1, int(max(1, ess[0]//50)))  
combined_for_corner = chain_post[::thin]
fig_corner = corner.corner(combined_for_corner, labels=labels, quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12})

fig_corner.suptitle(" ", fontsize=14)
plt.savefig("corner_fit.png")
plt.show()


alpha_fit, beta_fit, gamma_fit = medians
print(f"Best-fit (median) parameters: alpha={alpha_fit:.4f}, beta={beta_fit:.4f}, gamma={gamma_fit:.4f}")

t_model = np.linspace(np.min(t_data), np.max(t_data), 2000)
y_model = h_model(t_model, alpha_fit, beta_fit, gamma_fit)

plt.figure(figsize=(10, 5))
plt.plot(t_data, y_data, 'k.', alpha=0.25, markersize=2, label='data')
plt.plot(t_model, y_model, 'r-', lw=2, label=f'model α={alpha_fit:.3f}, β={beta_fit:.3f}, γ={gamma_fit:.3f}')
plt.xlabel('t')
plt.ylabel('h')
plt.title('Best-fit Model vs Data')
plt.legend()
plt.grid(True)
plt.savefig("best_fit.png")

plt.show()

print("\nFinished. Files saved:")




