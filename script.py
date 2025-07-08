import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

nifty_50 = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "MARUTI.NS", "NTPC.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "TITAN.NS", "INDUSINDBK.NS",
    "NESTLEIND.NS", "TECHM.NS", "JSWSTEEL.NS", "GRASIM.NS", "WIPRO.NS", "COALINDIA.NS", "ADANIPORTS.NS",
    "BAJAJ-AUTO.NS", "DRREDDY.NS", "BRITANNIA.NS", "EICHERMOT.NS", "CIPLA.NS", "ONGC.NS", "HINDALCO.NS",
    "HDFCLIFE.NS", "BAJAJFINSV.NS", "DIVISLAB.NS", "TATAMOTORS.NS", "BPCL.NS", "HEROMOTOCO.NS", "SBILIFE.NS",
    "TATASTEEL.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "M&M.NS", "DEEPAKNTR.NS", "ADANIENT.NS", "ICICIPRULI.NS"
]

data = yf.download(nifty_50, period="10y")['Close']
data.dropna(axis=0, how='any', inplace=True)  # Remove rows with missing prices

returns = np.log(data / data.shift(1)).dropna()

window_size = 250
step_size = 5      
max_eigs = []       
dates = []

window_size = 250 
step_size = 5 
top_k = 3       


eigval_traces = [[] for _ in range(top_k)]
dates = []

for start in range(0, len(returns) - window_size, step_size):
    end = start + window_size
    window_returns = returns.iloc[start:end]
    T_w = window_returns.shape[0]
    N = window_returns.shape[1]

    R = (window_returns - window_returns.mean()) / window_returns.std()
    X = R.values / np.sqrt(T_w)
    E_t = X.T @ X  # correlation matrix

    # Get top-k eigenvalues (descending)
    eigvals = np.linalg.eigvalsh(E_t)[::-1]
    for i in range(top_k):
        eigval_traces[i].append(eigvals[i])

    dates.append(window_returns.index[-1])

plt.figure(figsize=(14, 6))
for i in range(top_k):
    plt.plot(dates, eigval_traces[i], label=f"λ{i+1}(t)")
plt.axhline((1 + np.sqrt(N / window_size))**2, linestyle='--', color='gray', label='MP λ₊')
plt.title("Top Eigenvalues Over Time (Sliding Window)")
plt.xlabel("Time")
plt.ylabel("Eigenvalue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

standardized = (returns - returns.mean()) / returns.std()
T, N = standardized.shape
q = N / T
print(f"Total days (T): {T}, Total stocks (N): {N}, q = N/T = {q:.4f}")

X = standardized / np.sqrt(T)

E = np.dot(X.T, X)
eigenvalues = np.linalg.eigvalsh(E)

# Plot ESD 
plt.figure(figsize=(10, 6))
bins = np.linspace(eigenvalues.min(), eigenvalues.max(), 60)
plt.hist(eigenvalues, bins=bins, density=True, alpha=0.6, label="Empirical ESD")

# MP fit
lambda_min = (1 - np.sqrt(q))**2
lambda_max = (1 + np.sqrt(q))**2
lambda_vals = np.linspace(lambda_min, lambda_max, 1000)
mp_rho = (1 / (2 * np.pi * q * lambda_vals)) * np.sqrt((lambda_max - lambda_vals) * (lambda_vals - lambda_min))
mp_rho = np.nan_to_num(mp_rho)
plt.plot(lambda_vals, mp_rho, label="MP Fit", color='red')

plt.title("Empirical Spectral Density (ESD)")
plt.xlabel("Eigenvalue")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

threshold = np.percentile(eigenvalues, 95)
tail_eigs = eigenvalues[eigenvalues > threshold]

hist, bin_edges = np.histogram(tail_eigs, bins=20, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
nonzero = hist > 0
log_x = np.log(bin_centers[nonzero])
log_y = np.log(hist[nonzero])
slope, intercept, _, _, _ = linregress(log_x, log_y)
alpha = -slope

print(f"\nEstimated Power-law exponent (α) from eigenvalue tail: {alpha:.2f}")
