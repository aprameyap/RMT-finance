import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch
import imageio
from sklearn.preprocessing import StandardScaler

os.makedirs("frames", exist_ok=True)

nifty_50 = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "MARUTI.NS", "NTPC.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "TITAN.NS", "INDUSINDBK.NS",
    "NESTLEIND.NS", "TECHM.NS", "JSWSTEEL.NS", "GRASIM.NS", "WIPRO.NS", "COALINDIA.NS", "ADANIPORTS.NS",
    "BAJAJ-AUTO.NS", "DRREDDY.NS", "BRITANNIA.NS", "EICHERMOT.NS", "CIPLA.NS", "ONGC.NS", "HINDALCO.NS",
    "HDFCLIFE.NS", "BAJAJFINSV.NS", "DIVISLAB.NS", "TATAMOTORS.NS", "BPCL.NS", "HEROMOTOCO.NS", "SBILIFE.NS",
    "TATASTEEL.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "M&M.NS", "DEEPAKNTR.NS", "ADANIENT.NS", "ICICIPRULI.NS"
]

sector_map = {
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "POWERGRID": "Utilities", "NTPC": "Utilities",
    "ICICIBANK": "Financials", "HDFCBANK": "Financials", "KOTAKBANK": "Financials", "AXISBANK": "Financials",
    "SBIN": "Financials", "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "ICICIPRULI": "Financials",
    "SBILIFE": "Financials", "HDFCLIFE": "Financials", "INDUSINDBK": "Financials",
    "INFY": "IT", "TCS": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "ASIANPAINT": "Consumer", "HINDUNILVR": "Consumer", "NESTLEIND": "Consumer", "ITC": "Consumer",
    "BRITANNIA": "Consumer", "TITAN": "Consumer",
    "SUNPHARMA": "Healthcare", "CIPLA": "Healthcare", "DRREDDY": "Healthcare",
    "DIVISLAB": "Healthcare", "APOLLOHOSP": "Healthcare",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals", "COALINDIA": "Metals",
    "ADANIPORTS": "Transport", "ADANIENT": "Conglomerate", "GRASIM": "Materials", "SHREECEM": "Materials",
    "EICHERMOT": "Auto", "TATAMOTORS": "Auto", "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "M&M": "Auto",
    "LT": "Industrials", "DEEPAKNTR": "Chemicals", "BHARTIARTL": "Telecom"
}

data = yf.download(nifty_50, period="10y")['Close']
data.dropna(axis=0, how='any', inplace=True)
returns = np.log(data / data.shift(1)).dropna()

window_size = 250
step_size = 5
top_k = 3 # Top k eigenvalues to track

tickers_short = [ticker.split(".")[0] for ticker in data.columns]
sectors = [sector_map.get(t, "Other") for t in tickers_short]
unique_sectors = sorted(set(sectors))
sector_colors = dict(zip(unique_sectors, sns.color_palette("tab20", len(unique_sectors))))
bar_colors = [sector_colors[s] for s in sectors]

frames = []
dates = []

for start in range(0, len(returns) - window_size, step_size):
    end = start + window_size
    window_returns = returns.iloc[start:end]
    T_w = window_returns.shape[0]
    N = window_returns.shape[1]

    # Normalize returns
    R = (window_returns - window_returns.mean()) / window_returns.std()
    X = R.values / np.sqrt(T_w)
    E_t = X.T @ X

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(E_t)
    top_indices = np.argsort(eigvals)[-top_k:][::-1]
    top_vecs = eigvecs[:, top_indices]

    fig, axes = plt.subplots(1, top_k, figsize=(5*top_k, 4))
    for i in range(top_k):
        vec = top_vecs[:, i]
        axes[i].bar(range(N), vec, color=bar_colors)
        axes[i].set_title(f"v{i+1}(t): {window_returns.index[-1].date()}")
        axes[i].set_xticks([])
        axes[i].set_ylim([-0.3, 0.3])
    
    legend_handles = [Patch(color=color, label=sector) for sector, color in sector_colors.items()]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.12, 1))

    plt.tight_layout()
    frame_path = f"frames/frame_{start//step_size:04d}.png"
    plt.savefig(frame_path)
    plt.close()
    frames.append(frame_path)
    dates.append(window_returns.index[-1])

with imageio.get_writer("eigenvectors_animation.gif", mode="I", duration=0.2) as writer:
    for f in frames:
        image = imageio.imread(f)
        writer.append_data(image)

print("Saved animation as eigenvectors_animation.gif")
