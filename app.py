from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Protein Folding Dynamics", page_icon="🧬", layout="wide")

st.title("Protein Folding Dynamics")
st.subheader("Takens Embedding and Lyapunov Spectrum Analysis of Folding Trajectories")
st.write(
    "This interactive application reproduces the analysis in the submitted CMDS project notebook. "
    "It examines the dynamics-phase RMSD time series using summary statistics, ACF/PACF, "
    "the ADF test, mutual information, false nearest neighbours, Takens embedding and "
    "the Rosenstein divergence approach."
)

DEFAULT_CSV = Path(__file__).parent / "protein_folding_rmsd_timeseries.csv"

@st.cache_data
def load_data(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

uploaded = st.sidebar.file_uploader("Upload RMSD dataset (CSV)", type=["csv"])
if uploaded is not None:
    df = load_data(uploaded.getvalue())
    st.sidebar.success("Using uploaded CSV")
elif DEFAULT_CSV.exists():
    df = load_data(DEFAULT_CSV.read_bytes())
    st.sidebar.info("Using the project CSV included with this app")
else:
    st.error("Dataset not found. Upload the project CSV in the sidebar.")
    st.stop()

required_cols = {"Phase", "Time_ps", "RMSD_Angstrom"}
missing = required_cols.difference(df.columns)
if missing:
    st.error("The dataset is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

md = df[df["Phase"] == "dynamics"].copy()
md["Time_ps"] = pd.to_numeric(md["Time_ps"], errors="coerce")
md["RMSD_Angstrom"] = pd.to_numeric(md["RMSD_Angstrom"], errors="coerce")
md = md.dropna(subset=["Time_ps", "RMSD_Angstrom"]).reset_index(drop=True)
if len(md) < 40:
    st.error(f"Only {len(md)} valid dynamics observations were found. The project analysis needs a longer series.")
    st.stop()

x = md["RMSD_Angstrom"].to_numpy()
tau, m = 6, 4
dt, theiler, kmax = 0.5, 10, 15

# Project calculations, kept aligned with the submitted notebook.
def mutual_info(values, lag, bins=6):
    a, b = values[:-lag], values[lag:]
    p, _, _ = np.histogram2d(a, b, bins=bins)
    p /= p.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return (p[nz] * np.log(p[nz] / (px @ py)[nz])).sum()

def fnn_percent(values, delay, dimension, rtol=10, atol=2):
    n_points = len(values) - dimension * delay
    if n_points <= 1:
        return np.nan
    Y_fnn = np.column_stack([
        values[j * delay:j * delay + n_points] for j in range(dimension)
    ])
    d, idx = cKDTree(Y_fnn).query(Y_fnn, k=2)
    d, idx = np.maximum(d[:, 1], 1e-12), idx[:, 1]
    nxt = values[dimension * delay:dimension * delay + n_points]
    extra = np.abs(nxt - nxt[idx])
    false = (extra / d > rtol) | (
        np.sqrt(d**2 + extra**2) / values.std() > atol
    )
    return 100 * false.mean()

st.sidebar.markdown("---")
st.sidebar.markdown("**Notebook parameters**")
st.sidebar.write("Time delay τ = 6 frames (3 ps)")
st.sidebar.write("Embedding dimension m = 4")
st.sidebar.write("Frame interval = 0.5 ps")
st.sidebar.write("Theiler window = 10 frames")
st.sidebar.write("Maximum follow-up = 15 frames")

a, b, c = st.columns(3)
a.metric("Dynamics observations", f"{len(md)}")
b.metric("Mean RMSD", f"{md['RMSD_Angstrom'].mean():.3f} Å")
c.metric("RMSD standard deviation", f"{md['RMSD_Angstrom'].std():.3f} Å")

st.caption("The four minimization rows are excluded, as in the notebook. Only rows with Phase = dynamics are analysed.")

tab1, tab2, tab3, tab4 = st.tabs([
    "RMSD & Summary", "Time-series tests", "Embedding", "Lyapunov analysis"
])

with tab1:
    st.header("1. RMSD vs Time")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(md["Time_ps"], md["RMSD_Angstrom"])
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Backbone RMSD (Å)")
    ax.set_title("RMSD vs time")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.header("2. Summary statistics")
    st.dataframe(md["RMSD_Angstrom"].describe().to_frame(), use_container_width=True)
    st.write("First five dynamics observations")
    st.dataframe(md.head(), use_container_width=True)

with tab2:
    st.header("3. Autocorrelation plots")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(md["RMSD_Angstrom"], lags=30, ax=ax[0])
    plot_pacf(md["RMSD_Angstrom"], lags=30, ax=ax[1])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.header("4. Stationarity test (ADF)")
    stat, p, *_ = adfuller(md["RMSD_Angstrom"])
    q1, q2 = st.columns(2)
    q1.metric("ADF statistic", f"{stat:.3f}")
    q2.metric("p-value", f"{p:.4f}")
    if p < 0.05:
        st.success("At the 5% significance level, the unit-root null hypothesis is rejected. The series is classified as stationary by this test.")
    else:
        st.warning("At the 5% significance level, the unit-root null hypothesis is not rejected.")

    st.header("5. Choosing the time delay (τ) using mutual information")
    lags = list(range(1, 21))
    mi = [mutual_info(x, lag) for lag in lags]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, mi, marker="o")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Mutual information")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.info("Notebook selection: τ = 6 frames (3 ps), chosen as the first local minimum described in the project. A longer delay around lag 10–11 was not used because the series contains only 100 dynamics frames.")

with tab3:
    st.header("6. Choosing embedding dimension (m) using False Nearest Neighbours")
    dims = list(range(1, 7))
    pct = [fnn_percent(x, tau, dim) for dim in dims]
    fnn_table = pd.DataFrame({
        "Embedding dimension (m)": dims,
        "False neighbours (%)": [round(v, 1) for v in pct]
    })
    st.dataframe(fnn_table, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(dims), pct, marker="o")
    ax.set_xlabel("Embedding dimension (m)")
    ax.set_ylabel("% false neighbours")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.info("Notebook selection: m = 4, where the false-neighbour percentage reaches 0%.")

    st.header("7. Takens embedding")
    n = len(x) - (m - 1) * tau
    Y = np.column_stack([x[j * tau:j * tau + n] for j in range(m)])
    st.write(f"Embedded points: **{Y.shape[0]} × {Y.shape[1]}**")
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(Y[:, 0], Y[:, 1], marker="o", ms=3, lw=0.8)
    ax1.set_xlabel("x(t)")
    ax1.set_ylabel(f"x(t+{tau})")
    ax1.set_title("2D phase space")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(Y[:, 0], Y[:, 1], Y[:, 2], lw=0.8)
    ax2.set_xlabel("x(t)")
    ax2.set_ylabel(f"x(t+{tau})")
    ax2.set_zlabel(f"x(t+{2*tau})")
    ax2.set_title("3D phase space")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The four-dimensional embedding is calculated; the notebook visualises its first two and first three coordinates.")

with tab4:
    st.header("8. Largest Lyapunov exponent — Rosenstein divergence curve")
    n = len(x) - (m - 1) * tau
    Y = np.column_stack([x[j * tau:j * tau + n] for j in range(m)])
    tree = cKDTree(Y)
    nn = np.zeros(n, dtype=int)
    try:
        for i in range(n):
            _, idx = tree.query(Y[i], k=n)
            nn[i] = next(j for j in idx[1:] if abs(j - i) > theiler)
    except (StopIteration, IndexError):
        st.error("Could not find a valid nearest-neighbour pair for every embedded point.")
        st.stop()

    div = []
    for k in range(kmax + 1):
        distances = [
            np.linalg.norm(Y[i+k] - Y[nn[i]+k])
            for i in range(n)
            if i+k < n and nn[i]+k < n
            and np.linalg.norm(Y[i+k] - Y[nn[i]+k]) > 0
        ]
        div.append(np.mean(np.log(distances)) if distances else np.nan)
    div = np.array(div)
    time_ps = np.arange(kmax + 1) * dt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ps, div, marker="o")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Mean log divergence")
    ax.set_title("Rosenstein divergence curve")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    fit_to = 4
    slope = np.polyfit(time_ps[:fit_to + 1], div[:fit_to + 1], 1)[0]
    st.metric("Largest Lyapunov exponent (approx.)", f"{slope:.3f} per ps")
    st.write("The notebook fits the early rise from 0 to 2 ps (frames 0 to 4) to estimate the exponent.")
    st.warning("Interpretation limitation: the time series is short (100 dynamics frames), so the exponent is a rough estimate and should not be treated as definitive evidence of chaos.")

with st.expander("Dataset information"):
    st.write(f"Full CSV shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Columns:", ", ".join(df.columns.astype(str)))
    st.dataframe(df.head(10), use_container_width=True)

st.divider()
st.caption("Recreated from the submitted CMDS_Project notebook and its accompanying RMSD CSV. Calculations and parameter choices follow the notebook.")
