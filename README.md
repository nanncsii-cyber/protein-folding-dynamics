# Protein Folding Dynamics — Streamlit app

This app is based on the submitted `CMDS_Project(3).ipynb` notebook and `protein_folding_rmsd_timeseries(1).csv` dataset.

## Run locally
Install Python 3.10+ and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at the local address printed by Streamlit (usually http://localhost:8501).
The supplied CSV is included and loads automatically. You can also upload another CSV from the sidebar.

## Deploy online
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `protein_folding_rmsd_timeseries.csv`.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file to `app.py`.
5. Click **Deploy**.

The app filters `Phase == dynamics` and reproduces the notebook's analysis and parameter choices: τ=6, m=4, dt=0.5 ps, Theiler window=10, kmax=15, and early-fit frames 0–4.
