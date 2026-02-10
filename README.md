# IPL Win Probability Predictor

A machine learning project that predicts win probabilities during IPL cricket matches using ball-by-ball data from 2008 to 2025.

## What it does

- Merges and cleans IPL datasets (2008-2025)
- Engineers features from ball-by-ball data (run rate, required run rate, wickets remaining, etc.)
- Trains a Random Forest model on second innings data
- Predicts win probability at every ball of a match
- Plots win probability progression with wicket markers and innings breaks

## How it works

The model uses a two-phase approach:

- **First innings:** Heuristic-based probability using projected score
- **Second innings:** Random Forest model using features like required run rate, wickets remaining, balls remaining, and target

## Files

| File | Description |
|------|-------------|
| `merge_datasets.py` | Merges 2008-2024 and 2025 datasets into one |
| `clean_data.py` | Cleans merged data (fixes nulls, team names, types) |
| `win_probability.py` | Feature engineering, model training, prediction and plotting |

## Datasets

- `df_2025.csv` — Ball-by-ball data for IPL 2025
- `df_main_2008_2024.csv` — Ball-by-ball data for IPL 2008-2024
- `df_matches_2008_2024.csv` — Match-level data for IPL 2008-2024

After running the pipeline:
- `ipl_ball_by_ball_2008_2025.csv` — Merged ball-by-ball data
- `ipl_ball_by_ball_2008_2025_cleaned.csv` — Cleaned version
- `ipl_matches_2008_2025.csv` — Merged match data
- `ipl_matches_2008_2025_cleaned.csv` — Cleaned version

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn
```

# Note: few big files can not be uploaded on github. you can find them on: https://pdpuacin-my.sharepoint.com/:f:/g/personal/23bit168_pdpu_ac_in/IgATx3t0J_ayTYq7kQlExO_fAUssjubBD1N_FS-93j9ICqc?e=978FkR
