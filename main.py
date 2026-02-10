import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# Load datasets
ball_by_ball_df = pd.read_csv(
    'ipl_ball_by_ball_2008_2025_cleaned.csv',
    dtype={'non_striker': str, 'player_dismissed': str, 'fielder': str, 'dismissal_kind': str, 'extras_type': str}
)
matches_df = pd.read_csv('ipl_matches_2008_2025_cleaned.csv')
print("Datasets loaded.")


def create_features(ball_df, matches_df):

    print("Creating features...")
    ball_df = ball_df.copy()

    match_results = matches_df[['id', 'winner']].rename(columns={'id': 'match_id'})
    df = pd.merge(ball_df, match_results, on='match_id', how='left')

    match_innings_groups = df.groupby(['match_id', 'inning'])
    processed_data = []

    for (match_id, inning), group in match_innings_groups:
        group = group.sort_values(['over', 'ball']).reset_index(drop=True)
        batting_team = group['batting_team'].iloc[0]
        bowling_team = group['bowling_team'].iloc[0]

        group['cum_runs'] = group['total_runs'].cumsum()
        group['cum_wickets'] = group['is_wicket'].cumsum()
        group['ball_number'] = range(1, len(group) + 1)
        group['overs_completed'] = (group['ball_number'] - 1) // 6
        group['balls_in_over'] = (group['ball_number'] - 1) % 6
        group['overs_display'] = group['overs_completed'] + group['balls_in_over'] / 10

        balls_bowled = group['overs_completed'] + group['balls_in_over'] / 6
        group['run_rate'] = np.where(balls_bowled > 0, group['cum_runs'] / balls_bowled, 0)
        group['balls_played'] = group['ball_number']

        if inning == 1:
            target = group['cum_runs'].iloc[-1] + 1
            max_balls = group['balls_played'].iloc[-1]
            max_overs = (max_balls + 5) // 6
            group['req_runs'] = np.nan
            group['req_overs'] = np.nan
            group['req_run_rate'] = np.nan

        else:
            try:
                first_inning = match_innings_groups.get_group((match_id, 1))
                target = first_inning['total_runs'].sum() + 1
                max_overs = (len(first_inning) + 5) // 6

                group['req_runs'] = (target - group['cum_runs']).clip(lower=0)

                overs_faced = (group['ball_number'] - 1) // 6 + ((group['ball_number'] - 1) % 6) / 6
                group['req_overs'] = (max_overs - overs_faced).clip(lower=1/120)

                group['req_run_rate'] = np.where(
                    group['req_overs'] > 0,
                    group['req_runs'] / group['req_overs'],
                    999.0
                )
                group.loc[group['req_runs'] <= 0, 'req_run_rate'] = 0.0

            except KeyError:
                print(f"Warning: No first inning for match {match_id}")
                target = np.nan
                max_overs = 20
                group['req_runs'] = np.nan
                group['req_overs'] = np.nan
                group['req_run_rate'] = np.nan

        group['target'] = target if inning == 2 else np.nan
        group['max_overs'] = max_overs
        group['wickets_remaining'] = (10 - group['cum_wickets']).clip(lower=0)
        group['balls_remaining'] = (max_overs * 6 - group['balls_played']).clip(lower=0)

        winner = group['winner'].iloc[0] if pd.notna(group['winner'].iloc[0]) else None
        group['batting_team_won'] = 1 if winner == batting_team else 0

        processed_data.append(group)

    processed_df = pd.concat(processed_data, ignore_index=True)
    second_innings = processed_df[processed_df['inning'] == 2].copy()

    feature_cols = [
        'match_id', 'inning', 'batting_team', 'bowling_team',
        'over', 'ball', 'overs_display', 'cum_runs', 'cum_wickets',
        'wickets_remaining', 'balls_remaining', 'run_rate', 'target',
        'req_runs', 'req_run_rate', 'batting_team_won', 'balls_played'
    ]

    check_cols = [
        'wickets_remaining', 'balls_remaining', 'req_runs',
        'req_run_rate', 'cum_runs', 'cum_wickets', 'run_rate',
        'batting_team_won', 'target'
    ]
    model_data = second_innings[feature_cols].dropna(subset=check_cols)

    print(f"Total rows: {len(processed_df)}, Model rows: {len(model_data)}")
    return processed_df, model_data


def train_model(model_data):

    print("Training model...")

    features = [
        'wickets_remaining', 'balls_remaining', 'req_runs',
        'req_run_rate', 'cum_runs', 'cum_wickets', 'run_rate', 'target'
    ]
    X = model_data[features]
    y = model_data['batting_team_won']

    # handle bad values
    X = X.fillna(X.median())
    X.replace([np.inf, -np.inf], 999, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight='balanced', n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    return model, features


def calculate_win_probabilities(match_id, all_data, model, features):

    match_data = all_data[all_data['match_id'] == match_id].copy()
    match_data = match_data.sort_values(['inning', 'over', 'ball']).reset_index(drop=True)

    if match_data.empty:
        print(f"No data for match {match_id}")
        return None, None, None

    team1 = match_data[match_data['inning'] == 1]['batting_team'].iloc[0]
    team2 = match_data[match_data['inning'] == 1]['bowling_team'].iloc[0]

    match_data['win_prob_team1'] = 0.5
    match_data['win_prob_team2'] = 0.5

    # first innings - simple heuristic
    first_innings = match_data[match_data['inning'] == 1]
    for idx, row in first_innings.iterrows():
        wickets_lost = row['cum_wickets']
        balls_rem = row['balls_remaining'] if pd.notna(row['balls_remaining']) else 0
        projected = row['cum_runs'] + ((10 - wickets_lost) * 1.0) * (balls_rem / 6 if balls_rem > 0 else 0)
        score_factor = min(max(projected / 200, 0), 1)
        win_prob = 0.5 + (score_factor - 0.5) * 0.8
        win_prob = min(max(win_prob, 0.05), 0.95)

        match_data.loc[idx, 'win_prob_team1'] = win_prob
        match_data.loc[idx, 'win_prob_team2'] = 1 - win_prob

    # second innings - model prediction
    second_innings = match_data[match_data['inning'] == 2]
    if not second_innings.empty and model is not None:
        X_pred = second_innings[features].copy()
        X_pred = X_pred.fillna(0)
        X_pred.replace([np.inf, -np.inf], 999, inplace=True)

        win_probs = model.predict_proba(X_pred)[:, 1]
        match_data.loc[second_innings.index, 'win_prob_team2'] = win_probs
        match_data.loc[second_innings.index, 'win_prob_team1'] = 1 - win_probs

    # fill gaps
    match_data['win_prob_team1'] = match_data['win_prob_team1'].ffill().bfill().fillna(0.5)
    match_data['win_prob_team2'] = match_data['win_prob_team2'].ffill().bfill().fillna(0.5)

    return match_data, team1, team2


def plot_win_probability(match_data, team1, team2, match_id):

    if match_data is None:
        return

    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    team1_probs = match_data['win_prob_team1'].values
    team2_probs = match_data['win_prob_team2'].values

    plt.plot(match_data.index, team1_probs, label=f'{team1}', linewidth=2.5, color='darkorange')
    plt.plot(match_data.index, team2_probs, label=f'{team2}', linewidth=2.5, color='royalblue')

    # innings break line
    inn2_start = match_data[match_data['inning'] == 2].index.min()
    if pd.notna(inn2_start) and inn2_start > 0:
        plt.axvline(x=inn2_start, color='grey', linestyle='--', linewidth=1.5, label='Innings Break')
        plt.text(inn2_start / 2, 0.02, f"{team1} Innings", ha='center', fontsize=10)
        plt.text((inn2_start + len(match_data)) / 2, 0.02, f"{team2} Innings", ha='center', fontsize=10)

    # wickets
    wickets = match_data[match_data['is_wicket'] == 1]
    if not wickets.empty:
        wicket_probs = np.where(
            wickets['inning'] == 1,
            team1_probs[wickets.index],
            team2_probs[wickets.index]
        )
        plt.scatter(wickets.index, wicket_probs, color='red', s=60, zorder=5, label='Wicket', marker='X')

    plt.xlabel('Ball Number')
    plt.ylabel('Win Probability')
    plt.title(f'{team1} vs {team2} (Match {match_id})')
    plt.axhline(y=0.5, color='grey', linestyle=':', alpha=0.7)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# Main
all_data, model_data = create_features(ball_by_ball_df, matches_df)

win_prob_model, model_features = train_model(model_data)

match_id_example = 335982

match_plot_data, team1, team2 = calculate_win_probabilities(
    match_id_example, all_data, win_prob_model, model_features
)

if match_plot_data is not None:
    plot_win_probability(match_plot_data, team1, team2, match_id_example)
else:
    print(f"Could not plot match {match_id_example}")