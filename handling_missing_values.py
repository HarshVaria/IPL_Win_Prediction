import pandas as pd
import numpy as np

# loading datasets
print("Loading datasets...")
try:
    ball_by_ball_df = pd.read_csv('ipl_ball_by_ball_2008_2025.csv')
    matches_df = pd.read_csv('ipl_matches_2008_2025.csv')
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

print(f"Ball-by-ball data shape: {ball_by_ball_df.shape}")
print(f"Matches data shape: {matches_df.shape}")



# filling missing extras_type with 'none' (regular deliveries)
null_extras_count = ball_by_ball_df['extras_type'].isna().sum()
if null_extras_count > 0:
    print(f"Filling {null_extras_count} missing extras_type values with 'none'")
    ball_by_ball_df['extras_type'] = ball_by_ball_df['extras_type'].fillna('none')


# fixing numeric columns
for col in ['batsman_runs', 'extra_runs', 'total_runs', 'is_wicket']:
    null_count = ball_by_ball_df[col].isna().sum()
    if null_count > 0:
        print(f"Fixing {null_count} missing values in {col}")
        ball_by_ball_df[col] = pd.to_numeric(ball_by_ball_df[col], errors='coerce').fillna(0).astype(int)


# making sure over and ball are integers
for col in ['over', 'ball']:
    if ball_by_ball_df[col].dtype != 'int64':
        print(f"Converting {col} to integer")
        ball_by_ball_df[col] = pd.to_numeric(ball_by_ball_df[col], errors='coerce').fillna(0).astype(int)


# Fill missing super_over
if 'super_over' in matches_df.columns:
    null_super_over = matches_df['super_over'].isna().sum()
    if null_super_over > 0:
        print(f"Filling {null_super_over} missing super_over values with 'N'")
        matches_df['super_over'] = matches_df['super_over'].fillna('N')

# Fix team name inconsistencies
team_name_mapping = {
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
}


for old_name in team_name_mapping:
    for col in ['batting_team', 'bowling_team']:
        if col in ball_by_ball_df.columns:
            count = (ball_by_ball_df[col] == old_name).sum()
            if count > 0:
                print(f"Replacing {count} occurrences of '{old_name}' with '{team_name_mapping[old_name]}' in {col}")
                ball_by_ball_df[col] = ball_by_ball_df[col].replace(old_name, team_name_mapping[old_name])

    for col in ['team1', 'team2', 'winner', 'toss_winner']:
        if col in matches_df.columns:
            count = (matches_df[col] == old_name).sum()
            if count > 0:
                print(f"Replacing {count} occurrences of '{old_name}' with '{team_name_mapping[old_name]}' in {col}")
                matches_df[col] = matches_df[col].replace(old_name, team_name_mapping[old_name])

# Save cleaned datasets
print("\nSaving cleaned datasets...")
try:
    ball_by_ball_df.to_csv('ipl_ball_by_ball_2008_2025_cleaned.csv')
    matches_df.to_csv('ipl_matches_2008_2025_cleaned.csv')
    print("Datasets saved successfully.")
except Exception as e:
    print(f"Error saving datasets: {e}")

print("Data cleaning complete.")