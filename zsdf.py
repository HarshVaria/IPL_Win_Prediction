import pandas as pd

# Load datasets
df_2025 = pd.read_csv('df_2025.csv')
df_main_2008_2024 = pd.read_csv('df_main_2008_2024.csv')
df_matches_2008_2024 = pd.read_csv('df_matches_2008_2024.csv')

columns_final = [
    'match_id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball',
    'batter', 'bowler', 'non_striker', 'batsman_runs', 'extra_runs', 'total_runs',
    'extras_type', 'is_wicket', 'player_dismissed', 'dismissal_kind', 'fielder'
]

# Process 2008-2024
print("Processing 2008-2024 data...")
df_combined = pd.DataFrame(df_main_2008_2024[columns_final].values, columns=columns_final)

# Process 2025
print("Processing 2025 data...")
df_2025_processed = pd.DataFrame(columns=columns_final)

df_2025_processed['match_id'] = df_2025['match_id']
df_2025_processed['inning'] = df_2025['innings']
df_2025_processed['batting_team'] = df_2025['batting_team']
df_2025_processed['bowling_team'] = df_2025['bowling_team']
df_2025_processed['over'] = df_2025['over'].apply(lambda x: int(x))
df_2025_processed['ball'] = df_2025['over'].apply(lambda x: int((x - int(x)) * 10))
df_2025_processed['batter'] = df_2025['striker']
df_2025_processed['bowler'] = df_2025['bowler']
df_2025_processed['non_striker'] = None
df_2025_processed['batsman_runs'] = df_2025['runs_of_bat']
df_2025_processed['extra_runs'] = df_2025['extras']
df_2025_processed['total_runs'] = df_2025['runs_of_bat'] + df_2025['extras']

# extras type
def determine_extras_type(row):
    if row['wide'] == 1:
        return 'wides'
    elif row['legbyes'] == 1:
        return 'legbyes'
    elif row['byes'] == 1:
        return 'byes'
    elif row['noballs'] == 1:
        return 'noballs'
    return None

df_2025_processed['extras_type'] = df_2025.apply(determine_extras_type, axis=1)
df_2025_processed['is_wicket'] = df_2025['wicket_type'].notna().astype(int)
df_2025_processed['player_dismissed'] = df_2025['player_dismissed']
df_2025_processed['dismissal_kind'] = df_2025['wicket_type']
df_2025_processed['fielder'] = df_2025['fielder']

# combine ball by ball
print("Combining...")
df_combined = pd.concat([df_combined, df_2025_processed], ignore_index=True)

# match data for 2025
match_data_2025 = df_2025[['match_id', 'season', 'phase', 'date', 'venue', 'batting_team', 'bowling_team']].drop_duplicates()
match_data_2025 = match_data_2025.rename(columns={'match_id': 'id', 'phase': 'match_type'})

match_data_2025['city'] = match_data_2025['venue'].apply(lambda x: x.split(',')[1].strip() if ',' in x else x)
match_data_2025['player_of_match'] = None
match_data_2025['team1'] = match_data_2025['batting_team']
match_data_2025['team2'] = match_data_2025['bowling_team']
match_data_2025['toss_winner'] = None
match_data_2025['toss_decision'] = None
match_data_2025['winner'] = None
match_data_2025['result'] = None
match_data_2025['result_margin'] = None
match_data_2025['target_runs'] = None
match_data_2025['target_overs'] = None
match_data_2025['super_over'] = 'N'
match_data_2025['method'] = None
match_data_2025['umpire1'] = None
match_data_2025['umpire2'] = None

# combine matches
df_matches_combined = pd.concat([df_matches_2008_2024, match_data_2025], ignore_index=True)

# save
print("Saving...")
df_combined.to_csv('ipl_ball_by_ball_2008_2025.csv', index=False)
df_matches_combined.to_csv('ipl_matches_2008_2025.csv', index=False)

print("Done.")