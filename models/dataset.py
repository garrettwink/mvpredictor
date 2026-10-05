import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

conn = sqlite3.connect("nba.db")

df_stat = pd.read_sql_query("SELECT * FROM player_season_history", conn)
df_award = pd.read_sql_query("SELECT * FROM mvp_awards", conn)

df = df_stat.merge(
    df_award[['player_id', 'season', 'mvp_rank', 'points_won', 'mvp_vote_share']],
    on=['player_id', 'season'],
    how='left'
)

df['mvp_rank'] = pd.to_numeric(df['mvp_rank'], errors='coerce').astype('Int64')

df.loc[df['mvp_rank'] == 1, 'mvp'] = 1
df.fillna(0, inplace=True)

train_seasons = np.arange(2010, 2022, 1)   # 2010–2021
test_seasons = np.arange(2022, 2026, 1)    # 2022–2025

train = df[df['season'].isin(train_seasons)].sort_values('season', kind='stable').reset_index(drop=True)
test = df[df['season'].isin(test_seasons)].sort_values('season', kind='stable').reset_index(drop=True)

feature_cols = [
    'pts', 'ast', 'reb', 'stl', 'blk', 'tov',
    'fg_pct', 'fg3_pct', 'ft_pct', 'gp', 'mpg', 'usg_pct',
    'net_rating', 'pie', 'ts_pct', 'age',
    'team_win_pct', 'playoff_clinch'
]

id_cols = ['player_id', 'player_name', 'season']

group_sizes = train.groupby("season", sort=True).size().to_numpy()

X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = train['mvp_vote_share']
y_test = test['mvp_vote_share']
id_train = train[id_cols]
id_test = test[id_cols]




 

