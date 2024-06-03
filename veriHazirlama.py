# import pandas as pd

# # Read data from CSV files
# games_details = pd.read_csv('archive/games_details.csv')
# games = pd.read_csv('archive/games.csv')
# players = pd.read_csv('archive/players.csv')
# ranking = pd.read_csv('archive/ranking.csv')
# teams = pd.read_csv('archive/teams.csv')

# # Merge games and games_details on GAME_ID
# games_merged = pd.merge(games, games_details, on='GAME_ID')

# # Calculate mean statistics for home and away teams separately
# home_stats = games_merged.groupby('HOME_TEAM_ID').agg(
#     {'PTS': 'mean', 'FG_PCT': 'mean', 'FT_PCT': 'mean', 'FG3_PCT': 'mean', 
#      'AST': 'mean', 'REB': 'mean', 'HOME_TEAM_WINS': 'mean'}
# ).reset_index()

# away_stats = games_merged.groupby('VISITOR_TEAM_ID').agg(
#     {'PTS': 'mean', 'FG_PCT': 'mean', 'FT_PCT': 'mean', 'FG3_PCT': 'mean', 
#      'AST': 'mean', 'REB': 'mean'}
# ).reset_index()

# # Merge ranking and teams data
# team_data = pd.merge(ranking, teams, on='TEAM_ID')

# # Merge home and away statistics with team data
# home_team_data = pd.merge(team_data, home_stats, left_on='TEAM_ID', right_on='HOME_TEAM_ID')
# visit_team_data = pd.merge(team_data, away_stats, left_on='TEAM_ID', right_on='VISITOR_TEAM_ID')

# # Select relevant columns
# home_team_data = home_team_data[['TEAM_ID', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 
#                                  'AST', 'REB', 'HOME_RECORD', 'W_PCT']]
# visit_team_data = visit_team_data[['TEAM_ID', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 
#                                    'AST', 'REB', 'ROAD_RECORD', 'W_PCT']]

# # Rename columns to differentiate home and visit teams
# home_team_data.columns = ['TEAM_ID', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 
#                           'AST', 'REB', 'HOME_RECORD', 'W_PCT']
# visit_team_data.columns = ['TEAM_ID', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 
#                            'AST', 'REB', 'ROAD', 'W_PCT']

# # Select only numeric columns for the mean calculation, excluding 'TEAM_ID'
# numeric_columns = home_team_data.select_dtypes(include='number').columns
# numeric_columns = numeric_columns.drop('TEAM_ID', errors='ignore')

# numeric_columns1 = visit_team_data.select_dtypes(include='number').columns
# numeric_columns1 = numeric_columns1.drop('TEAM_ID', errors='ignore')

# # Group by 'TEAM_ID' and apply mean on numeric columns
# home_team_data_mean = home_team_data.groupby('TEAM_ID')[numeric_columns].mean().reset_index()
# visit_team_data_mean = visit_team_data.groupby('TEAM_ID')[numeric_columns1].mean().reset_index()

# # Write the final datasets to new CSV files
# home_team_data_mean.to_csv('archive/home_team_stats.csv', index=False)
# visit_team_data_mean.to_csv('archive/visit_team_stats.csv', index=False)


# import pandas as pd

# # Takım adı
# team_name = "Warriors"

# # Takım adının bulunduğu veri dosyasını yükleme
# teams_df = pd.read_csv('archive/teams.csv')

# # Takım adına göre TEAM_ID'yi bulma
# team_id = teams_df[teams_df['NICKNAME'] == team_name]['TEAM_ID'].values[0]

# # HOME_TEAM_ID'sine göre home_team_stats.csv dosyasından ilgili verileri çekme
# home_team_stats_df = pd.read_csv('archive/home_team_stats.csv')
# team_stats = home_team_stats_df[home_team_stats_df['TEAM_ID'] == team_id]
# print("TEAM_ID:", team_stats['TEAM_ID'])
# print("PTS:", team_stats['PTS'])
# print("FG_PCT:", team_stats['FG_PCT'])
# print("FT_PCT:", team_stats['FG_PCT'])
# print("FG3_PCT:", team_stats['FG3_PCT'])
# print("AST:", team_stats['AST'])
# print("REB:", team_stats['REB'])
# print("W_PCT:", team_stats['W_PCT'])



import pandas as pd

teams_df = pd.read_csv('archive/teams.csv')

team_name_H = "Warriors"
team_name_V = "Nuggets"

# Takım adına göre TEAM_ID'yi bulma
team_id_H= teams_df[teams_df['NICKNAME'] == team_name_H]['TEAM_ID'].values[0]
team_id_V = teams_df[teams_df['NICKNAME'] == team_name_V]['TEAM_ID'].values[0]

# HOME_TEAM_ID'sine göre home_team_stats.csv dosyasından ilgili verileri çekme
home_team_stats_df = pd.read_csv('archive/home_team_stats.csv')
team_stats = home_team_stats_df[home_team_stats_df['TEAM_ID'] == team_id_H]

visit_team_stats_df = pd.read_csv('archive/visit_team_stats.csv')
team_stats1 = visit_team_stats_df[visit_team_stats_df['TEAM_ID'] == team_id_H]

new_game = [team_stats['PTS'] * team_stats['W_PCT'], team_stats['FG_PCT'] * team_stats['W_PCT'], team_stats['FG_PCT'] * team_stats['W_PCT'], team_stats['FG3_PCT'] * team_stats['W_PCT'], team_stats['AST'] * team_stats['W_PCT'], team_stats['REB'] * team_stats['W_PCT'],team_stats1['PTS'] * team_stats1['W_PCT'], team_stats1['FG_PCT'] * team_stats1['W_PCT'],
             team_stats1['FG_PCT'] * team_stats1['W_PCT'], team_stats1['FG3_PCT'] * team_stats1['W_PCT'], team_stats1['AST'] * team_stats1['W_PCT'], team_stats1['REB'] * team_stats1['W_PCT']]

print("Warriors Stats (Home Team):")
print("PTS * W_PCT:", new_game[0])
print("FG_PCT * W_PCT:", new_game[1])
print("FT_PCT * W_PCT:", new_game[2])
print("FG3_PCT * W_PCT:", new_game[3])
print("AST * W_PCT:", new_game[4])
print("REB * W_PCT:", new_game[5])

print("\nDenver Stats (Visitor Team):")
print("PTS * W_PCT:", new_game[6])
print("FG_PCT * W_PCT:", new_game[7])
print("FT_PCT * W_PCT:", new_game[8])
print("FG3_PCT * W_PCT:", new_game[9])
print("AST * W_PCT:", new_game[10])
print("REB * W_PCT:", new_game[11])