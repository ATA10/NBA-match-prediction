import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
import csv
from tabulate import tabulate

# Matplotlib ve Seaborn görselleştirme için stil ayarları
plt.style.use('ggplot')
sns.set(style='whitegrid')

# Verileri yükleyin
games_details_df = pd.read_csv('archive/games_details.csv', dtype={'PLAYER_NAME': str})
games_df = pd.read_csv('archive/games.csv')
players_df = pd.read_csv('archive/players.csv')
ranking_df = pd.read_csv('archive/ranking.csv')
teams_df = pd.read_csv('archive/teams.csv')
home_team_stats_df = pd.read_csv('archive/home_team_stats.csv')
visit_team_stats_df = pd.read_csv('archive/visit_team_stats.csv')

# Gerekli özellikleri çıkarma
features = ['PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 
            'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']

# Hedef değişken: HOME_TEAM_WINS
target = 'HOME_TEAM_WINS'

X = games_df[features]
y = games_df[target]

# Eksik değerleri kontrol etme
print('Eksik değerleri kontrol etme')
print(X.isnull().sum())
print(y.isnull().sum())

# Eksik değerleri doldurma (varsa)
X.fillna(X.mean(), inplace=True)
y.fillna(y.mode()[0], inplace=True)

# Verilerin istatistiksel özetini çıkarma
print('Verilerin istatistiksel özetini çıkarma')
print(X.describe())
print(y.value_counts())

# Verileri görselleştirme
sns.pairplot(games_df[features + [target]], hue=target)
plt.show()
plt.savefig('Figure_1.png')

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Modelini Oluşturma ve Eğitme
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_scaled, y_train)

# Test seti üzerinde tahmin yapma
y_pred = knn.predict(X_test_scaled)

# Doğruluk oranını hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Doğruluk Oranı: {accuracy * 100:.2f}%')

## Doğruluk oranını görselleştirme
accuracy_percentage = accuracy * 100
plt.figure(figsize=(8, 6))
sns.barplot(x=['Model Doğruluk Oranı'], y=[accuracy_percentage], palette='Blues_d')
plt.ylim(0, 100)
plt.ylabel('Doğruluk Oranı (%)')
plt.title('Model Doğruluk Oranı')
plt.text(0, accuracy_percentage / 2, f'{accuracy_percentage:.2f}%', ha='center', va='center', color='white', fontsize=15)
plt.tight_layout()

# Görseli kaydetme
plt.savefig('model_accuracy.png')

# Sınıflandırma raporu ve karışıklık matrisi
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('KNN Confusion Matrix')
# plt.show()
plt.savefig('knn_confusion_matrix.png')

# Karar Ağacı Modeli
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_scaled, y_train)

# Karar Ağacı Görselleştirme
plt.figure(figsize=(20,10))
plot_tree(dtree, filled=True, feature_names=features, class_names=['Deplasman Kazanır', 'Ev Sahibi Kazanır'])
# plt.show()
plt.savefig('decision_tree.png')


#  Model kaydetme
# KNN modelini kaydetme
joblib.dump(knn, 'model/knn_model.pkl')

# Karar Ağacı modelini kaydetme
joblib.dump(dtree, 'model/decision_tree_model.pkl')


def predict_game_result(team_name_H, team_name_V, teams_df, home_team_stats_df, visit_team_stats_df, scaler, knn, dtree):
    # Takım adına göre TEAM_ID'yi bulma
    team_id_H = teams_df[teams_df['NICKNAME'] == team_name_H]['TEAM_ID'].values[0]
    team_id_V = teams_df[teams_df['NICKNAME'] == team_name_V]['TEAM_ID'].values[0]

    # HOME_TEAM_ID'sine göre home_team_stats.csv dosyasından ilgili verileri çekme
    team_stats_H = home_team_stats_df[home_team_stats_df['TEAM_ID'] == team_id_H]
    team_stats_V = visit_team_stats_df[visit_team_stats_df['TEAM_ID'] == team_id_V]

    new_game = [[
        team_stats_H['PTS'].values[0] * team_stats_H['W_PCT'].values[0],
        team_stats_H['FG_PCT'].values[0] * team_stats_H['W_PCT'].values[0],
        team_stats_H['FT_PCT'].values[0] * team_stats_H['W_PCT'].values[0],
        team_stats_H['FG3_PCT'].values[0] * team_stats_H['W_PCT'].values[0],
        team_stats_H['AST'].values[0] * team_stats_H['W_PCT'].values[0],
        team_stats_H['REB'].values[0] * team_stats_H['W_PCT'].values[0],
        team_stats_V['PTS'].values[0] * team_stats_V['W_PCT'].values[0],
        team_stats_V['FG_PCT'].values[0] * team_stats_V['W_PCT'].values[0],
        team_stats_V['FT_PCT'].values[0] * team_stats_V['W_PCT'].values[0],
        team_stats_V['FG3_PCT'].values[0] * team_stats_V['W_PCT'].values[0],
        team_stats_V['AST'].values[0] * team_stats_V['W_PCT'].values[0],
        team_stats_V['REB'].values[0] * team_stats_V['W_PCT'].values[0]
    ]]
    
    new_game_scaled = scaler.transform(new_game)
    
    # KNN Modeli ile Tahmin
    predicted_result_knn = knn.predict(new_game_scaled)
    knn_winner = team_name_H if predicted_result_knn[0] == 1 else team_name_V
    
    print(f'{team_name_H} - {team_name_V}')
    print(f'KNN Modeline Göre Tahmin Edilen Maç Sonucu: {knn_winner} (1: Ev sahibi kazanır, 0: Deplasman kazanır)')

    # Karar Ağacı Modeli ile Tahmin
    predicted_result_dtree = dtree.predict(new_game_scaled)
    dtree_winner = team_name_H if predicted_result_dtree[0] == 1 else team_name_V
    
    print(f'{team_name_H} - {team_name_V}')
    print(f'Karar Ağacı Modeline Göre Tahmin Edilen Maç Sonucu: {dtree_winner} (1: Ev sahibi kazanır, 0: Deplasman kazanır)')

    return knn_winner, dtree_winner

# # Örnek maç tahmini
# team_name_H = "Denver"
# team_name_V = "warriors"
# knn_result, dtree_result = predict_game_result(team_name_H, team_name_V, teams_df, home_team_stats_df, visit_team_stats_df, scaler, knn, dtree)
# print(f"KNN sonucu: {knn_result}, Karar Ağacı sonucu: {dtree_result}")

def create_league(teams):
    league = []
    for i, team in enumerate(teams):
        other_teams = teams[:i] + teams[i+1:]
        for other_team in other_teams:
            league.append((team, other_team))
    return league

def play_matches(league, predict_func):
    results = []
    for match in league:
        home_team, visitor_team = match
        knn_winner, dtree_winner = predict_func(home_team, visitor_team, teams_df, home_team_stats_df, visit_team_stats_df, scaler, knn, dtree)
        results.append((home_team, visitor_team, knn_winner, dtree_winner))
    return results

def calculate_points(results):
    team_points = {}
    for match in results:
        home_team, visitor_team, knn_winner, dtree_winner = match
        winner = knn_winner  # or dtree_winner based on your preference
        if winner == "Tie":
            team_points[home_team] = team_points.get(home_team, 0) + 1
            team_points[visitor_team] = team_points.get(visitor_team, 0) + 1
        else:
            team_points[winner] = team_points.get(winner, 0) + 3
    return team_points

def sort_standings(team_points):
    standings = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
    return standings

teams = teams_df['NICKNAME'].tolist()
league = create_league(teams)
results = play_matches(league, predict_game_result)
team_points = calculate_points(results)
standings = sort_standings(team_points)

# Print standings
print("League Standings:")
for i, (team, points) in enumerate(standings, 0):
    print(f"{i + 1}. {team}: {points} points")

# Save results to CSV
with open('match_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Home Team', 'Visitor Team', 'KNN Winner', 'Decision Tree Winner']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for match in results:
        writer.writerow({'Home Team': match[0], 'Visitor Team': match[1], 'KNN Winner': match[2], 'Decision Tree Winner': match[3]})

# Visualize standings
teams = [team[0] for team in standings]
points = [team[1] for team in standings]

plt.figure(figsize=(10, 6))
plt.barh(teams, points, color='skyblue')
plt.xlabel('Points')
plt.ylabel('Teams')
plt.title('League Standings')
plt.gca().invert_yaxis()
plt.show()
plt.savefig('Team_Skor.png')
