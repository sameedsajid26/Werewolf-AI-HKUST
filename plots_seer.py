import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# Set the root directory where the experiment folders are located
root_dir = "experiments_allan"

# Initialize an empty list to store data
all_metrics = []

# Required keys to ensure valid metrics files
required_keys = [
    'game_id', 'rounds_played', 'winner', 'seer_accuracy', 'voting_accuracy',
    'seer_reveal_rate', 'suspicion_change_rate', 'vote_discussion_alignment',
    'statement_variety_rate', 'werewolf_deception_rate'
]

# Walk through the experiments directory to find all game_metrics_*.json files
for dirpath, _, filenames in os.walk(root_dir):
    # Look for files matching game_metrics_*.json
    for file_path in glob.glob(os.path.join(dirpath, "game_metrics_*.json")):
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                # Verify that metrics is a dictionary
                if not isinstance(metrics, dict):
                    print(f"Skipping {file_path}: JSON is not a dictionary, got {type(metrics)}")
                    continue
                # Check for all required keys
                missing_keys = [key for key in required_keys if key not in metrics]
                if missing_keys:
                    print(f"Skipping {file_path}: Missing keys {missing_keys}")
                    continue
                # Check if rounds_played is valid
                if not isinstance(metrics['rounds_played'], (int, float)):
                    print(f"Skipping {file_path}: Invalid rounds_played value {metrics['rounds_played']}")
                    continue
                # Extract experiment name from the parent directory
                experiment_name = os.path.basename(os.path.dirname(file_path))
                metrics['experiment'] = experiment_name
                all_metrics.append(metrics)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {e}")

# Check if any valid data was collected
if not all_metrics:
    print("No valid game_metrics_*.json files found. Exiting.")
    exit()

# Convert to a pandas DataFrame
try:
    df = pd.DataFrame(all_metrics)
except Exception as e:
    print(f"Error creating DataFrame: {e}")
    print("Sample data:", all_metrics[:2])
    exit()

# Verify that rounds_played is in the DataFrame
if 'rounds_played' not in df.columns:
    print("Error: 'rounds_played' not found in DataFrame columns. Available columns:", list(df.columns))
    print("Sample data:", df.head().to_dict())
    exit()

# Ensure numeric columns are properly typed
numeric_columns = [
    'rounds_played', 'seer_accuracy', 'voting_accuracy', 'seer_reveal_rate',
    'suspicion_change_rate', 'vote_discussion_alignment', 'statement_variety_rate',
    'werewolf_deception_rate'
]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Column {col} not found in DataFrame")

# Check for missing values and warn if any
if df[numeric_columns].isna().any().any():
    print("Warning: Some numeric columns contain NaN values. Check data integrity.")
    print(df[numeric_columns].isna().sum())

# Save combined data to CSV
df.to_csv('combined_metrics.csv', index=False)
print("Combined data saved to 'combined_metrics.csv'")

# Compute summary statistics
total_games = len(df)
villager_wins = len(df[df['winner'] == 'Villagers win!'])
werewolf_wins = len(df[df['winner'] == 'Werewolves win!'])
villager_win_rate = villager_wins / total_games * 100 if total_games > 0 else 0
werewolf_win_rate = werewolf_wins / total_games * 100 if total_games > 0 else 0
avg_rounds = df['rounds_played'].mean() if not df['rounds_played'].isna().all() else 0

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total Games: {total_games}")
print(f"Villager Wins: {villager_wins} ({villager_win_rate:.2f}%)")
print(f"Werewolf Wins: {werewolf_wins} ({werewolf_win_rate:.2f}%)")
print(f"Average Rounds per Game: {avg_rounds:.2f}")
print("\nAverage Metrics:")
for col in numeric_columns:
    avg_value = df[col].mean() if col in df.columns and not df[col].isna().all() else 0
    print(f"{col}: {avg_value:.4f}")

# Set up plotting style
sns.set(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# Create directory for saving plots_allan
os.makedirs('plots_allan', exist_ok=True)

# 1. Win Rates Bar Chart
win_counts = df['winner'].value_counts(normalize=True) * 100
plt.figure(figsize=(8, 6))
win_counts.plot(kind='bar', color=['#4CAF50', '#F44336'])
plt.title('Win Rates: Villagers vs. Werewolves')
plt.xlabel('Winner')
plt.ylabel('Percentage of Games (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots_allan/win_rates.png', dpi=300)
plt.close()
print("Saved win_rates.png")

# 2. Box plots_allan for Numeric Metrics by Winner
for col in numeric_columns:
    if col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='winner', y=col, data=df, palette=['#4CAF50', '#F44336'])
        plt.title(f'{col.replace("_", " ").title()} by Game Winner')
        plt.xlabel('Winner')
        plt.ylabel(col.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(f'plots_allan/{col}_by_winner.png', dpi=300)
        plt.close()
        print(f"Saved {col}_by_winner.png")
    else:
        print(f"Skipping box plot for {col}: Column not found")

# 3. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Game Metrics')
plt.tight_layout()
plt.savefig('plots_allan/correlation_heatmap.png', dpi=300)
plt.close()
print("Saved correlation_heatmap.png")

# 4. Scatter Plot: Seer Accuracy vs. Voting Accuracy
if 'seer_accuracy' in df.columns and 'voting_accuracy' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='seer_accuracy', y='voting_accuracy', hue='winner', style='winner',
                    data=df, palette=['#4CAF50', '#F44336'], s=100)
    plt.title('Seer Accuracy vs. Voting Accuracy')
    plt.xlabel('Seer Accuracy')
    plt.ylabel('Voting Accuracy')
    plt.legend(title='Winner')
    plt.tight_layout()
    plt.savefig('plots_allan/seer_vs_voting_accuracy.png', dpi=300)
    plt.close()
    print("Saved seer_vs_voting_accuracy.png")
else:
    print("Skipping seer_vs_voting_accuracy plot: Missing required columns")

# 5. Scatter Plot: Rounds Played vs. Seer Accuracy
if 'rounds_played' in df.columns and 'seer_accuracy' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='rounds_played', y='seer_accuracy', hue='winner', style='winner',
                    data=df, palette=['#4CAF50', '#F44336'], s=100)
    plt.title('Rounds Played vs. Seer Accuracy')
    plt.xlabel('Rounds Played')
    plt.ylabel('Seer Accuracy')
    plt.legend(title='Winner')
    plt.tight_layout()
    plt.savefig('plots_allan/rounds_vs_seer_accuracy.png', dpi=300)
    plt.close()
    print("Saved rounds_vs_seer_accuracy.png")
else:
    print("Skipping rounds_vs_seer_accuracy plot: Missing required columns")

# 6. Box Plot: Seer Accuracy by Rounds Played
if 'rounds_played' in df.columns and 'seer_accuracy' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='rounds_played', y='seer_accuracy', data=df, palette='Set2')
    plt.title('Seer Accuracy by Rounds Played')
    plt.xlabel('Rounds Played')
    plt.ylabel('Seer Accuracy')
    plt.tight_layout()
    plt.savefig('plots_allan/seer_accuracy_by_rounds.png', dpi=300)
    plt.close()
    print("Saved seer_accuracy_by_rounds.png")
else:
    print("Skipping seer_accuracy_by_rounds plot: Missing required columns")

print("\nAll plots_allan saved in 'plots_allan' directory.")