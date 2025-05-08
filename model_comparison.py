import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# Set the root directory where the experiment folders are located
root_dir = "experiments"

# Initialize an empty list to store data
all_metrics = []

# Required keys to ensure valid metrics files
required_keys = [
    'game_id', 'rounds_played', 'winner', 'seer_accuracy', 'voting_accuracy', 'suspicion_change_rate', 'vote_discussion_alignment',
    'statement_variety_rate', 'werewolf_deception_rate'
]

# Walk through the experiments directory to find all game_metrics_*.json files
for dirpath, _, filenames in os.walk(root_dir):
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
                # Extract experiment name and determine model
                experiment_name = os.path.basename(os.path.dirname(file_path))
                if experiment_name.startswith('experiment_seer_4o_'):
                    model = '4o'
                elif experiment_name.startswith('experiment_seer_'):
                    model = '4o-mini'
                else:
                    print(f"Skipping {file_path}: Unknown model for experiment {experiment_name}")
                    continue
                metrics['experiment'] = experiment_name
                metrics['model'] = model
                all_metrics.append(metrics)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {e}")

# Check if any valid data Eurosystem
# Convert to a pandas DataFrame
if not all_metrics:
    print("No valid game_metrics_*.json files found. Exiting.")
    exit()

try:
    df = pd.DataFrame(all_metrics)
except Exception as e:
    print(f"Error creating DataFrame: {e}")
    print("Sample data:", all_metrics[:2])
    exit()

# Verify key columns
if 'rounds_played' not in df.columns:
    print("Error: 'rounds_played' not found in DataFrame columns. Available columns:", list(df.columns))
    exit()

# Ensure numeric columns are properly typed
numeric_columns = [
    'rounds_played', 'seer_accuracy', 'voting_accuracy',
    'suspicion_change_rate', 'vote_discussion_alignment', 'statement_variety_rate',
    'werewolf_deception_rate'
]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Column {col} not found in DataFrame")

# Check for missing values
if df[numeric_columns].isna().any().any():
    print("Warning: Some numeric columns contain NaN values:")
    print(df[numeric_columns].isna().sum())

# Save combined data to CSV
df.to_csv('combined_metrics_comp.csv', index=False)
print("Combined data saved to 'combined_metrics_comp.csv'")

# Compute summary statistics by model
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    total_games = len(model_df)
    villager_wins = len(model_df[model_df['winner'] == 'Villagers win!'])
    werewolf_wins = len(model_df[model_df['winner'] == 'Werewolves win!'])
    villager_win_rate = villager_wins / total_games * 100 if total_games > 0 else 0
    werewolf_win_rate = werewolf_wins / total_games * 100 if total_games > 0 else 0
    avg_rounds = model_df['rounds_played'].mean() if not model_df['rounds_played'].isna().all() else 0

    print(f"\nSummary Statistics for {model}:")
    print(f"Total Games: {total_games}")
    print(f"Villager Wins: {villager_wins} ({villager_win_rate:.2f}%)")
    print(f"Werewolf Wins: {werewolf_wins} ({werewolf_win_rate:.2f}%)")
    print(f"Average Rounds per Game: {avg_rounds:.2f}")
    print("Average Metrics:")
    for col in numeric_columns:
        avg_value = model_df[col].mean() if col in model_df.columns and not model_df[col].isna().all() else 0
        print(f"{col}: {avg_value:.4f}")

# Set up plotting style
sns.set(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# Create directory for saving plots_model_comp
os.makedirs('plots_model_comp', exist_ok=True)

# 1. Win Rates Bar Chart by Model
win_rates = df.groupby(['model', 'winner']).size().unstack(fill_value=0)
win_rates = win_rates.div(win_rates.sum(axis=1), axis=0) * 100
plt.figure(figsize=(10, 6))
win_rates.plot(kind='bar', color=['#4CAF50', '#F44336'])
plt.title('Win Rates by Model: Villagers vs. Werewolves')
plt.xlabel('Model')
plt.ylabel('Percentage of Games (%)')
plt.xticks(rotation=0)
plt.legend(title='Winner')
plt.tight_layout()
plt.savefig('plots_model_comp/win_rates_by_model.png', dpi=300)
plt.close()
print("Saved win_rates_by_model.png")

# 2. Box plots_model_comp for Numeric Metrics by Model and Winner
for col in numeric_columns:
    if col in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='model', y=col, hue='winner', data=df, palette=['#4CAF50', '#F44336'])
        plt.title(f'{col.replace("_", " ").title()} by Model and Winner')
        plt.xlabel('Model')
        plt.ylabel(col.replace("_", " ").title())
        plt.legend(title='Winner')
        plt.tight_layout()
        plt.savefig(f'plots_model_comp/{col}_by_model_winner.png', dpi=300)
        plt.close()
        print(f"Saved {col}_by_model_winner.png")
    else:
        print(f"Skipping box plot for {col}: Column not found")

# 3. Bar Chart of Average Metrics by Model
avg_metrics = df.groupby('model')[numeric_columns].mean().T
plt.figure(figsize=(12, 6))
avg_metrics.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Average Metrics by Model')
plt.xlabel('Metric')
plt.ylabel('Average Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('plots_model_comp/avg_metrics_by_model.png', dpi=300)
plt.close()
print("Saved avg_metrics_by_model.png")

# 4. Violin plots_model_comp for Key Metrics by Model
key_metrics = ['seer_accuracy', 'voting_accuracy', 'werewolf_deception_rate']
for col in key_metrics:
    if col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='model', y=col, data=df, palette=['#1f77b4', '#ff7f0e'])
        plt.title(f'{col.replace("_", " ").title()} Distribution by Model')
        plt.xlabel('Model')
        plt.ylabel(col.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(f'plots_model_comp/{col}_violin_by_model.png', dpi=300)
        plt.close()
        print(f"Saved {col}_violin_by_model.png")
    else:
        print(f"Skipping violin plot for {col}: Column not found")

# 5. Scatter Plot: Seer Accuracy vs. Voting Accuracy by Model
if 'seer_accuracy' in df.columns and 'voting_accuracy' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='seer_accuracy', y='voting_accuracy', hue='model', style='model',
                    data=df, palette=['#1f77b4', '#ff7f0e'], s=100)
    plt.title('Seer Accuracy vs. Voting Accuracy by Model')
    plt.xlabel('Seer Accuracy')
    plt.ylabel('Voting Accuracy')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('plots_model_comp/seer_vs_voting_accuracy_by_model.png', dpi=300)
    plt.close()
    print("Saved seer_vs_voting_accuracy_by_model.png")
else:
    print("Skipping seer_vs_voting_accuracy plot: Missing required columns")

print("\nAll plots_model_comp saved in 'plots_model_comp' directory.")