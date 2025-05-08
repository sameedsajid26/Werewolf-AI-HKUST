import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def find_metrics_files():
    """Find all metrics.json files in game_logs directories"""
    # Look for metrics.json files in any directory that starts with game_logs_
    metrics_files = glob.glob("game_logs_*/metrics.json")
    return metrics_files

def load_metrics_data(metrics_files):
    """Load and combine data from all metrics.json files"""
    all_data = []
    
    for file_path in metrics_files:
        try:
            # Extract game_id from the directory name
            game_id = os.path.basename(os.path.dirname(file_path)).replace("game_logs_", "")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Add game_id to the data
                data['game_id'] = game_id
                
                # Flatten nested dictionaries for easier analysis
                flattened_data = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flattened_data[f"{key}_{subkey}"] = subvalue
                    else:
                        flattened_data[key] = value
                
                all_data.append(flattened_data)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    return df

def create_visualizations(df):
    """Create various visualizations from the combined metrics data"""
    # Set up the style
    plt.style.use('ggplot')
    
    # Create a directory for the visualizations
    output_dir = "metrics_visualizations_allan"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the combined data to CSV
    df.to_csv(f"{output_dir}/combined_metrics.csv", index=False)
    print(f"Combined metrics saved to {output_dir}/combined_metrics.csv")
    
    # 1. Game Outcomes
    plt.figure(figsize=(10, 6))
    outcome_counts = df['winner'].value_counts()
    plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Game Outcomes')
    plt.savefig(f"{output_dir}/game_outcomes.png")
    plt.close()
    
    # 2. Rounds Played Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rounds_played'], bins=10)
    plt.title('Distribution of Rounds Played')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/rounds_played_distribution.png")
    plt.close()
    
    # 3. Seer Performance
    seer_metrics = ['seer_performance_seer_accuracy', 'seer_performance_seer_reveal_rate']
    plt.figure(figsize=(12, 6))
    df[seer_metrics].boxplot()
    plt.title('Seer Performance Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seer_performance.png")
    plt.close()
    
    # 4. Werewolf Performance
    werewolf_metrics = ['werewolf_performance_deception_rate', 'werewolf_performance_team_coordination']
    plt.figure(figsize=(12, 6))
    df[werewolf_metrics].boxplot()
    plt.title('Werewolf Performance Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/werewolf_performance.png")
    plt.close()
    
    # 5. Village Performance
    village_metrics = ['village_performance_voting_accuracy', 'village_performance_consensus_rate']
    plt.figure(figsize=(12, 6))
    df[village_metrics].boxplot()
    plt.title('Village Performance Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/village_performance.png")
    plt.close()
    
    # 6. Discussion Metrics
    discussion_metrics = ['discussion_metrics_suspicion_change_rate', 
                         'discussion_metrics_vote_discussion_alignment',
                         'discussion_metrics_statement_variety_rate']
    plt.figure(figsize=(12, 6))
    df[discussion_metrics].boxplot()
    plt.title('Discussion Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/discussion_metrics.png")
    plt.close()
    
    # 7. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(14, 12))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Metrics')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    # 8. Time Series of Key Metrics (if game_id contains timestamp)
    try:
        # Convert game_id to datetime if it's a timestamp
        df['date'] = pd.to_datetime(df['game_id'], format='%Y%m%d_%H%M%S')
        df = df.sort_values('date')
        
        # Plot time series of key metrics
        plt.figure(figsize=(14, 8))
        plt.plot(df['date'], df['seer_performance_seer_accuracy'], marker='o', label='Seer Accuracy')
        plt.plot(df['date'], df['village_performance_voting_accuracy'], marker='s', label='Voting Accuracy')
        plt.plot(df['date'], df['werewolf_performance_deception_rate'], marker='^', label='Werewolf Deception')
        plt.title('Key Metrics Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_over_time.png")
        plt.close()
    except:
        print("Could not create time series plot. Game IDs may not be timestamps.")
    
    # 9. Create a summary table
    summary = df.describe()
    summary.to_csv(f"{output_dir}/metrics_summary.csv")
    
    # 10. Create a detailed report
    with open(f"{output_dir}/metrics_report.txt", 'w') as f:
        f.write("WEREWOLF GAME METRICS REPORT\n")
        f.write("===========================\n\n")
        
        f.write(f"Total games analyzed: {len(df)}\n\n")
        
        f.write("GAME OUTCOMES\n")
        f.write("-------------\n")
        for outcome, count in outcome_counts.items():
            f.write(f"{outcome}: {count} ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("AVERAGE METRICS\n")
        f.write("--------------\n")
        f.write(f"Average rounds played: {df['rounds_played'].mean():.1f}\n")
        f.write(f"Average seer accuracy: {df['seer_performance_seer_accuracy'].mean():.2f}\n")
        f.write(f"Average voting accuracy: {df['village_performance_voting_accuracy'].mean():.2f}\n")
        f.write(f"Average werewolf deception rate: {df['werewolf_performance_deception_rate'].mean():.2f}\n")
        f.write(f"Average village consensus rate: {df['village_performance_consensus_rate'].mean():.2f}\n")
        f.write(f"Average statement variety rate: {df['discussion_metrics_statement_variety_rate'].mean():.2f}\n")
        f.write("\n")
        
        f.write("CORRELATIONS WITH GAME OUTCOME\n")
        f.write("-----------------------------\n")
        # Create a binary column for werewolf wins
        df['werewolf_win'] = (df['winner'] == 'Werewolves win!').astype(int)
        # Get numeric columns including the new werewolf_win column
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        correlations = df[numeric_cols].corr()['werewolf_win'].sort_values(ascending=False)
        for metric, corr in correlations.items():
            if metric != 'werewolf_win':
                f.write(f"{metric}: {corr:.3f}\n")
    
    print(f"Visualizations and reports saved to {output_dir}/")

def main():
    print("Finding metrics files...")
    metrics_files = find_metrics_files()
    
    if not metrics_files:
        print("No metrics.json files found. Make sure they are in directories named 'game_logs_TIMESTAMP'")
        return
    
    print(f"Found {len(metrics_files)} metrics files")
    
    print("Loading and combining data...")
    df = load_metrics_data(metrics_files)
    
    print("Creating visualizations...")
    create_visualizations(df)
    
    print("Done!")

if __name__ == "__main__":
    main() 