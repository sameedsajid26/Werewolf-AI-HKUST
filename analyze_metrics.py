import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_metrics_files():
    """Load all metrics files from the current directory."""
    metrics_files = glob.glob("game_metrics_20250419_203*.json")
    metrics_data = []
    
    for file in metrics_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Extract timestamp from filename
                timestamp = file.split('_')[2].split('.')[0]
                data['timestamp'] = timestamp
                metrics_data.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return metrics_data

def create_summary_statistics(metrics_data):
    """Create summary statistics from the metrics data."""
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(metrics_data)
    
    # Calculate summary statistics
    summary = {
        "Total Games": len(df),
        "Villager Wins": len(df[df['winner'] == "Villagers win!"]),
        "Werewolf Wins": len(df[df['winner'] == "Werewolves win!"]),
        "Average Rounds": df['rounds_played'].mean(),
        "Average Seer Accuracy": df['seer_accuracy'].mean(),
        "Average Voting Accuracy": df['voting_accuracy'].mean(),
        "Average Seer Reveal Rate": df['seer_reveal_rate'].mean(),
        "Average Suspicion Change Rate": df['suspicion_change_rate'].mean(),
        "Average Vote Discussion Alignment": df['vote_discussion_alignment'].mean(),
        "Average Statement Variety Rate": df['statement_variety_rate'].mean(),
        "Average Werewolf Deception Rate": df['werewolf_deception_rate'].mean()
    }
    
    return summary, df

def generate_report(summary, df):
    """Generate a comprehensive report from the metrics data."""
    # Create report directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/werewolf_game_analysis_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("WEREWOLF GAME ANALYSIS REPORT\n")
        f.write("============================\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Total Games Analyzed: {summary['Total Games']}\n")
        f.write(f"Villager Wins: {summary['Villager Wins']} ({summary['Villager Wins']/summary['Total Games']*100:.1f}%)\n")
        f.write(f"Werewolf Wins: {summary['Werewolf Wins']} ({summary['Werewolf Wins']/summary['Total Games']*100:.1f}%)\n")
        f.write(f"Average Rounds per Game: {summary['Average Rounds']:.2f}\n\n")
        
        # Performance metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("------------------\n")
        f.write(f"Average Seer Accuracy: {summary['Average Seer Accuracy']*100:.2f}%\n")
        f.write(f"Average Voting Accuracy: {summary['Average Voting Accuracy']*100:.2f}%\n")
        f.write(f"Average Seer Reveal Rate: {summary['Average Seer Reveal Rate']*100:.2f}%\n")
        f.write(f"Average Suspicion Change Rate: {summary['Average Suspicion Change Rate']*100:.2f}%\n")
        f.write(f"Average Vote Discussion Alignment: {summary['Average Vote Discussion Alignment']*100:.2f}%\n")
        f.write(f"Average Statement Variety Rate: {summary['Average Statement Variety Rate']*100:.2f}%\n")
        f.write(f"Average Werewolf Deception Rate: {summary['Average Werewolf Deception Rate']*100:.2f}%\n\n")
        
        # Game-by-game breakdown
        f.write("GAME-BY-GAME BREAKDOWN\n")
        f.write("----------------------\n")
        for i, row in df.iterrows():
            f.write(f"Game {i+1} (ID: {row['game_id']})\n")
            f.write(f"  Winner: {row['winner']}\n")
            f.write(f"  Rounds: {row['rounds_played']}\n")
            f.write(f"  Seer Accuracy: {row['seer_accuracy']*100:.2f}%\n")
            f.write(f"  Voting Accuracy: {row['voting_accuracy']*100:.2f}%\n")
            f.write(f"  Seer Reveal Rate: {row['seer_reveal_rate']*100:.2f}%\n")
            f.write(f"  Suspicion Change Rate: {row['suspicion_change_rate']*100:.2f}%\n")
            f.write(f"  Vote Discussion Alignment: {row['vote_discussion_alignment']*100:.2f}%\n")
            f.write(f"  Statement Variety Rate: {row['statement_variety_rate']*100:.2f}%\n")
            f.write(f"  Werewolf Deception Rate: {row['werewolf_deception_rate']*100:.2f}%\n\n")
    
    print(f"Report generated: {report_file}")
    return report_file

def create_visualizations(df):
    """Create visualizations from the metrics data."""
    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Generate timestamp for visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Win distribution pie chart
    plt.figure(figsize=(10, 6))
    win_counts = df['winner'].value_counts()
    plt.pie(win_counts, labels=win_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Win Distribution')
    plt.savefig(f"visualizations/win_distribution_{timestamp}.png")
    
    # 2. Performance metrics bar chart
    plt.figure(figsize=(12, 8))
    metrics = ['seer_accuracy', 'voting_accuracy', 'suspicion_change_rate', 
               'vote_discussion_alignment', 'statement_variety_rate', 'werewolf_deception_rate']
    metric_names = ['Seer Accuracy', 'Voting Accuracy', 'Suspicion Change Rate', 
                   'Vote Discussion Alignment', 'Statement Variety Rate', 'Werewolf Deception Rate']
    
    values = [df[metric].mean() * 100 for metric in metrics]
    bars = plt.bar(metric_names, values)
    plt.ylim(0, 100)
    plt.ylabel('Percentage (%)')
    plt.title('Average Performance Metrics')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"visualizations/performance_metrics_{timestamp}.png")
    
    # 3. Rounds per game bar chart
    plt.figure(figsize=(10, 6))
    rounds_counts = df['rounds_played'].value_counts().sort_index()
    plt.bar(rounds_counts.index, rounds_counts.values)
    plt.xlabel('Number of Rounds')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Game Length')
    plt.xticks(rounds_counts.index)
    plt.tight_layout()
    plt.savefig(f"visualizations/game_length_{timestamp}.png")
    
    print(f"Visualizations saved to visualizations/ directory")
    return timestamp

def main():
    print("Analyzing Werewolf game metrics...")
    metrics_data = load_metrics_files()
    
    if not metrics_data:
        print("No metrics files found. Please ensure game_metrics_*.json files are in the current directory.")
        return
    
    summary, df = create_summary_statistics(metrics_data)
    report_file = generate_report(summary, df)
    viz_timestamp = create_visualizations(df)
    
    print("\nAnalysis complete!")
    print(f"Report saved to: {report_file}")
    print(f"Visualizations saved with timestamp: {viz_timestamp}")

if __name__ == "__main__":
    main() 