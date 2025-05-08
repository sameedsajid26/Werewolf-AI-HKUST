# Werewolf Game Simulation

An AI-powered simulation of the classic Werewolf social deduction game, using Azure OpenAI to model player behavior and decision-making.

## Overview

This project simulates a Werewolf game where AI players take on different roles (Werewolf, Villager, Seer, Medic) and interact through night and day phases. The simulation tracks various metrics and generates detailed analysis of gameplay patterns.

## Game Components

- **Players**: 7-8 players with roles (2 Werewolves, 1 Seer, 1 Medic, 3-4 Villagers)
- **Phases**:
  - Night Phase: Werewolves choose victims, Seer investigates, Medic protects
  - Day Phase: Players discuss and vote to eliminate suspects
- **Discussion Rounds**: 2 rounds of discussion per day phase

## Key Files

- `game_optimized_log.py`: Main game implementation with detailed logging
- `game_optimized_2.py`: Enhanced version with improved player strategies
- `game_2_rounds.py`: Simplified version with fixed 2 discussion rounds
- `analyze_metrics.py`: Analyzes game metrics and generates reports
- `model_comparison.py`: Compares performance across different AI models
- `plots_seer.py`: Generates visualizations for game analysis

## Metrics Tracked

- Seer accuracy and investigation patterns
- Voting accuracy and alignment with discussions
- Werewolf deception rates
- Player activity levels
- Role reveal patterns
- Village consensus rates

## Setup

1. Create a `.env` file with Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_DEPLOYMENT=your_deployment
```

2. Install dependencies:
```bash
pip install openai python-dotenv pandas matplotlib seaborn
```

## Running the Game

```bash
python game_optimized_log.py
```

## Analysis

To analyze game results:
```bash
python analyze_metrics.py
```

This will generate:
- Summary statistics
- Performance metrics
- Visualizations in the `plots_model_comp` directory

## Output

The game generates:
- Game logs in JSON format
- Metrics files for analysis
- Visualization plots
- Detailed reports of player interactions and decisions

## Features

- AI-driven player decision making
- Role-based strategies
- Detailed game logging
- Comprehensive metrics tracking
- Visualization tools
- Model comparison capabilities
