import json
import random
import os
from datetime import datetime
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import sys

print("New Game Summary started...")

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI configuration from environment variables
api_key = os.getenv('AZURE_OPENAI_KEY')
api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
api_version = "2024-06-01"
deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT')

if not all([api_key, api_base, deployment_name]):
    print("Error: Missing required environment variables")
    print("Please ensure your .env file contains:")
    print("AZURE_OPENAI_ENDPOINT")
    print("AZURE_OPENAI_KEY")
    print("AZURE_OPENAI_DEPLOYMENT")
    sys.exit(1)

# Configuration with 7 players and 2 discussion rounds
CONFIG = {
    "players": [
        {"name": "Player1", "role": "Werewolf"},
        {"name": "Player2", "role": "Werewolf"},
        {"name": "Player3", "role": "Villager"},
        {"name": "Player4", "role": "Villager"},
        {"name": "Player5", "role": "Seer"},
        {"name": "Player6", "role": "Medic"},
        {"name": "Player7", "role": "Villager"},
        {"name": "Player8", "role": "Villager"},
        {"name": "Moderator", "role": "Moderator"}
    ],
    "azure_openai": {
        "endpoint": api_base,
        "api_key": api_key,
        "deployment_name": deployment_name,
        "api_version": api_version
    },
    "discussion_rounds": 2  # Fixed to 2 discussion rounds per day phase
}

class Player:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.is_alive = True
        self.last_protected = None
        self.knowledge = []  # All players track suspicions or Seer investigations
        self.suspicion_changes = []  # Track suspicion updates
        self.statements = []  # Track statements made by the player
        self.votes = []  # Track all votes cast by this player
        self.voted_for = []  # Track who voted for this player
        self.activity_level = 0  # Track how active a player is in discussions
        self.role_claims = []  # Track any role claims made
        if self.role == "Seer":
            self.knowledge = []  # List of (player_name, "Werewolf" or "Not a Werewolf")
        else:
            self.knowledge = []  # List of (player_name, suspicion_level) where suspicion_level is 0-1

    def __str__(self):
        return f"{self.name} ({self.role}, {'Alive' if self.is_alive else 'Dead'})"

class GameLogger:
    def __init__(self, game_id: str):
        # Create game-specific directory for all logs
        self.game_dir = f"game_logs_{game_id}"
        os.makedirs(self.game_dir, exist_ok=True)
        
        # Set up log files
        self.log_file = os.path.join(self.game_dir, "game_events.json")
        self.discussion_file = os.path.join(self.game_dir, "discussions.json")
        self.prompts_file = os.path.join(self.game_dir, "prompts.json")
        self.metrics_file = os.path.join(self.game_dir, "metrics.json")
        self.voting_file = os.path.join(self.game_dir, "voting_history.json")
        
        # Initialize log containers
        self.logs = []
        self.discussions = []
        self.prompts = []
        self.voting_history = []

    def log_event(self, event_type: str, data: Dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.logs.append(log_entry)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def log_discussion(self, event_type: str, data: Dict):
        disc_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.discussions.append(disc_entry)
        with open(self.discussion_file, 'w') as f:
            json.dump(self.discussions, f, indent=2)
            
    def log_prompts(self, event_type: str, data: Dict):
        prompt_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.prompts.append(prompt_entry)
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)
    
    def log_votes(self, round_num: int, votes: Dict):
        vote_entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            "votes": votes
        }
        self.voting_history.append(vote_entry)
        with open(self.voting_file, 'w') as f:
            json.dump(self.voting_history, f, indent=2)

class WerewolfGame:
    def __init__(self, players: List[Dict], azure_config: Dict, discussion_rounds: int, randomize_roles=True):
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        player_list = [p for p in players if p["role"] != "Moderator"]
        
        # Randomize roles if requested
        if randomize_roles:
            # Extract player names and roles
            names = [p["name"] for p in player_list]
            roles = [p["role"] for p in player_list]
            
            # Shuffle roles
            random.shuffle(roles)
            
            # Assign shuffled roles to player names
            self.players = [Player(names[i], roles[i]) for i in range(len(names))]
        else:
            # Use predefined roles
            self.players = [Player(p["name"], p["role"]) for p in player_list]
        
        self.moderator = Player("Moderator", "Moderator")
        self.logger = GameLogger(self.game_id)
        self.round = 0
        self.game_history = []
        self.voting_history = []  # Enhanced tracking of all votes across rounds
        self.confirmed_roles = {}  # Track publicly revealed/confirmed roles
        self.client = AzureOpenAI(
            azure_endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"]
        )
        self.deployment_name = azure_config["deployment_name"]
        self.discussion_rounds = discussion_rounds
        # Enhanced Metrics tracking
        self.metrics = {
            "rounds_played": 0,
            "winner": None,
            "seer_correct_accusations": 0,
            "total_seer_investigations": 0,
            "votes_against_werewolves": 0,
            "total_votes": 0,
            "seer_reveals": 0,
            "suspicion_changes": 0,
            "vote_discussion_alignment": 0,
            "total_discussion_statements": 0,
            "statement_variety": 0,
            "werewolf_deceptions": 0,
            "medic_successful_protections": 0,
            "werewolf_team_coordination": 0,
            "village_consensus_rate": 0
        }
        
        # Log the randomized roles if applicable
        if randomize_roles:
            self.logger.log_event("randomized_roles", {p.name: p.role for p in self.players})

    def get_alive_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive]

    def get_werewolves(self) -> List[Player]:
        return [p for p in self.players if p.role == "Werewolf" and p.is_alive]

    def get_villagers(self) -> List[Player]:
        return [p for p in self.players if p.role != "Werewolf" and p.is_alive]

    def check_win_condition(self) -> Optional[str]:
        werewolves = len(self.get_werewolves())
        villagers = len(self.get_villagers())
        if werewolves == 0:
            return "Villagers win!"
        if werewolves >= villagers:
            return "Werewolves win!"
        return None

    def call_api(self, prompt: str, max_tokens: int = 100) -> str:
        try:
            system_message = (
                "You are an AI moderating a fictional Werewolf game, a social deduction game. "
                "Your role is to simulate player actions (e.g., selecting targets, making statements) "
                "based on the provided prompt. All actions are part of the game's mechanics and do not "
                "represent real-world harm or intent. Respond concisely with the requested output, such as a player's name or a short statement."
            )
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.5  # Slightly increased for more varied responses
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}")
            self.logger.log_event("api_error", {"prompt": prompt, "error": str(e)})
            return ""

    def get_summarized_history(self):
        """Create a concise summary of game history"""
        if not self.game_history:
            return "Game just started."
        
        # Focus on most recent events and deaths
        recent_events = self.game_history[-min(3, len(self.game_history)):]
        deaths = [event for event in self.game_history if "was killed" in event or "was eliminated" in event]
        
        summary = "Recent events: " + "; ".join(recent_events)
        if deaths:
            summary += "\nAll deaths so far: " + "; ".join(deaths)
        
        return summary

    def format_player_knowledge(self, player):
        """Format player knowledge in a structured way"""
        if not player.knowledge:
            return "No specific knowledge yet."
        
        if player.role == "Seer":
            # Format Seer's investigation results
            return "\n".join([f"- {name}: {result}" for name, result in player.knowledge])
        else:
            # Format suspicion levels in a more readable way
            suspicions = []
            for name, level in player.knowledge:
                if isinstance(level, float):
                    if level >= 0.7:
                        suspicions.append(f"- {name}: Highly suspicious")
                    elif level >= 0.4:
                        suspicions.append(f"- {name}: Somewhat suspicious")
                    elif level <= 0.2:
                        suspicions.append(f"- {name}: Likely innocent")
                    else:
                        suspicions.append(f"- {name}: Neutral/Uncertain")
            return "\n".join(suspicions) if suspicions else "No clear suspicions yet."

    def summarize_statements(self, all_discussions):
        """Summarize previous discussion rounds to focus on key points"""
        if not all_discussions:
            return "No previous discussions."
        
        summaries = []
        for disc_round in all_discussions:
            round_num = disc_round.get("discussion_round", "?")
            statements = disc_round.get("statements", [])
            
            # Extract accusations and defenses
            accusations = []
            defenses = []
            for stmt in statements:
                player = stmt.get("player", "Unknown")
                statement = stmt.get("statement", "")
                
                if "suspect" in statement.lower() or "accuse" in statement.lower():
                    accusations.append(f"{player} accused/suspected someone")
                if "innocent" in statement.lower() or "defend" in statement.lower():
                    defenses.append(f"{player} defended someone")
            
            round_summary = f"Round {round_num}: "
            if accusations:
                round_summary += f"{len(accusations)} accusations; "
            if defenses:
                round_summary += f"{len(defenses)} defenses"
            
            summaries.append(round_summary)
        
        return "\n".join(summaries)

    def extract_key_accusations(self, all_discussions):
        """Extract the most significant accusations and defenses from all discussions"""
        if not all_discussions:
            return "No discussions yet."
        
        # Flatten all statements across rounds
        all_statements = []
        for disc_round in all_discussions:
            for stmt in disc_round.get("statements", []):
                all_statements.append(stmt)
        
        # Extract statements that mention players
        player_mentions = {}
        for stmt in all_statements:
            speaker = stmt.get("player", "Unknown")
            statement = stmt.get("statement", "")
            
            for player in self.get_alive_players():
                if player.name in statement and player.name != speaker:
                    if player.name not in player_mentions:
                        player_mentions[player.name] = []
                    player_mentions[player.name].append(f"{speaker}: {statement}")
        
        # Format the most significant mentions (limit to top 2 per player)
        key_mentions = []
        for player_name, mentions in player_mentions.items():
            if mentions:
                key_mentions.append(f"About {player_name}:")
                for mention in mentions[:2]:  # Limit to 2 mentions per player
                    key_mentions.append(f"- {mention}")
        
        return "\n".join(key_mentions) if key_mentions else "No significant accusations or defenses yet."

    def format_voting_history(self):
        """Format the voting history in a readable way"""
        if not self.voting_history:
            return "No voting history yet."
        
        voting_summary = []
        for round_votes in self.voting_history:
            round_num = round_votes.get("round", "?")
            votes = round_votes.get("votes", {})
            
            vote_summary = [f"Round {round_num} votes:"]
            for voter, vote in votes.items():
                vote_summary.append(f"- {voter} voted for {vote}")
            
            voting_summary.append("\n".join(vote_summary))
        
        return "\n\n".join(voting_summary)
    
    def analyze_player_activity(self, player, all_discussions):
        """Analyze how active/vocal a player has been"""
        if not all_discussions:
            return 0
        
        # Count meaningful statements (more than just a few words)
        statement_count = 0
        total_words = 0
        
        for disc_round in all_discussions:
            # Handle both dictionary and list formats
            if isinstance(disc_round, dict):
                statements = disc_round.get("statements", [])
            else:
                statements = disc_round
            
            for stmt in statements:
                if stmt.get("player") == player.name:
                    words = stmt.get("statement", "").split()
                    if len(words) > 3:  # Consider meaningful if more than 3 words
                        statement_count += 1
                        total_words += len(words)
        
        # Calculate activity score (0-10 scale)
        if statement_count == 0:
            return 0
        
        avg_words = total_words / statement_count
        activity_score = min(statement_count * (avg_words / 5), 10)  # Scale based on count and length
        
        return round(activity_score, 1)
    
    def get_role_strategy(self, player, discussion_round):
        """Provide role-specific strategy guidance based on game state"""
        alive_players = self.get_alive_players()
        game_stage = "early" if self.round <= 2 else "mid" if self.round <= 4 else "late"
        
        if player.role == "Werewolf":
            strategies = [
                "Try to blend in by accusing other players without drawing attention to yourself",
                "Defend your fellow werewolves subtly, but don't make it obvious",
                "Consider fake-claiming a role if pressured (but be careful, as this is risky)",
                "Try to create confusion by casting doubt on vocal players"
            ]
            
            # Enhanced werewolf strategy based on game stage
            if game_stage == "early":
                strategies.append("In early rounds, observe before making strong accusations")
            elif game_stage == "mid":
                strategies.append("Build on previous discussions to seem consistent")
                strategies.append("If a player is strongly suspected by villagers, join in to blend in")
            else:  # late game
                strategies.append("Coordinate with other werewolves to secure eliminations")
                strategies.append("Target confirmed villagers or suspected power roles")
                
            if len(self.get_werewolves()) < len(alive_players) // 3:
                strategies.append("Your team is outnumbered - focus on survival rather than aggression")
                
            return "\n".join([f"- {s}" for s in strategies[:3]])  # Limit to 3 strategies
            
        elif player.role == "Seer":
            strategies = [
                "Use your knowledge strategically without revealing your role too early",
                "If you've found a werewolf, consider carefully when to reveal this information",
                "If pressured, you might need to reveal your role to save yourself or confirm information",
                "Pay attention to contradictions in player statements compared to your knowledge"
            ]
            
            # Enhanced seer strategy based on game stage
            werewolf_found = any(result == "Werewolf" for _, result in player.knowledge)
            if werewolf_found:
                if game_stage == "early":
                    strategies.append("You've identified a werewolf early - consider waiting before revealing")
                else:
                    strategies.append("You've identified a werewolf - consider revealing this to unite the village")
            
            # New strategy for late game
            if game_stage == "late":
                strategies.append("In late game, revealing your role may be necessary to save the village")
                
            return "\n".join([f"- {s}" for s in strategies[:3]])  # Limit to 3 strategies
            
        elif player.role == "Medic":
            strategies = [
                "Keep your role secret to avoid being targeted by werewolves",
                "Pay attention to discussions to identify potential Seers to protect",
                "Vary your protection targets to be unpredictable",
                "Consider protecting players who are under suspicion but you believe are innocent"
            ]
            
            # Enhanced medic strategy based on game stage
            if game_stage == "mid" or game_stage == "late":
                strategies.append("Prioritize protecting players who seem to have important information")
                strategies.append("Don't waste protection on inactive players")
            
            if game_stage == "late":
                strategies.append("In late game, consider self-protection if you're being targeted")
                
            return "\n".join([f"- {s}" for s in strategies[:3]])  # Limit to 3 strategies
            
        else:  # Villager
            strategies = [
                "Analyze player statements carefully for inconsistencies",
                "Be careful about who you trust, but work with others to identify werewolves",
                "Don't reveal too much about your suspicions too early",
                "Pay attention to voting patterns from previous rounds"
            ]
            
            # Enhanced villager strategy based on game stage
            if game_stage == "mid" or game_stage == "late":
                strategies.append("Be suspicious of quiet players who don't contribute")
                strategies.append("Look for patterns in voting - werewolves often protect each other")
            
            if game_stage == "late":
                strategies.append("In late game, consensus is crucial - try to align with other trusted villagers")
                
            return "\n".join([f"- {s}" for s in strategies[:3]])  # Limit to 3 strategies

    def get_voting_strategy(self, player):
        """Provide voting strategy guidance based on role and game state"""
        game_stage = "early" if self.round <= 2 else "mid" if self.round <= 4 else "late"
        
        if player.role == "Werewolf":
            werewolves = self.get_werewolves()
            if game_stage == "late" and len(werewolves) > 1:
                return (
                    "Coordinate votes with your werewolf teammates to eliminate key villagers. "
                    "Target confirmed or suspected Seers or Medics first. "
                    "Avoid voting for fellow werewolves at all costs. "
                    "If there's a clear village consensus against someone, consider joining it to blend in."
                )
            else:
                return (
                    "Vote strategically to eliminate villagers, especially those who might be the Seer or Medic. "
                    "Avoid voting for your fellow werewolves. Consider voting for players who are suspicious of you "
                    "or your teammates. Try to align your vote with village consensus if possible."
                )
                
        elif player.role == "Seer":
            werewolf_found = any(result == "Werewolf" for _, result in player.knowledge)
            if werewolf_found and game_stage != "early":
                return (
                    "Vote for a confirmed werewolf from your investigations. If you need to reveal your role "
                    "to convince others, do so strategically. Your vote carries important information."
                )
            else:
                return (
                    "Use your investigation results to guide your vote. Prioritize voting for confirmed werewolves. "
                    "If you haven't found a werewolf yet, vote based on suspicious behavior. "
                    "Consider the consequences of revealing your knowledge through your vote."
                )
                
        elif player.role == "Medic":
            if game_stage == "late":
                return (
                    "Vote strategically based on all available information. Prioritize eliminating confirmed "
                    "or strongly suspected werewolves. Be suspicious of quiet players or those with inconsistent statements. "
                    "Your survival is important for the village."
                )
            else:
                return (
                    "Vote based on observed behavior and discussion patterns. Try to identify werewolves through "
                    "their inconsistencies or suspicious defenses. Be wary of players who seem to be working together."
                )
                
        else:  # Villager
            if game_stage == "late":
                return (
                    "In this critical stage, voting consensus is crucial. Look at voting history to identify patterns. "
                    "Be suspicious of quiet players who haven't contributed meaningfully. "
                    "Trust players who have consistently voted against confirmed werewolves."
                )
            else:
                return (
                    "Vote based on the evidence from discussions. Look for inconsistencies in statements. "
                    "Consider who made the most logical arguments. Be cautious of players making vague accusations."
                )

    def identify_key_targets(self, player_role):
        """Identify key targets based on role and game state"""
        alive_players = self.get_alive_players()
        
        if player_role == "Werewolf":
            # Werewolves should target Seers, Medics, or vocal villagers
            # First, look for confirmed or suspected power roles
            targets = []
            
            # Find anyone who's claimed to be Seer or Medic
            for player in alive_players:
                if player.role != "Werewolf" and any("I am the Seer" in stmt or "I am the Medic" in stmt for stmt in player.statements):
                    targets.append((player.name, 1.0))  # High priority
            
            # Find vocal players with high activity (likely influential)
            for player in alive_players:
                if player.role != "Werewolf" and player.activity_level >= 7:
                    targets.append((player.name, 0.8))  # High priority
            
            # If no clear targets, find players who are suspicious of werewolves
            if not targets:
                for player in alive_players:
                    if player.role != "Werewolf":
                        for werewolf in self.get_werewolves():
                            if any(werewolf.name in stmt and "suspect" in stmt for stmt in player.statements):
                                targets.append((player.name, 0.7))  # Medium-high priority
            
            # If still no clear targets, target random villagers
            if not targets:
                for player in alive_players:
                    if player.role != "Werewolf":
                        targets.append((player.name, 0.5))  # Medium priority
            
            return targets
            
        elif player_role == "Medic":
            # Medics should protect Seers, vocal villagers, or themselves if targeted
            targets = []
            
            # First, find confirmed or suspected Seers
            for player in alive_players:
                if player.role != "Werewolf" and any("I am the Seer" in stmt for stmt in player.statements):
                    targets.append((player.name, 1.0))  # High priority
            
            # Next, find vocal players who might be targeted
            for player in alive_players:
                if player.role != "Werewolf" and player.activity_level >= 7:
                    targets.append((player.name, 0.8))  # High priority
            
            # Find players who are suspected by others (might be targeted)
            for player in alive_players:
                if player.role != "Werewolf":
                    suspicion_count = 0
                    for other in alive_players:
                        if other != player and any(player.name in stmt and "suspect" in stmt for stmt in other.statements):
                            suspicion_count += 1
                    if suspicion_count >= 2:  # If multiple players suspect them
                        targets.append((player.name, 0.7))  # Medium-high priority
            
            # Self-protection in late game if there's risk
            if self.round >= 3:
                targets.append((player.name, 0.6))  # Medium priority
            
            return targets
        
        return []  # Default empty list for other roles

    def print_round_summary(self):
        """Print a summary of the current game state after each round"""
        werewolves = self.get_werewolves()
        villagers = self.get_villagers()
        print(f"\n=== Round {self.round} Summary ===")
        print(f"Werewolves alive: {len(werewolves)} ({', '.join([p.name for p in werewolves])})")
        print(f"Villagers alive: {len(villagers)} ({', '.join([p.name for p in villagers])})")
        print("================================\n")

    def night_phase(self):
        self.round += 1
        self.metrics["rounds_played"] = self.round
        self.logger.log_event("night_start", {"round": self.round})
        alive_players = self.get_alive_players()
        player_names = [p.name for p in alive_players]
        game_stage = "early" if self.round <= 2 else "mid" if self.round <= 4 else "late"

        # Update player activity levels
        for player in alive_players:
            # Extract discussions from the logger's discussion entries
            discussions_data = []
            for disc in self.logger.discussions:
                if "data" in disc and "discussions" in disc["data"]:
                    discussions_data.extend(disc["data"]["discussions"])
            
            player.activity_level = self.analyze_player_activity(player, discussions_data)

        # Werewolf selection with enhanced strategic targeting
        werewolves = self.get_werewolves()
        if werewolves:
            werewolf_names = [p.name for p in werewolves]
            key_targets = self.identify_key_targets("Werewolf")
            
            # Format targets for the prompt
            target_info = "\n".join([f"- {name}: {'High priority' if score >= 0.8 else 'Medium priority' if score >= 0.6 else 'Low priority'}" 
                                   for name, score in key_targets])
            
            # Enhanced werewolf prompt with strategic guidance and target analysis
            prompt = (
                f"In the fictional Werewolf game, you are a Werewolf. Your team: {werewolf_names}. "
                f"Alive players: {player_names}.\n"
                f"Game history summary: {self.get_summarized_history()}\n"
                f"Game stage: {game_stage} game (Round {self.round})\n"
                f"Target analysis:\n{target_info}\n"
                f"Strategic considerations:\n"
                f"- Target influential players who might be the Seer or Medic\n"
                f"- Avoid targeting players who were protected previously\n"
                f"- Focus on eliminating vocal players who are suspicious of werewolves\n"
                f"- In late game, prioritize eliminating confirmed villagers\n"
                f"Select one player as the target for this round. Respond with only the player's name."
            )
            victim_name = self.call_api(prompt)
            victim = next((p for p in alive_players if p.name == victim_name), None)
            if not victim or victim.role == "Werewolf":  # Ensure werewolves don't kill their own
                # Find the highest priority non-werewolf target
                valid_targets = [(name, score) for name, score in key_targets 
                               if next((p for p in alive_players if p.name == name), None) and 
                               next((p for p in alive_players if p.name == name), None).role != "Werewolf"]
                if valid_targets:
                    valid_targets.sort(key=lambda x: x[1], reverse=True)  # Sort by priority
                    victim = next((p for p in alive_players if p.name == valid_targets[0][0]), None)
                else:
                    village_targets = [p for p in alive_players if p.role != "Werewolf"]
                    if village_targets:
                        victim = random.choice(village_targets)  # Fallback
                    else:
                        victim = None
            if victim:
                self.logger.log_event("werewolf_choice", {"victim": victim.name, "reasoning": prompt})
        else:
            victim = None

        # Seer investigation with enhanced strategic targeting
        seer = next((p for p in alive_players if p.role == "Seer"), None)
        if seer:
            valid_targets = [p for p in alive_players if p.name != seer.name]  # Exclude self
            valid_names = [p.name for p in valid_targets]
            
            # Already investigated players
            investigated = [name for name, _ in seer.knowledge]
            uninvestigated = [p.name for p in valid_targets if p.name not in investigated]
            
            # Analyze player behavior to prioritize suspicious players
            suspicious_players = []
            for player in valid_targets:
                if player.name not in investigated:
                    suspicion_score = 0
                    # Suspicious behavior: defending suspected players, inconsistent statements, voting patterns
                    for stmt in player.statements:
                        # Check for defensive statements of suspected players
                        for other in valid_targets:
                            if other.name in stmt and "defend" in stmt.lower() and any(other.name in s and "suspect" in s for s in [p.statements for p in valid_targets if p != player and p != other]):
                                suspicion_score += 0.2
                    
                    # Check voting patterns
                    if player.votes:
                        for vote in player.votes:
                            voted_player = next((p for p in self.players if p.name == vote), None)
                            if voted_player and voted_player.role != "Werewolf":
                                suspicion_score += 0.1  # Suspicious if consistently voting against villagers
                    
                    if suspicion_score > 0:
                        suspicious_players.append((player.name, suspicion_score))
            
            suspicious_players.sort(key=lambda x: x[1], reverse=True)
            
            if valid_names:
                # Enhanced seer prompt with strategic guidance and suspicion analysis
                prompt = (
                    f"In the fictional Werewolf game, you are the Seer. Alive players (excluding yourself): {valid_names}.\n"
                    f"Game history summary: {self.get_summarized_history()}\n"
                    f"Your previous investigations: {seer.knowledge}\n"
                    f"Players you haven't investigated yet: {uninvestigated}\n"
                    f"Suspicious players based on behavior:\n"
                    f"{', '.join([name for name, _ in suspicious_players[:3]]) if suspicious_players else 'None identified'}\n"
                    f"Strategic considerations:\n"
                    f"- Prioritize investigating players who show suspicious behavior\n"
                    f"- In early game, focus on gathering information on multiple players\n"
                    f"- In late game, focus on confirming your suspicions\n"
                    f"- Balance between checking new players and verifying suspicions\n"
                    f"Select one player to investigate their role. Respond with only the player's name."
                )
                target_name = self.call_api(prompt)
                target = next((p for p in valid_targets if p.name == target_name), None)
                if not target:
                    # Prioritize uninvestigated players if available
                    if uninvestigated:
                        target = next((p for p in valid_targets if p.name in uninvestigated), None)
                    elif suspicious_players:
                        target = next((p for p in valid_targets if p.name == suspicious_players[0][0]), None)
                    else:
                        target = random.choice(valid_targets)  # Fallback
                result = "Werewolf" if target.role == "Werewolf" else "Not a Werewolf"
                seer.knowledge.append((target.name, result))
                self.metrics["total_seer_investigations"] += 1
                self.logger.log_event("seer_investigation", {"seer": seer.name, "target": target.name, "result": result, "reasoning": prompt})
            else:
                self.logger.log_event("seer_investigation", {"seer": seer.name, "error": "No valid targets"})

        # Medic protection with enhanced strategic targeting
        medic = next((p for p in alive_players if p.role == "Medic"), None)
        if medic:
            key_targets = self.identify_key_targets("Medic")
            valid_targets = [p for p in alive_players if p.name != medic.last_protected]
            valid_names = [p.name for p in valid_targets]
            
            # Format targets for the prompt
            target_info = "\n".join([f"- {name}: {'High priority' if score >= 0.8 else 'Medium priority' if score >= 0.6 else 'Low priority'}" 
                                   for name, score in key_targets if name in valid_names])
            
            # Enhanced medic prompt with strategic guidance and target analysis
            prompt = (
                f"In the fictional Werewolf game, you are the Medic. Alive players: {valid_names}.\n"
                f"Game history summary: {self.get_summarized_history()}\n"
                f"You cannot protect {medic.last_protected or 'none'} again this round.\n"
                f"Game stage: {game_stage} game (Round {self.round})\n"
                f"Target analysis:\n{target_info}\n"
                f"Strategic considerations:\n"
                f"- Protect players who might be the Seer or other key roles\n"
                f"- Prioritize vocal players who are likely targets for werewolves\n"
                f"- In late game, don't waste protection on inactive players\n"
                f"- Consider self-protection if you're at risk\n"
                f"Select one player to protect this round. Respond with only the player's name."
            )
            protected_name = self.call_api(prompt)
            protected = next((p for p in valid_targets if p.name == protected_name), None)
            if not protected:
                # Select from high priority targets first
                high_priority = [(name, score) for name, score in key_targets if score >= 0.8 and name in valid_names]
                if high_priority:
                    protected_name = high_priority[0][0]
                    protected = next((p for p in valid_targets if p.name == protected_name), None)
                else:
                    protected = random.choice(valid_targets)  # Fallback
            
            if protected:
                medic.last_protected = protected.name
                self.logger.log_event("medic_protection", {"medic": medic.name, "protected": protected.name, "reasoning": prompt})
            else:
                protected = None
        else:
            protected = None

        # Resolve night actions and update game history
        if victim and protected and victim.name == protected.name:
            self.logger.log_event("night_result", {"victim": victim.name, "saved": True})
            self.game_history.append(f"Night {self.round}: No one was killed (Medic saved someone)")
            self.metrics["medic_successful_protections"] += 1
        elif victim:
            victim.is_alive = False
            self.logger.log_event("night_result", {"victim": victim.name, "saved": False})
            self.game_history.append(f"Night {self.round}: {victim.name} was killed")
        else:
            self.game_history.append(f"Night {self.round}: No one was killed")
        
        # Print round summary after night phase
        self.print_round_summary()

    def day_phase(self):
        self.logger.log_event("day_start", {"round": self.round})
        alive_players = self.get_alive_players()
        player_names = [p.name for p in alive_players]
        game_summary = self.get_summarized_history()
        game_stage = "early" if self.round <= 2 else "mid" if self.round <= 4 else "late"
        all_discussions = []
        all_prompts = []

        # Two Discussion Rounds
        for discussion_round in range(1, self.discussion_rounds + 1):
            discussion = []
            prompts = []
            # Create a summary of previous statements for this round
            previous_statements_summary = self.summarize_statements(all_discussions) if all_discussions else "No previous discussion yet."
            
            # Get voting history summary
            voting_history = self.format_voting_history()
            
            random.shuffle(alive_players)  # Randomize speaking order
            for player in alive_players:
                # Get a concise summary of previous statements in this round
                current_round_statements = "\n".join([f"{p['player']}: {p['statement']}" for p in discussion]) if discussion else "No statements yet."
                
                # Get player-specific knowledge in a more structured format
                knowledge_str = self.format_player_knowledge(player)
                
                # Create role-specific strategic guidance
                strategy_guidance = self.get_role_strategy(player, discussion_round)
                
                # Adjust role information based on role and game state
                if player.role == "Werewolf":
                    werewolves = self.get_werewolves()
                    teammates = [p.name for p in werewolves if p != player]
                    role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}."
                    
                    # Enhanced werewolf discussion strategy for late game
                    if game_stage == "late":
                        coordination_hint = (
                            f"In this late stage, coordinate subtly with your teammates. "
                            f"Focus on discrediting players who might be the Seer or Medic. "
                            f"If there's a consensus building against a villager, support it to blend in."
                        )
                        strategy_guidance += f"\n{coordination_hint}"
                elif player.role == "Seer":
                    role_info = f"You are the Seer. Your investigations: {knowledge_str}."
                    
                    # Adjusted Seer strategy based on findings and game stage
                    werewolf_found = any(result == "Werewolf" for _, result in player.knowledge)
                    if werewolf_found and game_stage != "early":
                        reveal_hint = (
                            f"You have found a werewolf. Consider revealing your role strategically "
                            f"to convince others. In late game, this information is crucial for the village."
                        )
                        strategy_guidance += f"\n{reveal_hint}"
                elif player.role == "Medic":
                    role_info = f"You are the Medic."
                    
                    # Keep track of who you've protected
                    if player.last_protected:
                        role_info += f" Last night you protected {player.last_protected}."
                    
                    # Medic strategy adjustment
                    if game_stage == "late":
                        medic_hint = (
                            f"In late game, your survival is critical. Be careful about revealing your role, "
                            f"but focus on identifying werewolves through voting patterns and behavior."
                        )
                        strategy_guidance += f"\n{medic_hint}"
                else:
                    role_info = f"You are a {player.role}."
                    
                    # Villager strategy adjustment for late game
                    if game_stage == "late":
                        villager_hint = (
                            f"In late game, be suspicious of quiet players who haven't contributed meaningfully. "
                            f"Look for voting patterns that suggest werewolf coordination."
                        )
                        strategy_guidance += f"\n{villager_hint}"

                prompt = (
                    f"In the fictional Werewolf game, {role_info}\n"
                    f"Game summary: {game_summary}\n"
                    f"Alive players: {', '.join(player_names)}\n"
                    f"Game stage: {game_stage} game (Round {self.round})\n"
                    f"Discussion round {discussion_round} of {self.discussion_rounds}\n"
                    f"Previous discussion rounds summary: {previous_statements_summary}\n"
                    f"Current round statements: {current_round_statements}\n"
                    f"Your knowledge: {knowledge_str}\n"
                    f"Voting history: {voting_history}\n"
                    f"Strategic guidance: {strategy_guidance}\n"
                    f"Now, as {player.name}, make a strategic statement about who you suspect or defend. "
                    f"Your statement should directly advance your win condition while appearing logical to others. "
                    f"Be specific with your reasoning and avoid vague statements. "
                    f"Respond with only your in-character statement (1-2 sentences)."
                )
                statement = self.call_api(prompt, max_tokens=100)  # Increased token limit for more substantive statements
                discussion.append({"player": player.name, "statement": statement})
                prompts.append({"player": player.name, "prompt": prompt})
                player.statements.append(f"Round {discussion_round}: {statement}")
                self.metrics["total_discussion_statements"] += 1
                
                # Analyze statement for role claims
                if "I am the Seer" in statement or "as the Seer" in statement:
                    player.role_claims.append(("Seer", self.round, discussion_round))
                    self.confirmed_roles[player.name] = "Claimed Seer"
                elif "I am the Medic" in statement or "as the Medic" in statement:
                    player.role_claims.append(("Medic", self.round, discussion_round))
                    self.confirmed_roles[player.name] = "Claimed Medic"

                # Check for Seer reveal
                if player.role == "Seer" and "I am the Seer" in statement:
                    self.metrics["seer_reveals"] += 1

                # Check for Werewolf deception
                if player.role == "Werewolf" and any(p.name in statement for p in self.get_villagers()):
                    self.metrics["werewolf_deceptions"] += 1

                # Update suspicions based on statement
                for p in alive_players:
                    if p.name != player.name and p.name in statement:
                        existing_suspicion = next((level for name, level in player.knowledge if name == p.name and isinstance(level, float)), 0.0)
                        
                        # Enhanced suspicion modeling
                        suspicion_change = 0
                        if "suspect" in statement.lower() or "accuse" in statement.lower():
                            if p.name in statement.lower():
                                suspicion_change = 0.25  # Increased suspicion for direct accusations
                        elif "innocent" in statement.lower() or "defend" in statement.lower():
                            if p.name in statement.lower():
                                suspicion_change = -0.2  # Decreased suspicion for defense
                        
                        # Adjust based on role claims and confirmed information
                        if player.role == "Seer" and any(p.name in res and res[1] == "Werewolf" for res in player.knowledge):
                            suspicion_change = 0.5  # Significant increase for Seer accusation
                            
                        new_suspicion = max(0.0, min(1.0, existing_suspicion + suspicion_change))
                        
                        if existing_suspicion != new_suspicion:
                            player.knowledge = [(name, level) for name, level in player.knowledge if name != p.name]
                            player.knowledge.append((p.name, new_suspicion))
                            player.suspicion_changes.append({"round": self.round, "discussion_round": discussion_round, "target": p.name, "new_suspicion": new_suspicion})
                            self.metrics["suspicion_changes"] += 1

                # Track statement variety
                current_targets = [p.name for p in alive_players if p.name in statement and p.name != player.name]
                past_targets = []
                for past_stmt in player.statements[:-1]:
                    for p in alive_players:
                        if p.name in past_stmt and p.name != player.name:
                            past_targets.append(p.name)
                if current_targets and all(target not in past_targets for target in current_targets):
                    self.metrics["statement_variety"] += 1

            all_discussions.append({"discussion_round": discussion_round, "statements": discussion})
            all_prompts.append({"discussion_round": discussion_round, "prompts": prompts})

        self.logger.log_event("discussion", {"discussions": all_discussions})
        self.logger.log_discussion("discussion", {"discussions": all_discussions})
        self.logger.log_prompts("discussion", {"prompts": all_prompts})

        # Voting with enhanced prompts and strategy
        votes = {}
        vote_reasons = {}
        for player in alive_players:
            valid_targets = [p.name for p in alive_players if p.name != player.name]  # Exclude self
            knowledge_str = self.format_player_knowledge(player)
            
            # Create role-specific voting strategy
            voting_strategy = self.get_voting_strategy(player)
            
            # Create a discussion summary focused on key accusations
            discussion_summary = self.extract_key_accusations(all_discussions)
            
            # Enhanced voting context
            voting_context = self.format_voting_history()
            
            # Adjust role information based on role and game state
            if player.role == "Werewolf":
                werewolves = self.get_werewolves()
                teammates = [p.name for p in werewolves if p != player]
                
                # Check if werewolves should coordinate votes in late game
                if game_stage == "late" and len(werewolves) > 1:
                    # Look for a consensus village target
                    village_consensus = None
                    villager_accusations = {}
                    for p in self.get_villagers():
                        for stmt in p.statements:
                            for other in alive_players:
                                if other.name in stmt and "suspect" in stmt.lower() and other.role != "Werewolf":
                                    villager_accusations[other.name] = villager_accusations.get(other.name, 0) + 1
                    
                    if villager_accusations:
                        # Find the most accused villager
                        village_consensus = max(villager_accusations.items(), key=lambda x: x[1])[0]
                        
                        # If there's a strong consensus, werewolves should join it to blend in
                        if villager_accusations[village_consensus] >= len(self.get_villagers()) / 2:
                            role_info = (
                                f"You are a Werewolf. Your teammates are {', '.join(teammates)}. "
                                f"IMPORTANT: The village seems to be forming a consensus against {village_consensus}. "
                                f"Consider voting with this consensus to blend in, unless another werewolf is at risk."
                            )
                            self.metrics["werewolf_team_coordination"] += 1
                        else:
                            role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}."
                    else:
                        role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}."
                else:
                    role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}."
            elif player.role == "Seer":
                werewolf_found = any(result == "Werewolf" for _, result in player.knowledge)
                if werewolf_found:
                    werewolf_names = [name for name, result in player.knowledge if result == "Werewolf"]
                    role_info = (
                        f"You are the Seer. Your investigations have identified these werewolves: {', '.join(werewolf_names)}. "
                        f"Your vote should prioritize eliminating a confirmed werewolf."
                    )
                else:
                    role_info = f"You are the Seer. Your investigations: {knowledge_str}."
            else:
                role_info = f"You are a {player.role}."

            # Enhanced voting prompt with strategic guidance
            prompt = (
                f"In the fictional Werewolf game, {role_info}\n"
                f"Game summary: {game_summary}\n"
                f"Game stage: {game_stage} game (Round {self.round})\n"
                f"Alive players (excluding yourself): {', '.join(valid_targets)}\n"
                f"Key accusations and defenses from discussions:\n{discussion_summary}\n"
                f"Your knowledge: {knowledge_str}\n"
                f"Voting history: {voting_context}\n"
                f"Role claims in the game: {', '.join([f'{name}: {role}' for name, role in self.confirmed_roles.items()])}\n"
                f"Voting strategy: {voting_strategy}\n"
                f"Based on all information, vote for one player to eliminate, "
                f"or respond with 'Pass' if you truly cannot decide. Respond with only the player's name or 'Pass'."
            )
            vote = self.call_api(prompt)
            
            # Get the reasoning behind the vote
            reason_prompt = (
                f"You just voted for {vote} in the Werewolf game. "
                f"In 1-2 sentences, explain your strategic reasoning for this vote. "
                f"Be specific about why this target advances your win condition."
            )
            vote_reason = self.call_api(reason_prompt, max_tokens=75)
            
            if vote == "Pass" or vote not in valid_targets:
                votes[player.name] = "Pass"
                vote_reasons[player.name] = vote_reason
            else:
                votes[player.name] = vote
                vote_reasons[player.name] = vote_reason
                player.votes.append(vote)
                
                # Update player being voted for
                voted_player = next((p for p in alive_players if p.name == vote), None)
                if voted_player:
                    voted_player.voted_for.append(player.name)
                
                self.metrics["total_votes"] += 1
                if next(p for p in alive_players if p.name == vote).role == "Werewolf":
                    self.metrics["votes_against_werewolves"] += 1
                
                # Check if vote aligns with discussion
                last_suspicions = [s["target"] for s in player.suspicion_changes if s["round"] == self.round]
                if vote in last_suspicions:
                    self.metrics["vote_discussion_alignment"] += 1
                
                # Check for werewolf team coordination
                if player.role == "Werewolf" and len(self.get_werewolves()) > 1:
                    other_werewolf_votes = [votes.get(w.name) for w in self.get_werewolves() if w.name != player.name and w.name in votes]
                    if other_werewolf_votes and all(v == vote for v in other_werewolf_votes):
                        self.metrics["werewolf_team_coordination"] += 1
            
            self.logger.log_event("vote_reason", {"player": player.name, "vote": votes[player.name], "reason": vote_reason})

        self.logger.log_event("votes", votes)
        self.voting_history.append({"round": self.round, "votes": votes, "reasons": vote_reasons})
        self.logger.log_votes(self.round, votes)

        # Tally votes and track consensus
        vote_counts = {}
        for vote in votes.values():
            if vote != "Pass":
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
                
        # Calculate village consensus rate
        if vote_counts:
            max_votes = max(vote_counts.values())
            consensus_level = max_votes / len(alive_players)
            self.metrics["village_consensus_rate"] += consensus_level
            
            # Find player with most votes (randomly select if tied)
            eliminated_candidates = [name for name, count in vote_counts.items() if count == max_votes]
            eliminated_name = random.choice(eliminated_candidates)
            eliminated = next((p for p in alive_players if p.name == eliminated_name), None)
            if eliminated:
                eliminated.is_alive = False
                self.logger.log_event("elimination", {"eliminated": eliminated.name, "votes_received": max_votes, "total_voters": len(alive_players)})
                self.game_history.append(f"Day {self.round}: {eliminated.name} was eliminated with {max_votes} votes out of {len(alive_players)} voters")
                
                # Check if Seer accused a Werewolf correctly
                if eliminated.role == "Werewolf":
                    seer = next((p for p in alive_players if p.role == "Seer"), None)
                    if seer and votes.get(seer.name) == eliminated_name:
                        self.metrics["seer_correct_accusations"] += 1
        
        # Print round summary after day phase
        self.print_round_summary()

    def save_metrics(self):
        """Save detailed game metrics"""
        metrics_summary = {
            "game_id": self.game_id,
            "rounds_played": self.metrics["rounds_played"],
            "winner": self.metrics["winner"],
            "seer_performance": {
                "seer_accuracy": (self.metrics["seer_correct_accusations"] / self.metrics["total_seer_investigations"]
                                if self.metrics["total_seer_investigations"] > 0 else 0),
                "seer_reveal_rate": self.metrics["seer_reveals"] / self.metrics["rounds_played"] if self.metrics["rounds_played"] > 0 else 0,
                "total_investigations": self.metrics["total_seer_investigations"],
            },
            "werewolf_performance": {
                "deception_rate": self.metrics["werewolf_deceptions"] / self.metrics["total_discussion_statements"] if self.metrics["total_discussion_statements"] > 0 else 0,
                "team_coordination": self.metrics["werewolf_team_coordination"] / self.metrics["rounds_played"] if self.metrics["rounds_played"] > 0 else 0,
            },
            "medic_performance": {
                "successful_protections": self.metrics["medic_successful_protections"],
            },
            "village_performance": {
                "voting_accuracy": (self.metrics["votes_against_werewolves"] / self.metrics["total_votes"]
                                   if self.metrics["total_votes"] > 0 else 0),
                "consensus_rate": self.metrics["village_consensus_rate"] / self.metrics["rounds_played"] if self.metrics["rounds_played"] > 0 else 0,
            },
            "discussion_metrics": {
                "suspicion_change_rate": self.metrics["suspicion_changes"] / self.metrics["total_discussion_statements"] if self.metrics["total_discussion_statements"] > 0 else 0,
                "vote_discussion_alignment": self.metrics["vote_discussion_alignment"] / self.metrics["total_votes"] if self.metrics["total_votes"] > 0 else 0,
                "statement_variety_rate": self.metrics["statement_variety"] / self.metrics["total_discussion_statements"] if self.metrics["total_discussion_statements"] > 0 else 0,
                "total_statements": self.metrics["total_discussion_statements"],
            }
        }
        with open(self.logger.metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

    def run(self):
        self.logger.log_event("game_start", {"players": [(p.name, p.role) for p in self.players]})
        while True:
            self.night_phase()
            print(f"Night phase {self.round} completed...")
            win_result = self.check_win_condition()
            if win_result:
                self.metrics["winner"] = win_result
                self.logger.log_event("game_end", {"result": win_result})
                self.save_metrics()
                break
            self.day_phase()
            print(f"Day phase {self.round} completed...")
            win_result = self.check_win_condition()
            if win_result:
                self.metrics["winner"] = win_result
                self.logger.log_event("game_end", {"result": win_result})
                self.save_metrics()
                break
        print(f"Game Over: {win_result}")
        print(f"All game logs saved in directory: {self.logger.game_dir}")

if __name__ == "__main__":
    game = WerewolfGame(CONFIG["players"], CONFIG["azure_openai"], CONFIG["discussion_rounds"], randomize_roles=True)
    game.run()