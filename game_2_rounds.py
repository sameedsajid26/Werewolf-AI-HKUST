import json
import random
from datetime import datetime
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import os, sys

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
        if self.role == "Seer":
            self.knowledge = []  # List of (player_name, "Werewolf" or "Not a Werewolf")
        else:
            self.knowledge = []  # List of (player_name, suspicion_level) where suspicion_level is 0-1

    def __str__(self):
        return f"{self.name} ({self.role}, {'Alive' if self.is_alive else 'Dead'})"

class GameLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs = []
        self.discussion_file = "disc" + log_file
        self.discussions = []
        self.prompts_file = "prompt" + log_file
        self.prompts = []

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
            "data": data
        }
        self.discussions.append(disc_entry)
        with open(self.discussion_file, 'w') as f:
            json.dump(self.discussions, f, indent=2)
            
    def log_prompts(self, event_type: str, data: Dict):
        prompt_entry = {
            "data": data
        }
        self.prompts.append(prompt_entry)
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)

class WerewolfGame:
    def __init__(self, players: List[Dict], azure_config: Dict, discussion_rounds: int):
        self.players = [Player(p["name"], p["role"]) for p in players if p["role"] != "Moderator"]
        self.moderator = Player("Moderator", "Moderator")
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = GameLogger(f"werewolf_game_log_{start_time}.json")
        self.metrics_file = f"game_metrics_{start_time}.json"
        self.round = 0
        self.game_history = []
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
            "seer_reveals": 0,  # Track Seer role reveals
            "suspicion_changes": 0,  # Track suspicion updates
            "vote_discussion_alignment": 0,  # Votes aligned with discussion suspicions
            "total_discussion_statements": 0,  # Total statements made
            "statement_variety": 0,  # Track unique statement targets
            "werewolf_deceptions": 0  # Track Werewolf attempts to mislead
        }

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
                temperature=0.8  # Slightly increased for more varied responses
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}")
            self.logger.log_event("api_error", {"prompt": prompt, "error": str(e)})
            return ""

    def night_phase(self):
        self.round += 1
        self.metrics["rounds_played"] = self.round
        self.logger.log_event("night_start", {"round": self.round})
        alive_players = self.get_alive_players()
        player_names = [p.name for p in alive_players]

        # Werewolf selection
        werewolves = self.get_werewolves()
        if werewolves:
            werewolf_names = [p.name for p in werewolves]
            prompt = (
                f"In the fictional Werewolf game, you are a Werewolf. Your team: {werewolf_names}. "
                f"Alive players: {player_names}. Select one player as the target for this round. "
                f"Respond with only the player's name."
            )
            victim_name = self.call_api(prompt)
            victim = next((p for p in alive_players if p.name == victim_name), None)
            if not victim:
                victim = random.choice(alive_players)  # Fallback
            self.logger.log_event("werewolf_choice", {"victim": victim.name})
        else:
            victim = None

        # Seer investigation
        seer = next((p for p in alive_players if p.role == "Seer"), None)
        if seer:
            valid_targets = [p for p in alive_players if p.name != seer.name]  # Exclude self
            valid_names = [p.name for p in valid_targets]
            if valid_names:
                prompt = (
                    f"In the fictional Werewolf game, you are the Seer. Alive players (excluding yourself): {valid_names}. "
                    f"Select one player to investigate their role. Respond with only the player's name."
                )
                target_name = self.call_api(prompt)
                target = next((p for p in valid_targets if p.name == target_name), None)
                if not target:
                    target = random.choice(valid_targets)  # Fallback for invalid target
                result = "Werewolf" if target.role == "Werewolf" else "Not a Werewolf"
                seer.knowledge.append((target.name, result))
                self.metrics["total_seer_investigations"] += 1
                self.logger.log_event("seer_investigation", {"seer": seer.name, "target": target.name, "result": result})
            else:
                self.logger.log_event("seer_investigation", {"seer": seer.name, "error": "No valid targets"})

        # Medic protection
        medic = next((p for p in alive_players if p.role == "Medic"), None)
        if medic:
            valid_targets = [p for p in alive_players if p.name != medic.last_protected]
            valid_names = [p.name for p in valid_targets]
            prompt = (
                f"In the fictional Werewolf game, you are the Medic. Alive players: {valid_names}. "
                f"Select one player to protect this round (cannot be {medic.last_protected or 'none'}). "
                f"Respond with only the player's name."
            )
            protected_name = self.call_api(prompt)
            protected = next((p for p in valid_targets if p.name == protected_name), None)
            if protected:
                medic.last_protected = protected.name
                self.logger.log_event("medic_protection", {"medic": medic.name, "protected": protected.name})
            else:
                protected = None
        else:
            protected = None

        # Resolve night actions and update game history
        if victim and protected and victim.name == protected.name:
            self.logger.log_event("night_result", {"victim": victim.name, "saved": True})
            self.game_history.append(f"Night {self.round}: No one was killed")
        elif victim:
            victim.is_alive = False
            self.logger.log_event("night_result", {"victim": victim.name, "saved": False})
            self.game_history.append(f"Night {self.round}: {victim.name} was killed")
        else:
            self.game_history.append(f"Night {self.round}: No one was killed")

    def day_phase(self):
        self.logger.log_event("day_start", {"round": self.round})
        alive_players = self.get_alive_players()
        player_names = [p.name for p in alive_players]
        game_history_str = "\n".join(self.game_history) if self.game_history else "No game history yet."
        all_discussions = []
        all_prompts = []

        # Two Discussion Rounds
        for discussion_round in range(1, self.discussion_rounds + 1):
            discussion = []
            prompts = []
            previous_statements = []
            random.shuffle(alive_players)  # Randomize speaking order
            for player in alive_players:
                context = "\n".join(previous_statements) if previous_statements else "No statements yet."
                knowledge_str = "\n".join([f"{name}: {result}" for name, result in player.knowledge]) if player.knowledge else "None yet."
                past_statements = "\n".join(player.statements) if player.statements else "None yet."
                previous_round_statements = json.dumps(all_discussions[-1] if all_discussions else [])  # Previous round's discussion
                if player.role == "Werewolf":
                    werewolves = self.get_werewolves()
                    teammates = [p.name for p in werewolves if p != player]
                    role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}. Suspicions: {knowledge_str}"
                elif player.role == "Seer":
                    role_info = (
                        f"You are the Seer. Your investigations: {knowledge_str}. "
                        f"Use your knowledge strategically to guide the village without revealing your role unless critical "
                        f"(e.g., to save a confirmed non-Werewolf or accuse a confirmed Werewolf with strong evidence)."
                    )
                else:
                    role_info = f"You are a {player.role}. Suspicions: {knowledge_str}"

                prompt = (
                    f"In the fictional Werewolf game, {role_info}\n"
                    f"Game history:\n{game_history_str}\n"
                    f"Alive players: {', '.join(player_names)}\n"
                    f"Discussion round {discussion_round} statements:\n{context}\n"
                    f"Previous discussion round (if any):\n{previous_round_statements}\n"
                    f"Your previous statements this day phase:\n{past_statements}\n"
                    f"Now, as {player.name}, make a short statement about who you suspect or defend, "
                    f"considering the game history, previous rounds, and current discussion. "
                    f"Avoid repeating your previous statements; provide a new perspective or target if possible. "
                    f"Respond with your statement."
                )
                statement = self.call_api(prompt, max_tokens=50)
                discussion.append({"player": player.name, "statement": statement})
                prompts.append({"player": player.name, "prompt": prompt})
                previous_statements.append(f"{player.name}: {statement}")
                player.statements.append(f"Round {discussion_round}: {statement}")
                self.metrics["total_discussion_statements"] += 1

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
                        new_suspicion = min(existing_suspicion + 0.2, 1.0) if "suspect" in statement.lower() else max(existing_suspicion - 0.2, 0.0)
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

        # Voting
        votes = {}
        for player in alive_players:
            valid_targets = [p.name for p in alive_players if p.name != player.name]  # Exclude self
            knowledge_str = "\n".join([f"{name}: {result}" for name, result in player.knowledge]) if player.knowledge else "None yet."
            if player.role == "Werewolf":
                werewolves = self.get_werewolves()
                teammates = [p.name for p in werewolves if p != player]
                role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}. Suspicions: {knowledge_str}"
            elif player.role == "Seer":
                role_info = (
                    f"You are the Seer. Your investigations: {knowledge_str}. "
                    f"Use your knowledge strategically to guide your vote without revealing your role unless critical."
                )
            else:
                role_info = f"You are a {player.role}. Suspicions: {knowledge_str}"

            prompt = (
                f"In the fictional Werewolf game, {role_info}\n"
                f"Game history:\n{game_history_str}\n"
                f"Alive players (excluding yourself): {', '.join(valid_targets)}\n"
                f"All discussion rounds:\n{json.dumps(all_discussions)}\n"
                f"Based on the game history and all discussions, select one player to vote out this round, "
                f"or respond with 'Pass' if you lack sufficient information. Respond with only the player's name or 'Pass'."
            )
            vote = self.call_api(prompt)
            if vote == "Pass" or vote not in valid_targets:
                votes[player.name] = "Pass"
            else:
                votes[player.name] = vote
                self.metrics["total_votes"] += 1
                if next(p for p in alive_players if p.name == vote).role == "Werewolf":
                    self.metrics["votes_against_werewolves"] += 1
                # Check if vote aligns with discussion
                last_suspicions = [s["target"] for s in player.suspicion_changes if s["round"] == self.round]
                if vote in last_suspicions:
                    self.metrics["vote_discussion_alignment"] += 1
            self.logger.log_event("vote_reason", {"player": player.name, "vote": votes[player.name], "prompt": prompt})

        self.logger.log_event("votes", votes)

        # Tally votes
        vote_counts = {}
        for vote in votes.values():
            if vote != "Pass":
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
        if vote_counts:
            max_votes = max(vote_counts.values())
            eliminated_name = random.choice([name for name, count in vote_counts.items() if count == max_votes])
            eliminated = next((p for p in alive_players if p.name == eliminated_name), None)
            if eliminated:
                eliminated.is_alive = False
                self.logger.log_event("elimination", {"eliminated": eliminated.name})
                self.game_history.append(f"Day {self.round}: {eliminated.name} was eliminated")
                # Check if Seer accused a Werewolf correctly
                if eliminated.role == "Werewolf":
                    seer = next((p for p in alive_players if p.role == "Seer"), None)
                    if seer and votes.get(seer.name) == eliminated_name:
                        self.metrics["seer_correct_accusations"] += 1

    def save_metrics(self):
        metrics_summary = {
            "game_id": self.logger.log_file,
            "rounds_played": self.metrics["rounds_played"],
            "winner": self.metrics["winner"],
            "seer_accuracy": (self.metrics["seer_correct_accusations"] / self.metrics["total_seer_investigations"]
                             if self.metrics["total_seer_investigations"] > 0 else 0),
            "voting_accuracy": (self.metrics["votes_against_werewolves"] / self.metrics["total_votes"]
                               if self.metrics["total_votes"] > 0 else 0),
            "seer_reveal_rate": self.metrics["seer_reveals"] / self.metrics["rounds_played"] if self.metrics["rounds_played"] > 0 else 0,
            "suspicion_change_rate": self.metrics["suspicion_changes"] / self.metrics["total_discussion_statements"] if self.metrics["total_discussion_statements"] > 0 else 0,
            "vote_discussion_alignment": self.metrics["vote_discussion_alignment"] / self.metrics["total_votes"] if self.metrics["total_votes"] > 0 else 0,
            "statement_variety_rate": self.metrics["statement_variety"] / self.metrics["total_discussion_statements"] if self.metrics["total_discussion_statements"] > 0 else 0,
            "werewolf_deception_rate": self.metrics["werewolf_deceptions"] / self.metrics["total_discussion_statements"] if self.metrics["total_discussion_statements"] > 0 else 0
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

    def run(self):
        self.logger.log_event("game_start", {"players": [(p.name, p.role) for p in self.players]})
        while True:
            self.night_phase()
            print("Night phase started...")
            win_result = self.check_win_condition()
            if win_result:
                self.metrics["winner"] = win_result
                self.logger.log_event("game_end", {"result": win_result})
                self.save_metrics()
                break
            self.day_phase()
            print("Day phase started...")
            win_result = self.check_win_condition()
            if win_result:
                self.metrics["winner"] = win_result
                self.logger.log_event("game_end", {"result": win_result})
                self.save_metrics()
                break
        print(f"Game Over: {win_result}")

if __name__ == "__main__":
    game = WerewolfGame(CONFIG["players"], CONFIG["azure_openai"], CONFIG["discussion_rounds"])
    game.run()