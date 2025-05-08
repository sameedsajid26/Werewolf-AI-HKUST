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

# Configuration with 7 players, 2 discussion rounds, and experiment ID
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
    "discussion_rounds": 2,
    "experiment_id": "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S"),  # Format: experiment_YYYYMMDD_HHMMSS
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
    def __init__(self, log_file: str, experiment_folder: str):
        self.experiment_folder = experiment_folder
        # Ensure experiment folder exists
        os.makedirs(self.experiment_folder, exist_ok=True)
        # Construct file paths within experiment folder
        self.log_file = os.path.join(self.experiment_folder, log_file)
        self.discussion_file = os.path.join(self.experiment_folder, f"disc_{log_file}")
        self.prompts_file = os.path.join(self.experiment_folder, f"prompt_{log_file}")
        self.logs = []
        self.discussions = []
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
    def __init__(self, players: List[Dict], azure_config: Dict, discussion_rounds: int, experiment_id: str, randomize_roles=True):
        player_list = [p for p in players if p["role"] != "Moderator"]
        
        # Randomize roles if requested
        if randomize_roles:
            names = [p["name"] for p in player_list]
            roles = [p["role"] for p in player_list]
            random.shuffle(roles)
            self.players = [Player(names[i], roles[i]) for i in range(len(names))]
        else:
            self.players = [Player(p["name"], p["role"]) for p in player_list]
        
        self.moderator = Player("Moderator", "Moderator")
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_folder = os.path.join("experiments", experiment_id)
        self.logger = GameLogger(f"werewolf_game_log_{start_time}.json", self.experiment_folder)
        self.metrics_file = os.path.join(self.experiment_folder, f"game_metrics_{start_time}.json")
        self.round = 0
        self.game_history = []
        self.client = AzureOpenAI(
            azure_endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"]
        )
        self.deployment_name = azure_config["deployment_name"]
        self.discussion_rounds = discussion_rounds
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
            "werewolf_deceptions": 0
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
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}")
            self.logger.log_event("api_error", {"prompt": prompt, "error": str(e)})
            return ""

    def get_summarized_history(self):
        if not self.game_history:
            return "Game just started."
        recent_events = self.game_history[-min(3, len(self.game_history)):]
        deaths = [event for event in self.game_history if "was killed" in event or "was eliminated" in event]
        summary = "Recent events: " + "; ".join(recent_events)
        if deaths:
            summary += "\nAll deaths so far: " + "; ".join(deaths)
        return summary

    def format_player_knowledge(self, player):
        if not player.knowledge:
            return "No specific knowledge yet."
        if player.role == "Seer":
            return "\n".join([f"- {name}: {result}" for name, result in player.knowledge])
        else:
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
        if not all_discussions:
            return "No previous discussions."
        summaries = []
        for disc_round in all_discussions:
            round_num = disc_round.get("discussion_round", "?")
            statements = disc_round.get("statements", [])
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
        if not all_discussions:
            return "No discussions yet."
        all_statements = []
        for disc_round in all_discussions:
            for stmt in disc_round.get("statements", []):
                all_statements.append(stmt)
        player_mentions = {}
        for stmt in all_statements:
            speaker = stmt.get("player", "Unknown")
            statement = stmt.get("statement", "")
            for player in self.get_alive_players():
                if player.name in statement and player.name != speaker:
                    if player.name not in player_mentions:
                        player_mentions[player.name] = []
                    player_mentions[player.name].append(f"{speaker}: {statement}")
        key_mentions = []
        for player_name, mentions in player_mentions.items():
            if mentions:
                key_mentions.append(f"About {player_name}:")
                for mention in mentions[:2]:
                    key_mentions.append(f"- {mention}")
        return "\n".join(key_mentions) if key_mentions else "No significant accusations or defenses yet."

    def get_role_strategy(self, player, discussion_round):
        alive_players = self.get_alive_players()
        if player.role == "Werewolf":
            strategies = [
                "Try to blend in by accusing other players without drawing attention to yourself",
                "Defend your fellow werewolves subtly, but don't make it obvious",
                "Consider fake-claiming a role if pressured (but be careful, as this is risky)",
                "Try to create confusion by casting doubt on vocal players"
            ]
            if discussion_round == 1:
                strategies.append("In the first round, observe before making strong accusations")
            else:
                strategies.append("Build on previous discussions to seem consistent")
            if len(self.get_werewolves()) < len(alive_players) // 3:
                strategies.append("Your team is outnumbered - focus on survival rather than aggression")
            return "\n".join([f"- {s}" for s in strategies[:3]])
        elif player.role == "Seer":
            strategies = [
                "you are the main guy to help the villagers, because you can check a player in each round. if you get to know who the wearwolf is. try eliminating them, and guide others to vote them out as well. ",
                "If you've found a werewolf, consider carefully when to reveal this information",
                "If pressured, you should your role to save yourself or confirm information",
                "Pay attention to contradictions in player statements compared to your knowledge"
            ]
            werewolf_found = any(result == "Werewolf" for _, result in player.knowledge)
            if werewolf_found:
                strategies.append("You've identified a werewolf - consider how to use this information most effectively")
            return "\n".join([f"- {s}" for s in strategies[:3]])
        elif player.role == "Medic":
            strategies = [
                "Keep your role secret to avoid being targeted by werewolves",
                "Pay attention to discussions to identify potential Seers to protect",
                "Vary your protection targets to be unpredictable",
                "Consider protecting players who are under suspicion but you believe are innocent"
            ]
            return "\n".join([f"- {s}" for s in strategies[:3]])
        else:  # Villager
            strategies = [
                "look for what the seer says. they know useful information",
                "Be careful about who you trust, but work with others to identify werewolves",
                "Don't reveal too much about your suspicions too early",
                "Pay attention to voting patterns from previous rounds"
            ]
            return "\n".join([f"- {s}" for s in strategies[:3]])

    def get_voting_strategy(self, player):
        if player.role == "Werewolf":
            return (
                "Vote strategically to eliminate villagers, especially those who might be the Seer or Medic. "
                "Avoid voting for your fellow werewolves. Consider voting for players who are suspicious of you "
                "or your teammates. Try to align your vote with village consensus if possible."
            )
        elif player.role == "Seer":
            return (
                "Use your investigation results to guide your vote. Prioritize voting for confirmed werewolves. help influence others in voting too "
                "If you haven't found a werewolf yet, vote based on suspicious behavior. "
                "Consider the consequences of revealing your knowledge through your vote."
            )
        elif player.role == "Medic":
            return (
                "Vote based on observed behavior and discussion patterns. Try to identify werewolves through discussions "
                "their inconsistencies or suspicious defenses. Be wary of players who seem to be working together."
            )
        else:  # Villager
            return (
                "Vote based on the evidence from discussions. Look for inconsistencies in statements. "
                "Consider who made the most logical arguments. Be cautious of players making vague accusations."
            )

    def night_phase(self):
        self.round += 1
        self.metrics["rounds_played"] = self.round
        self.logger.log_event("night_start", {"round": self.round})
        print(f"\n{'='*50}")
        print(f"Night Phase - Round {self.round}")
        print(f"{'='*50}")
        
        alive_players = self.get_alive_players()
        player_names = [p.name for p in alive_players]
        print(f"\nAlive Players: {', '.join(player_names)}")

        # Werewolf selection
        werewolves = self.get_werewolves()
        if werewolves:
            werewolf_names = [p.name for p in werewolves]
            print(f"\nWerewolves ({', '.join(werewolf_names)}) are choosing their target...")
            prompt = (
                f"In the fictional Werewolf game, you are a Werewolf. Your team: {werewolf_names}. "
                f"Alive players: {player_names}.\n"
                f"Game history summary: {self.get_summarized_history()}\n"
                f"Strategic considerations:\n"
                f"- Target influential players who might be the Seer or Medic\n"
                f"- Avoid targeting players who were protected previously\n"
                f"- Consider eliminating players who are suspicious of werewolves\n"
                f"Select one player as the target for this round. Respond with only the player's name."
            )
            victim_name = self.call_api(prompt)
            victim = next((p for p in alive_players if p.name == victim_name), None)
            if not victim:
                victim = random.choice(alive_players)  # Fallback
            self.logger.log_event("werewolf_choice", {"victim": victim.name})
            print(f"Werewolves have chosen to target: {victim.name}")
        else:
            victim = None
            print("\nNo werewolves remain!")

        # Seer investigation
        seer = next((p for p in alive_players if p.role == "Seer"), None)
        if seer:
            print(f"\n{seer.name} (Seer) is investigating a player...")
            valid_targets = [p for p in alive_players if p.name != seer.name]
            valid_names = [p.name for p in valid_targets]
            investigated = [name for name, _ in seer.knowledge]
            uninvestigated = [p.name for p in valid_targets if p.name not in investigated]
            if valid_names:
                prompt = (
                    f"In the fictional Werewolf game, you are the Seer. Alive players (excluding yourself): {valid_names}.\n"
                    f"Game history summary: {self.get_summarized_history()}\n"
                    f"Your previous investigations: {seer.knowledge}\n"
                    f"Players you haven't investigated yet: {uninvestigated}\n"
                    f"Strategic considerations:\n"
                    f"- Prioritize players who are vocal or suspicious\n"
                    f"- Consider investigating players who others suspect\n"
                    f"- Balance between checking new players and verifying suspicions\n"
                    f"Select one player to investigate their role. Respond with only the player's name."
                )
                target_name = self.call_api(prompt)
                target = next((p for p in valid_targets if p.name == target_name), None)
                if not target:
                    if uninvestigated:
                        target = next((p for p in valid_targets if p.name in uninvestigated), None)
                    else:
                        target = random.choice(valid_targets)
                result = "Werewolf" if target.role == "Werewolf" else "Not a Werewolf"
                seer.knowledge.append((target.name, result))
                self.metrics["total_seer_investigations"] += 1
                self.logger.log_event("seer_investigation", {"seer": seer.name, "target": target.name, "result": result})
                print(f"{seer.name} investigated {target.name} and found they are {result}")
            else:
                self.logger.log_event("seer_investigation", {"seer": seer.name, "error": "No valid targets"})
                print(f"{seer.name} has no valid targets to investigate")
        else:
            print("\nNo Seer remains!")

        # Medic protection
        medic = next((p for p in alive_players if p.role == "Medic"), None)
        if medic:
            print(f"\n{medic.name} (Medic) is choosing who to protect...")
            valid_targets = [p for p in alive_players if p.name != medic.last_protected]
            valid_names = [p.name for p in valid_targets]
            prompt = (
                f"In the fictional Werewolf game, you are the Medic. Alive players: {valid_names}.\n"
                f"Game history summary: {self.get_summarized_history()}\n"
                f"You cannot protect {medic.last_protected or 'none'} again this round.\n"
                f"Strategic considerations:\n"
                f"- Protect players who might be the Seer or other key roles\n"
                f"- Consider protecting players who are vocal against werewolves\n"
                f"- Avoid predictable protection patterns\n"
                f"Select one player to protect this round. Respond with only the player's name."
            )
            protected_name = self.call_api(prompt)
            protected = next((p for p in valid_targets if p.name == protected_name), None)
            if protected:
                medic.last_protected = protected.name
                self.logger.log_event("medic_protection", {"medic": medic.name, "protected": protected.name})
                print(f"{medic.name} has chosen to protect {protected.name}")
            else:
                protected = None
        else:
            protected = None
            print("\nNo Medic remains!")

        # Resolve night actions
        if victim and protected and victim.name == protected.name:
            self.logger.log_event("night_result", {"victim": victim.name, "saved": True})
            self.game_history.append(f"Night {self.round}: No one was killed")
            print(f"\n{protected.name} was protected by the Medic and survived the night!")
        elif victim:
            victim.is_alive = False
            self.logger.log_event("night_result", {"victim": victim.name, "saved": False})
            self.game_history.append(f"Night {self.round}: {victim.name} was killed")
            print(f"\n{victim.name} was killed by the werewolves!")
        else:
            self.game_history.append(f"Night {self.round}: No one was killed")
            print("\nNo one was killed during the night!")

    def day_phase(self):
        self.logger.log_event("day_start", {"round": self.round})
        print(f"\n{'='*50}")
        print(f"Day Phase - Round {self.round}")
        print(f"{'='*50}")
        
        alive_players = self.get_alive_players()
        player_names = [p.name for p in alive_players]
        print(f"\nAlive Players: {', '.join(player_names)}")
        
        game_summary = self.get_summarized_history()
        all_discussions = []
        all_prompts = []

        # Two Discussion Rounds
        for discussion_round in range(1, self.discussion_rounds + 1):
            print(f"\nDiscussion Round {discussion_round} of {self.discussion_rounds}")
            print("-"*30)
            discussion = []
            prompts = []
            previous_statements_summary = self.summarize_statements(all_discussions) if all_discussions else "No previous discussion yet."
            random.shuffle(alive_players)
            for player in alive_players:
                current_round_statements = "\n".join([f"{p['player']}: {p['statement']}" for p in discussion]) if discussion else "No statements yet."
                knowledge_str = self.format_player_knowledge(player)
                strategy_guidance = self.get_role_strategy(player, discussion_round)
                if player.role == "Werewolf":
                    werewolves = self.get_werewolves()
                    teammates = [p.name for p in werewolves if p != player]
                    role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}."
                elif player.role == "Seer":
                    role_info = f"You are the Seer. Your investigations: {knowledge_str}."
                else:
                    role_info = f"You are a {player.role}."
                prompt = (
                    f"In the fictional Werewolf game, {role_info}\n"
                    f"Game summary: {game_summary}\n"
                    f"Alive players: {', '.join(player_names)}\n"
                    f"Discussion round {discussion_round} of {self.discussion_rounds}\n"
                    f"Previous discussion rounds summary: {previous_statements_summary}\n"
                    f"Current round statements: {current_round_statements}\n"
                    f"Your knowledge: {knowledge_str}\n"
                    f"Strategic guidance: {strategy_guidance}\n"
                    f"Now, as {player.name}, make a strategic statement about who you suspect or defend. "
                    f"Your statement should directly advance your win condition while appearing logical to others. "
                    f"Be specific with your reasoning and avoid vague statements. "
                    f"Respond with only your in-character statement (1-2 sentences)."
                )
                statement = self.call_api(prompt, max_tokens=75)
                discussion.append({"player": player.name, "statement": statement})
                prompts.append({"player": player.name, "prompt": prompt})
                player.statements.append(f"Round {discussion_round}: {statement}")
                print(f"{player.name}: {statement}")
                self.metrics["total_discussion_statements"] += 1
                if player.role == "Seer" and "I am the Seer" in statement:
                    self.metrics["seer_reveals"] += 1
                if player.role == "Werewolf" and any(p.name in statement for p in self.get_villagers()):
                    self.metrics["werewolf_deceptions"] += 1
                for p in alive_players:
                    if p.name != player.name and p.name in statement:
                        existing_suspicion = next((level for name, level in player.knowledge if name == p.name and isinstance(level, float)), 0.0)
                        new_suspicion = min(existing_suspicion + 0.2, 1.0) if "suspect" in statement.lower() else max(existing_suspicion - 0.2, 0.0)
                        if existing_suspicion != new_suspicion:
                            player.knowledge = [(name, level) for name, level in player.knowledge if name != p.name]
                            player.knowledge.append((p.name, new_suspicion))
                            player.suspicion_changes.append({"round": self.round, "discussion_round": discussion_round, "target": p.name, "new_suspicion": new_suspicion})
                            self.metrics["suspicion_changes"] += 1
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
        print("\nVoting Phase")
        print("-"*30)
        votes = {}
        for player in alive_players:
            valid_targets = [p.name for p in alive_players if p.name != player.name]
            knowledge_str = self.format_player_knowledge(player)
            voting_strategy = self.get_voting_strategy(player)
            discussion_summary = self.extract_key_accusations(all_discussions)
            if player.role == "Werewolf":
                werewolves = self.get_werewolves()
                teammates = [p.name for p in werewolves if p != player]
                role_info = f"You are a Werewolf. Your teammates are {', '.join(teammates)}."
            elif player.role == "Seer":
                role_info = f"You are the Seer. Your investigations: {knowledge_str}."
            else:
                role_info = f"You are a {player.role}."
            prompt = (
                f"In the fictional Werewolf game, {role_info}\n"
                f"Game summary: {game_summary}\n"
                f"Alive players (excluding yourself): {', '.join(valid_targets)}\n"
                f"Key accusations and defenses from discussions:\n{discussion_summary}\n"
                f"Your knowledge: {knowledge_str}\n"
                f"Voting strategy: {voting_strategy}\n"
                f"Based on all information, vote for one player to eliminate, "
                f"or respond with 'Pass' if you truly cannot decide. Respond with only the player's name or 'Pass'."
            )
            vote = self.call_api(prompt)
            if vote == "Pass" or vote not in valid_targets:
                votes[player.name] = "Pass"
                print(f"{player.name} chose to pass their vote")
            else:
                votes[player.name] = vote
                print(f"{player.name} voted to eliminate {vote}")
                self.metrics["total_votes"] += 1
                if next(p for p in alive_players if p.name == vote).role == "Werewolf":
                    self.metrics["votes_against_werewolves"] += 1
                last_suspicions = [s["target"] for s in player.suspicion_changes if s["round"] == self.round]
                if vote in last_suspicions:
                    self.metrics["vote_discussion_alignment"] += 1
            self.logger.log_event("vote_reason", {"player": player.name, "vote": votes[player.name], "prompt": prompt})

        self.logger.log_event("votes", votes)

        # Tally votes
        print("\nVote Results:")
        print("-"*30)
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
                print(f"{eliminated.name} was eliminated with {max_votes} votes!")
                if eliminated.role == "Werewolf":
                    print(f"{eliminated.name} was a Werewolf!")
                    seer = next((p for p in alive_players if p.role == "Seer"), None)
                    if seer and votes.get(seer.name) == eliminated_name:
                        print(f"The Seer correctly identified a Werewolf!")
                        self.metrics["seer_correct_accusations"] += 1
        else:
            print("No one received any votes - no elimination this round")

    def save_metrics(self):
        metrics_summary = {
            "game_id": os.path.basename(self.logger.log_file),
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
        os.makedirs(self.experiment_folder, exist_ok=True)
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
    game = WerewolfGame(
        CONFIG["players"],
        CONFIG["azure_openai"],
        CONFIG["discussion_rounds"],
        CONFIG["experiment_id"],
        randomize_roles=True
    )
    game.run()