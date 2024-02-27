import csv
import random
from fuzzywuzzy import fuzz

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


class FuzzyLogicGame:
    def __init__(self):
        self.kb_file = "game_kb.csv"
        self.statements = [
            "There is uncertainty about whether elephants are the largest land mammals.",
            "Some sources suggest that dolphins might be considered fish rather than mammals.",
            "I'm not sure if the polar bear is the largest terrestrial carnivore.",
            "There is ambiguity regarding whether penguins can fly or not.",
            "It is unclear whether the Amazon Rainforest is the largest tropical rainforest in the world.",
            "Some experts argue that the Venus flytrap is the fastest closing plant in the world."
        ]
        self.score = 0

    def load_knowledge_base(self):
        kb = []
        with open(self.kb_file, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                kb.append(row)
        return kb

    def update_knowledge_base(self, statement, decision):
        # Check if the number of lines in the file is a multiple of 100
        with open(self.kb_file, "r") as file:
            num_lines = sum(1 for line in file)
            if num_lines % 100 == 0:
                self.rewrite_file()

        # Append the new statement and decision to the file
        with open(self.kb_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([statement, decision])
            file.write('\n')

    def rewrite_file(self):
        # Read the existing content of the file
        kb = self.load_knowledge_base()
        # Rewrite the file line by line
        with open(self.kb_file, "w", newline="") as file:
            writer = csv.writer(file)
            for row in kb:
                writer.writerow(row)

    def present_statement(self):
        # Randomly select a statement
        statement = random.choice(self.statements)
        print("Statement:", statement)
        return statement

    def get_player_decision(self):
        print("How would you interpret this statement?")
        print("1. Treat it as true.")
        print("2. Treat it as false.")
        print("3. Treat it as uncertain.")
        decision = input("Enter your decision (1/2/3): ").strip()
        return decision

    def evaluate_decision(self, statement, decision):
        kb = self.load_knowledge_base()

        # Find the decision stored in the knowledge base for the given statement
        correct_decision = None
        for kb_statement, kb_decision in kb:
            if kb_statement == statement:
                correct_decision = kb_decision
                break

        if correct_decision is None:
            print("Error: Statement not found in knowledge base.")
            return

        # Calculate similarity between user's decision and correct decision
        similarity = fuzz.ratio(str(correct_decision), decision)

        # Output similarity level to user
        print(f"Similarity to correct decision: {similarity}%")

        # Evaluate the correctness of the decision based on fuzzy logic
        if similarity >= 80:
            print("Correct! Your decision is very close to the correct one.")
            self.score += 1
        elif similarity >= 60:
            print("You're on the right track, but there's some uncertainty in your decision.")
        else:
            print("Incorrect! Your decision is quite different from the correct one.")
            self.score -= 1

    def play(self):
        print("Welcome to the Fuzzy Logic Game!")
        print("In this game, you'll be presented with statements related to nature, animals, and general knowledge.")
        print("You must decide whether each statement is true, false, or uncertain.")
        print("Let's begin!")

        while True:
            print("\nCurrent Score:", self.score)
            statement = self.present_statement()
            decision = self.get_player_decision()
            self.update_knowledge_base(statement, decision)
            self.evaluate_decision(statement, decision)

            play_again = input("Do you want to play again? (y/n): ").strip().lower()
            if play_again != "y":
                print("Thanks for playing! Final Score:", self.score)
                break


# Testing the game
game = FuzzyLogicGame()
game.play()
