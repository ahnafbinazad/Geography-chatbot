import random

import pandas as pd
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from text_to_speech import text_to_speech

read_expr = Expression.fromstring


class PlaceGuessingGame:
    kb = []

    def __init__(self, kb_file='logical-kb.csv'):
        self.places = ['Australia', 'Asia', 'France', 'Africa', 'Brazil', 'India', 'Europe']
        self.read_expr = Expression.fromstring
        self.load_knowledge_base(kb_file)

    def load_knowledge_base(self, kb_file):
        kb_data = pd.read_csv(kb_file, header=None)

        for row in kb_data[0]:
            expr = self.read_expr(row)

            # Check for contradictions before adding to the knowledge base
            if not self.has_contradiction(expr):
                self.kb.append(expr)
            else:
                raise ValueError(f"Contradiction found: {row}. Terminating program.")

    def has_contradiction(self, new_expr):
        # Check if the negation of the new expression is already in the knowledge base
        negation_expr = self.read_expr('-' + str(new_expr))
        return negation_expr in self.kb

    def play(self, voiceEnabled):
        rounds = "How many rounds do you want to play? "
        print(rounds)
        text_to_speech(voiceEnabled, rounds)
        num_rounds = int(input("> "))
        points = 0

        for rounds in range(1, num_rounds + 1):
            chosen_place = random.choice(self.places)
            answer = f"Round {rounds}: Is {chosen_place} a country or a continent? "
            print(answer)
            text_to_speech(voiceEnabled, answer)
            user_guess = input("> ").strip().lower()

            expr = read_expr(user_guess + '(' + chosen_place.lower() + ')')

            result_positive = ResolutionProver().prove(expr, self.kb, verbose=False)

            if result_positive:
                output = "That is correct!"
                print(output)
                text_to_speech(voiceEnabled, output)
                points += 1

            else:
                output = "That is incorrect!"
                print(output)
                text_to_speech(voiceEnabled, output)

        output = f"You got {points} out of {num_rounds} questions correct!"
        print(output)
        text_to_speech(voiceEnabled,output)
