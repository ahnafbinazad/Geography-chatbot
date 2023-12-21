#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""

# Import the SimilarityFallback module for handling fallback scenarios
from similarity_fallback import SimilarityFallback

# Initialize the SimilarityFallback class
similarity_fallback = SimilarityFallback()

# Import the Weather module for handling weather-related queries
from weather import Weather

# Initialize the Weather class
weather = Weather()

#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipedia
#######################################################

#######################################################
# Initialise AIML agent
#######################################################
import aiml

# Create a Kernel object for AIML processing
kern = aiml.Kernel()

# Set the text encoding to None for unicode I/O
kern.setTextEncoding(None)

# Use the Kernel's bootstrap() method to initialize the Kernel
kern.bootstrap(learnFiles="mybot-basic.xml")

#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")

#######################################################
# Main loop
#######################################################
while True:
    # Get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

    # Pre-process user input and determine the response agent (if needed)
    responseAgent = 'aiml'

    # Activate the selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)

    # Post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            try:
                # Get a summary from Wikipedia based on user input
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd == 2:
            # Get weather information based on parameters
            weather.get_weather(params)
        elif cmd == 99:
            # Fallback to similarity-based response
            most_similar_index = similarity_fallback.calculate_cosine_similarity(userInput)
            if most_similar_index == 0:
                print("I did not get that, please try again.")
            else:
                relevant_answer = similarity_fallback.get_relevant_answer(most_similar_index)
                print(relevant_answer)
    else:
        print(answer)
