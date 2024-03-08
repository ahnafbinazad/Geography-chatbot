# Import necessary modules
import os
import time
from flag_recogniser import FlagRecogniser
from weather import Weather
import wikipedia
from aiml import Kernel
from similarity_fallback import SimilarityFallback
from knowledge_base_inferencing import KnowledgeBaseInferencing
from text_to_speech import text_to_speech
from google_search import google
from fuzzy_game import FuzzyLogicGame
from video_recognition import VideoFlagRecognizer

# Temporary fix for an AttributeError in the time module
# TO-DO: Remove this time import once the issue is resolved
time.clock = time.time

# Initialize the FuzzyLogicGame class
fuzzy_game = FuzzyLogicGame()

# Initialize the FlagRecogniser class
flag_recogniser = FlagRecogniser()

# Initialize the VideoFlagRecognizer class
video_recognizer = VideoFlagRecognizer()

# Initialize the SimilarityFallback class
similarity_fallback = SimilarityFallback()

# Initialize the Weather class
weather = Weather()

# Initialize the KnowledgeBaseInferencing class
kb_inferencing = KnowledgeBaseInferencing()

# Create a Kernel object for AIML processing
kern = Kernel()
kern.verbose(False)

# Set the text encoding to None for Unicode I/O
kern.setTextEncoding(None)

# Use the Kernel's bootstrap() method to initialize the Kernel with AIML files
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome message to the user
print("Welcome to the Geography chatbot. Please feel free to ask questions. Just make sure they are related to geography!")

# Ask user if they want to enable text to speech
voiceEnabled = False
voice = input('Press y to enable text to speech: ')
if voice == 'y':
    voiceEnabled = True
    print('Text to speech has been enabled.')
else:
    print('Text to speech will remain disabled.')

# Main loop for user interaction
while True:
    # Get user input
    try:
        userInput = input("> ")
        if voiceEnabled: os.system(f"say {userInput}")  # Speak user input if text to speech is enabled
    except (KeyboardInterrupt, EOFError) as e:
        bye = 'Bye!'
        print(bye)
        if voiceEnabled: os.system(f"say {bye}")  # Speak "Bye!" if text to speech is enabled
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

        # Define functions for different command cases
        def case_0():  # Exit program
            print(params[1])
            exit()

        def case_1():  # Retrieve summary from Wikipedia
            try:
                wSummary = wikipedia.summary(params[1], sentences=1, auto_suggest=True)
                print(wSummary)
            except:
                search = "I can't find the answer for that. Let me ask Google and give you some websites to look at."
                print(search)
                google(userInput)

        def case_2():  # Retrieve weather information
            weather.get_weather(params, voiceEnabled)

        def case_31():  # Process input for knowledge base inference
            kb_inferencing.process_input(case=31, params=params, pattern=' IS ', voiceEnabled=voiceEnabled)

        def case_32():  # Process input for knowledge base inference
            kb_inferencing.process_input(case=32, params=params, pattern=' IS ', voiceEnabled=voiceEnabled)

        def case_51():  # Recognize flags from images
            flag_recogniser.flag_recogniser(voiceEnabled)

        def case_52():  # Recognize flags from videos
            video_recognizer.recognise_video()

        def case_62():  # Play the fuzzy game
            if voiceEnabled:
                output = "The fuzzy game does not support text to speech. Press y if you would like to continue."
                print(output)
                play = input("> ")
                if play == 'y':
                    fuzzy_game.play()
                else:
                    output = "Aborting fuzzy game."
                    print(output)
                    pass
            else:
                fuzzy_game.play()

        def case_99():  # Fallback to similarity-based response
            most_similar_index = similarity_fallback.calculate_cosine_similarity(userInput)
            if most_similar_index is not None:
                relevant_answer = similarity_fallback.get_relevant_answer(most_similar_index)
                print(relevant_answer)
                text_to_speech(voiceEnabled, relevant_answer)
            else:
                search = 'I cannot find the answer for that. Let me ask Google and give you some websites to look at.'
                print(search)
                text_to_speech(voiceEnabled, search)
                google(userInput)

        # Define switch cases for command execution
        switch_cases = {
            0: case_0,
            1: case_1,
            2: case_2,
            31: case_31,
            32: case_32,
            51: case_51,
            52: case_52,
            62: case_62,
            99: case_99,
        }

        # Execute the appropriate function based on the command
        locals().get(f'case_{cmd}', lambda: print("Invalid command"))()

    else:
        # Print the answer from AIML or other response agents
        print(answer)
        text_to_speech(voiceEnabled, answer)  # Speak the answer if text to speech is enabled
