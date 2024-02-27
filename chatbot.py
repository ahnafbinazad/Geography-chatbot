from flag_recogniser import flag_recogniser
from weather import Weather
import wikipedia
from aiml import Kernel
from similarity_fallback import SimilarityFallback
from knowledge_base_inferencing import KnowledgeBaseInferencing
from text_to_speech import text_to_speech
from logic_game import PlaceGuessingGame
from google_search import google
from fuzzy_game import FuzzyLogicGame
import os

# This time import is a last resort patch to eradicate the error
# "AttributeError: module 'time' has no attribute 'clock'"
# TO-DO: Remove this time import
import time
time.clock = time.time

# Initialize the games class
game = PlaceGuessingGame()
fuzzy_game = FuzzyLogicGame()

# Initialize the SimilarityFallback class
similarity_fallback = SimilarityFallback()

# Initialize the Weather class
weather = Weather()

# Initialize the KnowledgeBaseInferencing class
kb_inferencing = KnowledgeBaseInferencing()

# Create a Kernel object for AIML processing
kern = Kernel()
kern.verbose(False)

# Set the text encoding to None for unicode I/O
kern.setTextEncoding(None)

# Use the Kernel's bootstrap() method to initialize the Kernel
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome user
print("Welcome to the Geography chatbot. Please feel free to ask questions from me. Just make sure they are not out "
      "of this world!")

# Askk user if they want to enable text to speech
voiceEnabled = False

voice = input('Press y to enable text to speech: ')
if voice == 'y':
    voiceEnabled = True
    print('Text to speech has been enabled.')
else:
    print('Text to speech will remain disabled.')

# Main loop
while True:
    # Get user input
    try:
        userInput = input("> ")
        if voiceEnabled: os.system(f"say {userInput}")
    except (KeyboardInterrupt, EOFError) as e:
        bye = 'Bye!'
        print(bye)
        if voiceEnabled: os.system(f"say {bye}")
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


        def case_0():
            print(params[1])
            exit()


        def case_1():
            try:
                # Get a summary from Wikipedia based on user input
                wSummary = wikipedia.summary(params[1], sentences=1, auto_suggest=True)
                # text_to_speech(voiceEnabled, wSummary)
                print(wSummary)
            except:
                search = "I can't find the answer for that, let me ask google and give you some websites to look at."
                print(search)
                text_to_speech(voiceEnabled, search)

                google(userInput)


        def case_2():
            # Get weather information based on parameters
            weather.get_weather(params, voiceEnabled)


        def case_31():  # if input pattern is "I know that * contains *"
            case = 31
            kb_inferencing.process_input(case, params, ' IS ', voiceEnabled)


        def case_32():  # if the input pattern is "check that * contains *"
            case = 32
            kb_inferencing.process_input(case, params, ' IS ', voiceEnabled)


        def case_51():  # case for flag recognition
            flag_recogniser(voiceEnabled)
            return

        def case_61():  # case to play logic game
            game.play(voiceEnabled)

        def case_62():  # case to play fuzzy game
            if voiceEnabled:
                output = "The fuzzy game does not support text to speech, press y if you would still like to continue"
                print(output)
                text_to_speech(voiceEnabled, output)
                play = input("> ")

                if play == 'y':
                    fuzzy_game.play()
                else:
                    pass

            else:
                fuzzy_game.play()

        def case_99():
            # Fallback to similarity-based response
            most_similar_index = similarity_fallback.calculate_cosine_similarity(userInput)
            if most_similar_index is not None:
                relevant_answer = similarity_fallback.get_relevant_answer(most_similar_index)
                print(relevant_answer)
                text_to_speech(voiceEnabled, relevant_answer)
            else:
                search = "I can't find the answer for that, let me ask google and give you some websites to look at."
                print(search)
                text_to_speech(voiceEnabled, search)

                google(userInput)


        # Define the switch cases
        switch_cases = {
            0: case_0,
            1: case_1,
            2: case_2,
            31: case_31,
            32: case_32,
            51: case_51,
            61: case_61,
            62: case_62,
            99: case_99,
        }
        # Call the appropriate function based on the command
        locals().get(f'case_{cmd}', lambda: print("Invalid command"))()
    else:
        print(answer)
        text_to_speech(voiceEnabled, answer)
