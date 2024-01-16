import os


def text_to_speech(voiceEnabled, output):
    if voiceEnabled:
        os.system(f"say {output}")