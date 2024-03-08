import os

def text_to_speech(voiceEnabled, output):
    """
    Converts text to speech if voiceEnabled is True.

    Parameters:
    - voiceEnabled: Boolean, indicating whether text-to-speech is enabled.
    - output: String, text to be converted to speech.
    """
    if voiceEnabled:
        os.system(f"say {output}")
