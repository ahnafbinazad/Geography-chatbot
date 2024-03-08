import json
import requests
from text_to_speech import text_to_speech

APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f"


class Weather:
    def get_weather(self, params, voiceEnabled):
        """
        Retrieves weather information based on the provided location.

        Parameters:
        - params: List, containing the command parameters where params[1] is the location.
        - voiceEnabled: Boolean, indicating whether text-to-speech is enabled.
        """
        succeeded = False
        api_url = r"https://api.openweathermap.org/data/2.5/weather?q="
        response = requests.get(api_url + params[1] + r"&units=metric&APPID=" + APIkey)

        if response.status_code == 200:
            response_json = json.loads(response.content)
            if response_json:
                t = response_json['main']['temp']
                tmi = response_json['main']['temp_min']
                tma = response_json['main']['temp_max']
                hum = response_json['main']['humidity']
                wsp = response_json['wind']['speed']
                conditions = response_json['weather'][0]['description']

                output = (f"In {params[1].capitalize()}, the temperature is {t}°C, varying between {tmi} and {tma} at "
                          f"the moment. Humidity is {hum}%, and wind speed is {wsp} m/s. {conditions}")

                print(output)
                text_to_speech(voiceEnabled, output)

                succeeded = True
        if not succeeded:
            print("Sorry, I could not resolve the location you gave me.")
