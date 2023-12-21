import json
import requests

# insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f"


class Weather:
    def get_weather(self, params):
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
                print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is",
                      hum, "%, wind speed ", wsp, "m/s,", conditions)
                succeeded = True
        if not succeeded:
            print("Sorry, I could not resolve the location you gave me.")
