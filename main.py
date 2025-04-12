import requests
from prediction import Predict
import art

# api key to get the current climatic conditions of the desired location
API_KEY=""

print(art.art)
lati=float(input("Tell me the latitude of a location in India?"))
long=float(input("Tell me the longitude of a location in India?"))


# link to the api where the weather conditions can be accessed
link="https://api.openweathermap.org/data/2.5/weather?"
parameter={
    "lat":lati,
    "lon":long,
    "appid":API_KEY
}
# using requests to access the data
request=requests.get(link,params=parameter)
# saving the information in the json format in data variable
data=request.json()

# calling the module and giving it new data
pre=Predict(data['main']['temp'],data['main']['humidity'],data['wind']['speed'],data['main']['pressure'],data['clouds']['all'])
# training the model and testing the result
pop=pre.precipitation_probability()
# print the weather condition
print(f"The weather condition in {data['name']} :{pre.weather_condition(pop)}")
