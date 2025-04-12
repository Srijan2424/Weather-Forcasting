import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class Predict:
    def __init__(self,temp,humid,win,pres,cld):
        self.temperature=temp
        self.pressure=pres
        self.humidity=humid
        self.wind=win
        self.clouds=cld

    # calculating the precipitation percentage
    def precipitation_probability(self):
        # reading the file
        data_training = pd.read_csv("india_weather_custom_100.csv")

        # making an array with the different stats taken from the csv file
        x=data_training[['Temperature','Humidity','Wind','Pressure','Cloud Cover']]
        y=data_training[['Precipitation']]

        # normalising the data
        scalar=StandardScaler()
        X_scaled=scalar.fit_transform(x)

        # dividing the data into 4 different sections to test and train
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # using linear regression to train the data
        model=LinearRegression()
        model.fit(X_train,y_train)

        # finding the predicted value
        y_pred=model.predict(X_test)

        # finding out the mean square error
        mse=mean_squared_error(y_test,y_pred)

        # using the new data to find out the result for this data
        new_data = pd.DataFrame([[self.temperature, self.wind, self.pressure, self.humidity, self.clouds]], columns=['Temperature', 'Wind', 'Pressure', 'Humidity','Cloud Cover'])
        pop_prediction = model.predict(new_data)[0]

        # keeping the values of the data within a 0 and 100
        pop_prediction = np.clip(pop_prediction, 0, 100)

        # printing the result
        print(f'Precipitation Percentage:{round(pop_prediction[0])} %')
        # printing the error percentage of  training the data
        print(f"Error Percentage:{round(math.sqrt(mse))} %")
        return pop_prediction

    # processing the weather conditions
    def weather_condition(self,pop):
        if pop < 10:
            return "Clear or Sunny"
        elif 10 <= pop < 30:
            return "Partly Cloudy"
        elif 30<= pop < 50:
            return "Cloudy with Possible Showers"
        elif 50<= pop < 70:
            return "Rain Likely"
        elif 70<= pop < 90:
            return "Heavy Rain Expected"
        else:
            return "Stormy or Very Heavy Rain"