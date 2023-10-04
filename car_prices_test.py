import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn import metrics
import pickle
with open('Cars_prices.pkl', 'rb') as file:
    # load the model object from the file
    model = pickle.load(file)
    km_driven=1000000
    fuel= 1           ##Diesel=0   Petrol=3   CNG =2  LPG=1 
    seller_type= 0  #individual=1, Dealer=0,Trustmark Dealer= 2
    transmission=1  #Manual=1, 'Automatic=0
    owner= 3    #First Owner=0 , Second Owner=2, Third Owner=4, Fourth & Above Owner=1, Test Drive Car=3
    mileage=25.0
    engine=1500
    max_power=111
    seats= 5 
    car_age= 4
    other_brands= 1         # put 1 if it's not Audi,BMW<Jaguar,Land,Lexus,Volvo and 0 if it one of them
    brand_Audi= 0           # put 1 if it's Audi and 0 if not
    brand_BMW=  0            # put 1 if it's BMW and 0 if not
    brand_Jaguar=0           # put 1 if it's Jaguar and 0 if not
    brand_Land=   0          # put 1 if it's Land and 0 if not
    brand_Lexus=   0         # put 1 if it's Lexus and 0 if not
    brand_Volvo=    0        # put 1 if it's Volvo and 0 if not

input={'km_driven':km_driven,
        'fuel':fuel, 'seller_type':seller_type,
        'transmission':transmission, 'owner':owner, 
        'mileage':mileage, 'engine':engine,
        'max_power':max_power, 'seats':seats, 
        'car_age':car_age, 'other brands':other_brands,
        'brand_Audi':brand_Audi, 'brand_BMW':brand_BMW,
         'brand_Jaguar':brand_Jaguar, 'brand_Land':brand_Land,
         'brand_Lexus':brand_Lexus, 'brand_Volvo':brand_Volvo}
predicted_class = model.predict([list(input.values())])

print(predicted_class)

