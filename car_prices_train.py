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
df=pd.read_csv('Car details v3.csv')
df
df.info()
df.describe()
df.columns[df.isna().any()]
df.isnull().sum()
df.torque=df.torque.fillna(0)
df.torque
df2=df[df.torque==0].head(50)
df2
df.dropna(inplace=True)
df
df.info()
df['car_age']=2023-df.year
df
df.drop('year',axis=1,inplace=True)
df
discrete_col=['fuel','seller_type','transmission','owner','seats']
num_row=3
num_col=2
i=0
def Convert_col_to_numeric(w):
    we=str(w)
    word=we.split(' ')
    if len(word) ==2:
        return float (word[0])
    try:
        return float (we)
    except:
         return None
df.mileage=df.mileage.apply(Convert_col_to_numeric)
df
df.engine=df.engine.apply(Convert_col_to_numeric)
df
def convert_max_power_to_num(x):
    print("Converting:", x)
    if pd.isnull(x) or not x.strip():
        return None
    x = str(x)
    tokens = x.split(' ')
    if len(tokens) == 2:
        return float(tokens[0])
    try:
        return float(x)
    except ValueError as e:
        print("Error:", e)
        print("Value causing error:", x)
        return None
df['max_power'] = df['max_power'].replace(' bhp', np.nan)
df['max_power']=df['max_power'].apply(Convert_col_to_numeric)
df
df["car_brand_name"] = df["name"].str.extract('([^\s]+)')
df["car_brand_name"] = df["car_brand_name"].astype("category")
df.drop('name',axis=1,inplace=True)
df
df=df.drop(['torque'],axis=1,)
df
def check_others(brand):
    if brand.lower() in ['ambassador', 'ashok', 'chevrolet', 'daewoo', 'datsun',
                         'fiat', 'force', 'ford', 'honda', 'hyundai', 'isuzu', 'jeep',
                         'kia', 'mg', 'mahindra', 'maruti', 'mercedes-benz', 'mitsubishi', 'nissan',
                         'opel', 'renault', 'skoda', 'tata', 'toyota', 'volkswagen']:
        return 1
    else:
        return 0
df['other brands'] = df['car_brand_name'].apply(lambda x: check_others(x))
df
def check_brand(brand):
    if brand.lower() in ['ambassador', 'ashok', 'chevrolet', 'daewoo', 'datsun',
                         'fiat', 'force', 'ford', 'honda', 'hyundai', 'isuzu', 'jeep',
                         'kia', 'mg', 'mahindra', 'maruti', 'mercedes-benz', 'mitsubishi', 'nissan',
                         'opel', 'renault', 'skoda', 'tata', 'toyota', 'volkswagen']:
        return 0
    else:
        return brand
df['car_brand_name'] =df['car_brand_name'].apply(check_brand)
dff=pd.get_dummies(df['car_brand_name'],prefix='brand')
dff['brand_0'].value_counts()
dff.drop('brand_0', inplace=True, axis=1)
dff
df=pd.concat([df,dff],axis=1)
df
df=df.drop('car_brand_name',axis=1)
df
Encode_list=['fuel','fuel','seller_type','owner','transmission',]
encoder= LabelEncoder()
for i in Encode_list:
    df[i]=encoder.fit_transform(df[i])
nan_rows = df[df['max_power'].isna()]
df.dropna(subset=['max_power'], inplace=True)
x=df.drop('selling_price',axis=1)
x
y=df['selling_price']
y
x_tarin,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
def Train_test_split(model,modelName):
    model.fit(x,y)
    model_train_score=model.score(x_tarin,y_train)
    model_test_score=model.score(x_test,y_test)
    print(f"{modelName} model score on Training data: {model_train_score * 100}%\n{modelName} model score on Testing data: {model_test_score * 100}%")
    return model
def R2(model,modelName):
    acc=r2_score(y_test,model.predict(x_test))
    print(f"R2 Score for {modelName} is {acc * 100}%")

model_random=RandomForestRegressor()
Train_Test_Model=Train_test_split(model_random,'random forest regression')
R2(model_random,'r2')
y_pred = model_random.predict(x_test)
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared: {}'.format(metrics.r2_score(y_test, y_pred)))
# open file to write the model
with open('Cars_prices.pkl', 'wb') as file:
    # dump the model object into the file
    pickle.dump(Train_Test_Model, file) #put model in file

# print confirmation message
print("Trained model saved successfully!",file)

