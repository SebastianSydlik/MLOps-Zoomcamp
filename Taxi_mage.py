import numpy as np
import pandas as pd
from mage_ai.io.file import FileIO
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data():
    df_raw = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet', engine='pyarrow')
    return df_raw
#    loader = FileIO(verbose=True)
#    df = loader.load(
#    'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
#   )



def clean_data(df_raw):
    df = pd.DataFrame.copy(df_raw)
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def fit_model(df):

    dv = DictVectorizer(sparse=False)

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    #df_val[categorical] = df_val[categorical].astype(str)

    train_dicts = df[categorical].to_dict(orient='records')
    #val_dicts = df_val[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    #X_val = dv.transform(val_dicts)  #CRUCIAL to use transform here instead of fit_transform, otherwiseget a mismatch in dimensionality between 
                                    #train and val feature matrix and thus cannot predict values.

    target = 'duration'
    y_train = df[target].values
    #y_val = df_val[target].values
    return (X_train, y_train)

def train_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    #y_pred = lr.predict(X_train)

    #mean_squared_error(y_train, y_pred, squared=False)
    return lr.intercept_


df_raw=load_data()
print(df_raw.shape)
df=clean_data(df_raw)
print(df.shape)
X_train, y_train = fit_model(df)
intercept = train_model(X_train, y_train)
print(intercept)
    
