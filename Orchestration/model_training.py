import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def read_dataframe(filename):

    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >=0) & (df.duration <=60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    train_dict = df[categorical + numerical].to_dict(orient = 'records')
    df[categorical] = df[categorical].astype(str)

    return df

df_train = read_dataframe('../data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('../data/green_tripdata_2021-02.parquet')

df_train['PU_DU'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DU'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

categorical = ['PU_DU']#'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_Dict = df_train[categorical + numerical].to_dict(orient = 'records')
X_train = dv.fit_transform(train_Dict)

val_Dict = df_val[categorical + numerical].to_dict(orient = 'records')
X_val = dv.transform(val_Dict)


target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values