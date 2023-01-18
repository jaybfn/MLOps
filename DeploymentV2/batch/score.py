
import os
import pickle
import sys
import pandas as pd
import mlflow
import uuid # generates a unique ID 

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# generateing a list of uuid with length of the dataframe 

def generate_uuid(n):
    
    ride_id = list(map(lambda x: str(uuid.uuid4()), range(len(n))))
    return ride_id

def read_dataframe(filename: str):
    
    df = pd.read_parquet(filename)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuid(df)
    return df

def prepare_dictionaries(df: pd.DataFrame):

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def load_model(run_id):

    logged_model = f'gs://artifact-storage-mlflow/2/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def apply_model(input_file,run_id, output_file):

    print(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'loading the model with RUN_ID= {run_id}...')
    model = load_model(run_id)

    print(f'applying the model')
    y_pred = model.predict(dicts)

    print(f'saving the result at PATH = {output_file}...')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_dutation'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_dutation']  - df_result['predicted_duration']
    df_result['model_version'] = run_id

    return df_result.to_parquet(output_file, index = False)

def run():
  
    taxi_type = sys.argv[1] #'green'
    year = int(sys.argv[2]) #2021
    month = int(sys.argv[3]) #2 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'../../data/output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    run_id = sys.argv[4]#os.getenv('RUN_ID','8afa88e2ba804c409da9215ac724c283')

    apply_model(
        input_file= input_file,
        run_id=run_id, 
        output_file=output_file
    )
    
if __name__=='__main__':
    run()

