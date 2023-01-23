
import os
import pickle
import sys
import pandas as pd
import mlflow
import uuid # generates a unique ID 
from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline



key_file_path = "prefect-storage.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file_path

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
#artifact-storage-mlflow/2/8afa88e2ba804c409da9215ac724c283/artifacts/model
    logged_model = f'gs://artifact-storage-mlflow/2/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def save_results(df, y_pred, run_id, output_file):

    print(f'saving the result at PATH = {output_file}...')
    print('\n')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_dutation'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_dutation']  - df_result['predicted_duration']
    df_result['model_version'] = run_id
    df_result.to_parquet(output_file, index = False)
    


@task
def apply_model(input_file, upload_file, run_id, output_file):
    logger = get_run_logger()
    print('\n')
    logger.info(f'downloading the data from {input_file}...')
    df_download = read_dataframe(input_file)

    logger.info(f'uploading the data to {upload_file}...')
    df_download.to_parquet(upload_file, index = False)

    logger.info(f'reading the data from {upload_file}...')
    df = read_dataframe(upload_file)
    dicts = prepare_dictionaries(df)

    logger.info(f'loading the model with RUN_ID= {run_id}...')
    model = load_model(run_id)

    logger.info(f'applying the model')
    y_pred = model.predict(dicts)

    save_results(df, y_pred, run_id, output_file)
    return output_file
    
def get_paths(run_date, taxi_type, run_id):

    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    upload_file = f'gs://nyc-duration-prediction-jay/nyc-taxi-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'gs://nyc-duration-prediction-jay/taxi_type={taxi_type}/year={year:04d}/month={month:02d}/{run_id}.parquet'

    return input_file, upload_file, output_file

@flow

def ride_duration_prediction(
        taxi_type: str,  
        run_id: str,
        run_date: datetime = None):
    
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    input_file, upload_file, output_file = get_paths(run_date, taxi_type, run_id)
  
    apply_model(
        input_file= input_file,
        upload_file = upload_file,
        run_id=run_id, 
        output_file=output_file
    )

def run():
  
    taxi_type = sys.argv[1] #'green'
    year = int(sys.argv[2]) #2021
    month = int(sys.argv[3]) #2 

    run_id = sys.argv[4]#os.getenv('RUN_ID','8afa88e2ba804c409da9215ac724c283')

    ride_duration_prediction(
        taxi_type= taxi_type,  
        run_id= run_id,
        run_date = datetime(year=year, month=month, day=1))
    
if __name__=='__main__':
    run()

