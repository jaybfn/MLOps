import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

@task
def read_dataframe(filename):

    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >=1) & (df.duration <=60)]
  
    categorical = ['PULocationID', 'DOLocationID']
 
    df[categorical] = df[categorical].astype(str)

    return df

@task
def add_features(df_train, df_val):

    # df_train = read_dataframe(train_path)
    # df_val = read_dataframe(train_val)

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

    return X_train, y_train, X_val, y_val, dv



#################### Modeling ##################

@task
def train_model_search(train, valid, y_val):
    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag('model','xgboost')
            mlflow.log_params(params)
            booster = xgb.train(params = params,
                    dtrain = train,
                    num_boost_round = 100,
                    evals =[(valid,'validation')],
                    early_stopping_rounds = 1)
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared = False)
            mlflow.log_metric('rmse', rmse)
        return {'loss':rmse,'status':STATUS_OK}
        
    search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4,100,1)),
    'learning_rate': hp.loguniform('learning_rate', -3,0),
    'reg_alpha':hp.loguniform('reg_alpha', -5,-1),
    'reg_lambda':hp.loguniform('reg_lambda', -6,1),
    'min_child_weight': hp.loguniform('min_child_weight', -1,3),
    'objective':'reg:linear',
    'seed':42 }

    best_result = fmin(
    fn = objective,
    space = search_space,
    algo = tpe.suggest,
    max_evals = 1,
    trials = Trials()
    )
    return best_result


@task
def train_best_model(train, valid, y_val, dv):

    with mlflow.start_run():

        params = {
        'learning_rate': 0.20472,
        'max_depth': 17,
        'min_child_weight': 1.240261172,
        'objective':'reg:linear',
        'reg_alpha':0.285678967,
        'reg_lambda': 0.004264404814,
        'seed':42
        }
        mlflow.log_params(params)
        
        booster = xgb.train(params = params,
                        dtrain = train,
                        num_boost_round = 100,
                        evals =[(valid,'validation')],
                        early_stopping_rounds = 50)

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric('rmse', rmse)

        with open('model/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('model/preprocessor.b', artifact_path='preprocessor')

        mlflow.xgboost.log_model(booster, artifact_path='model_mlflow')

train_path = '../data/green_tripdata_2021-01.parquet'
train_val = '../data/green_tripdata_2021-02.parquet'

@flow(task_runner=SequentialTaskRunner())
def main(train_path, train_val):

    mlflow.set_tracking_uri("sqlite:///prediction.db")
    mlflow.set_experiment("nycity-taxi-experiment")
    X_train = read_dataframe(train_path)
    X_val = read_dataframe(train_val)

    X_train, y_train, X_val, y_val, dv = add_features(X_train, X_val).result()  # .result is when you use add_feature function as @task

    train = xgb.DMatrix(X_train, label = y_train)
    valid = xgb.DMatrix(X_val, label = y_val)
    train_model_search(train, valid, y_val)
    train_best_model(train, valid, y_val, dv)

# from prefect.deployments import DeploymentSpec
# from prefect.orion.schemas.schedules import IntervalSchedule
# from prefect.flow_runners import SubprocessFlowRunner
# from datetime import timedelta

# DeploymentSpec(
#     flow=main,
#     name="model_training",
#     schedule=IntervalSchedule(interval=timedelta(minutes=5)),
#     flow_runner = SubprocessFlowRunner(),
#     tags=['ml']
# )

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta

deployment = Deployment.build_from_flow(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    work_queue_name="ml"
)

deployment.apply()  

    



