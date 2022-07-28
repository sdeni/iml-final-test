import pandas as pd
from pandas_profiling import ProfileReport
import os

from sklearn.metrics import mean_squared_error
from datetime import timedelta

import numpy as np

import xgboost as xgb

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import matplotlib.pyplot as plt

@task
def load_data(path):
    data = pd.read_parquet(path)
    data.lpep_dropoff_datetime = pd.to_datetime(data.lpep_dropoff_datetime)
    data.lpep_pickup_datetime = pd.to_datetime(data.lpep_pickup_datetime)

    data['duration'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    data.duration = data.duration.apply(lambda td: td.total_seconds() / 60)
    data = data[(data.duration >= 1) & (data.duration <= 60)]
    
    data['PULocationID'].astype(str, copy=False)
    data['DOLocationID'].astype(str, copy=False)
    return data

@task
def check_data_quality(df):
    profile = ProfileReport(df, title="Data quality report")
    
    os.makedirs('reports', exist_ok=True)
    
    profile.to_file("reports/data-quality-report.html")

@task(retries=3)
def generate_datasets(train_frame, val_frame):
    num_features = ['trip_distance', 'extra', 'fare_amount']
    cat_features = ['PULocationID', 'DOLocationID']

    X_train = train_frame[num_features + cat_features]
    X_val = val_frame[num_features + cat_features] 

    y_train = train_frame['duration']
    y_val = val_frame['duration'] 
    return X_train, X_val, y_train, y_val

class TrainHistoryCollector(xgb.callback.TrainingCallback):
    def __init__(self, rounds):
        self.rounds = rounds
        self.x = np.linspace(0, self.rounds, self.rounds)
        self.history = []

    def _get_key(self, data, metric):
        return f'{data}-{metric}'

    def after_iteration(self, model, epoch, evals_log):
        
        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                key = self._get_key(data, metric_name)
                metric = log[len(log) - 1]
                print(f"after_iteration: {epoch} {key} {metric}")

                self.history.append(metric)

        return False

@task
def train_model(X_train, y_train, X_val, y_val):
    best_params = {
        'max_depth': 5,
        'min_child': 19.345653147972058,
        'objective': 'reg:linear',
        'reg_alpha': 0.031009193638004067,
        'reg_lambda': 0.013053945835415701,
        'seed': 111
    }

    train = xgb.DMatrix(X_train, label=y_train)
    validation = xgb.DMatrix(X_val, label=y_val)

    num_boost_round = 100
    train_history = TrainHistoryCollector(num_boost_round)

    booster = xgb.train(
        params = best_params,
        dtrain = train,
        evals = [(validation, "validation")],
        num_boost_round = num_boost_round,
        early_stopping_rounds = 50,
        callbacks=[train_history]
    )

    y_preds = booster.predict(validation)
    rmse = mean_squared_error(y_preds, y_val, squared=False)

    return (booster, train_history)

@task
def estimate_quality(model, X_val, y_val):
    validation = xgb.DMatrix(X_val, label=y_val)
    y_pred = model.predict(validation)
    return mean_squared_error(y_pred, y_val, squared=False)

@task
def all_ok_task():
    print('ALL_GOOD_TASK')

@task
def bad_performance_task(train_history):
    print('Model performance is bad!')
    plt.plot(train_history.x, train_history.history)
    plt.savefig('reports/train-log.png')


@flow(task_runner=SequentialTaskRunner())
def nyc_duration_flow():
    train_frame = load_data('green_tripdata_2021-01.parquet')
    check_data_quality(train_frame)

    val_frame = load_data('green_tripdata_2021-02.parquet')
    X_train, X_val, y_train, y_val = generate_datasets(train_frame, val_frame).result()
    model, train_history = train_model(X_train, y_train, X_val, y_val).result()
    rmse = estimate_quality(model, X_val, y_val).result()

    print(f"Model quality: {rmse}")

    threshold = 5.0

    if rmse > threshold:
        bad_performance_task(train_history)
    else:
        all_ok_task()

nyc_duration_flow()
