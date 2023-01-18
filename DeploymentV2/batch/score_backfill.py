import score
from prefect import flow
from dateutil.relativedelta import relativedelta
from datetime import datetime

@flow
def ride_duration_prediction_backfill():
    start_date = datetime(year=2021, month=3, day=1)
    end_date = datetime(year=2022, month=4, day=1)

    d = start_date

    while d <= end_date:
        score.ride_duration_prediction(
            taxi_type = 'green',
            run_id = '8afa88e2ba804c409da9215ac724c283',
            run_date = d
        )

        d = d + relativedelta(months=1)

if __name__=='__main__':

    ride_duration_prediction_backfill()
