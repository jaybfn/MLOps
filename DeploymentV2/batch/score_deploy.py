from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    flow_location = 'score.py',
    name="ride_duration_prediction",
    parameters={
        "taxi_type": "green",
        "run_id": "8afa88e2ba804c409da9215ac724c283"
    },
    flow_storage='6c925200-3b50-4199-853f-a5f8bfc461c5',
    schedule=CronSchedule(cron="0 3 2 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml']
)


