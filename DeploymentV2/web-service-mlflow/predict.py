import pickle
import mlflow
from flask import Flask, request, jsonify
#from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

#MLFLOW_TRACKING_URI = 'http://34.141.21.45:5000'  # postgres on SQL engine GCP
RUN_ID = '8afa88e2ba804c409da9215ac724c283'

#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logged_model = f'gs://artifact-storage-mlflow/2/{RUN_ID}/artifacts/model'
#logged_model = f'runs:/{RUN_ID}/model'


model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):

    features = {}
    features['PU_DO'] = '%s_%s' %(ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features



def predict(features):

    preds = model.predict(features)
    return float(preds[0])



app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ =="__main__":

    app.run(debug=True, host='0.0.0.0', port=9696)