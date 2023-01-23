import requests


ride = {
    "PULocationID":175,
    "DOLocationID": 15,
    "trip_distance": 100
}


url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
