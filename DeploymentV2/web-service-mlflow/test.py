import requests


ride = {
    "PULocationID":100,
    "DOLocationID": 15,
    "trip_distance": 250
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
