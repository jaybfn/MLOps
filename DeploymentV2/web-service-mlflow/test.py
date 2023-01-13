import requests


ride = {
    "PULocationID":448,
    "DOLocationID": 110,
    "trip_distance": 150
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
