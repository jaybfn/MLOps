
build docker container

---> docker build -t ride-duration-prediction-service:v1 .

to run the docker container

---> docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1