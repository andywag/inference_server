# Inference Engine

This repository contains code for an inference engine on IPUs. There are 2 modes

1. Online : Low Latency Always Available - Code stored in online folder
2. Offline : Batched Process attached through celery queue - Code stored in batch folder

Both modes consist of a "server" which is running the server side functionallity. This 
consists of FastAPI, RabbitMQ, Celery, and MongoDB. The models can be run on any device containing
an IPU and attaches to the server using the Rabbit Message Queue.  

## Configuration 
This solution requires rabbitMQ and MongoDB to be running. The IP address is required so that devices can be registered. 
1. Put IP Address of host into online/release/config.yaml

## Server Startup
The server side can be run using docker compose. 

1. docker-compose build
2. docker-compose up

## Online Startup

The online models are registered with the server through the ZMQ message queue and can be run : 
1. run `python3 release_proto.py run --config server` from online/release folder

This mode will start ner, gpt2 and bart. The config.yaml file can be used change the models run. 
Additional models can be added by : 

1. Creating a prototype similar to ner_proto.py, squad_proto.py. 
2. Creating an interface to your model which supports a simple input interface run(inputs, callback)

## Offline Startup

This mode runs on a host which is attached to IPUs. The models are run on demand using the run queue. 

1. run `celery -A celery_worker worker --loglevel=INFO -n fine` from batch folder