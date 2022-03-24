# Inference Engine

This repository contains code for an inference engine on IPUs. It is a work in progress. 

## Installation
### Triton

source ./install_triton.sh : Potentially (Most Likely Issues with Install)

### Python Libraries
1. Requirements.txt : TBD
2. Standard IPU Setup with Pytorch required

## Run

The current code is not containerized and requires multiple parallel procsesses to be run. 

1. IPUs : "python3 release_proto.py" run in project folder
2. Triton : source start_triton_prototype.sh
3. FastAPI : "python3 fastapi_runner.py" in project folder
4. Node : "serve -s -l 8000" in react/inference folder