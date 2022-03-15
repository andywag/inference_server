#!/bin/bash
INSTANCES=${1:-1}    

PYTHONPATH="${PYTHONPATH}:./"
PYTHONPATH="${PYTHONPATH}:./run"
#PYTHONPATH="${PYTHONPATH}:./serve"
#PYTHONPATH="${PYTHONPATH}:./client"

#pkill -9 -f "triton_server"
#pkill -9 -f "offline_bert_async_loadgen_"
#pkill -9 -f "triton_python_backend_stub"
pkill -9 -f "bert_runner"

#ray stop
#ray start --head --port=6379

python3 serve_new/bert_runner_zmq.py --instance $INSTANCES


