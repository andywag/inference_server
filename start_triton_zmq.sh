PYTHONPATH="${PYTHONPATH}:/localdata/andyw/projects/public_examples/applications/inference/bert"
PYTHONPATH="${PYTHONPATH}:/localdata/andyw/projects/public_examples/applications/inference/bert/run"
PYTHONPATH="${PYTHONPATH}:/localdata/andyw/projects/public_examples/applications/inference/bert/serve_new"

./triton/server/install/bin/tritonserver --model-repository ./models/bert_zmq --backend-directory ./triton/python_backend
