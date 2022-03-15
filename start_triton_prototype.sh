
PROJECT_PATH="./project"

PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}/bart"
PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}/bert"
PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}/bert/run"
#PYTHONPATH="${PYTHONPATH}:/localdata/andyw/projects/public_examples/applications/inference/bert/serve_new"
PYTHONPATH="${PYTHONPATH}:./model_proto"
PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}/gpt2_model"
PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}/../public_api"


export TRITON_PROJECT_PROTO_PATH="./prototypes"

./triton/server/install/bin/tritonserver --model-repository ./project/gen_models --backend-directory ./triton/python_backend
