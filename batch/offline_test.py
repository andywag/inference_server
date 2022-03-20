from offline.infer import main
from offline.offline_config import InferDescription

import logging
logger = logging.getLogger()

if __name__ == "__main__":
    inference_config = InferDescription()
    main(inference_config,None,None,logger)