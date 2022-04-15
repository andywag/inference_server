# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL := /bin/bash

MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /workspace

BUILD_DIR := build
DATA_DIR := $(BUILD_DIR)/data
BERT_DIR := $(DATA_DIR)/bert_tf_v1_1_large_fp32_384_v2
RESULT_DIR := $(BUILD_DIR)/result
MLPERF_CONF := $(BUILD_DIR)/mlperf.conf
FEATURE_CACHE := eval_features.pickle

# Handle different nvidia-docker version
ifneq ($(wildcard /usr/bin/nvidia-docker),)
	DOCKER_RUN_CMD := nvidia-docker run
else
	DOCKER_RUN_CMD := docker run --gpus=all
endif

.PHONY: setup
setup:
	@if [ ! -e $(BUILD_DIR) ]; then \
		mkdir $(BUILD_DIR); \
	fi
	@if [ ! -e $(MLPERF_CONF) ]; then \
		cp mlperf.conf $(MLPERF_CONF); \
	fi
	#@$(MAKE) -f $(MAKEFILE_NAME) init_submodule
	@$(MAKE) -f $(MAKEFILE_NAME) download_data
	@$(MAKE) -f $(MAKEFILE_NAME) download_model

#.PHONY: init_submodule
#init_submodule:
#	@git submodule update --init DeepLearningExamples

.PHONY: download_data
download_data:
	@if [ ! -e $(DATA_DIR) ]; then \
		mkdir $(DATA_DIR); \
	fi
	@if [ ! -e $(DATA_DIR)/dev-v1.1.json ]; then \
		wget -O $(DATA_DIR)/dev-v1.1.json https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json?raw=true; \
	fi
	@if [ ! -e $(DATA_DIR)/evaluate-v1.1.py ]; then \
		wget -O $(DATA_DIR)/evaluate-v1.1.py https://github.com/allenai/bi-att-flow/raw/master/squad/evaluate-v1.1.py; \
	fi
	@if [ ! -e $(BERT_DIR) ]; then \
		mkdir $(BERT_DIR) ; \
	fi
	@if [ ! -e $(RESULT_DIR) ]; then \
		mkdir $(RESULT_DIR); \
	fi

.PHONY: download_model
download_model:
	@$(MAKE) -f $(MAKEFILE_NAME) download_tf_model

.PHONY: download_tf_model
download_tf_model:
	@if [ ! -e $(BERT_DIR)/model.ckpt-5474.data-00000-of-00001 ]; then \
		wget -O $(BERT_DIR)/model.ckpt-5474.data-00000-of-00001 https://zenodo.org/record/3733868/files/model.ckpt-5474.data-00000-of-00001?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/model.ckpt-5474.index ]; then \
		wget -O $(BERT_DIR)/model.ckpt-5474.index https://zenodo.org/record/3733868/files/model.ckpt-5474.index?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/model.ckpt-5474.meta ]; then \
		wget -O $(BERT_DIR)/model.ckpt-5474.meta https://zenodo.org/record/3733868/files/model.ckpt-5474.meta?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/vocab.txt ]; then \
		wget -O $(BERT_DIR)/vocab.txt https://zenodo.org/record/3733868/files/vocab.txt?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/model.pb ]; then \
		wget -O $(BERT_DIR)/model.pb https://zenodo.org/record/3939747/files/model.pb?download=1; \
	fi



.PHONY: evaluate
evaluate:
	@python3 $(DATA_DIR)/evaluate-v1.1.py.json $(DATA_DIR)/dev-v1.1.json $(RESULT_DIR)/predictions.json

.PHONY: clean
clean:
	@rm -rf ${BUILD_DIR}
	@rm -f  ${FEATURE_CACHE}
	@rm -f  onnxruntime_profile__*.json
