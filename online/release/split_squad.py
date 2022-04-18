# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

import collections
import json
import math
import os
import subprocess
import sys

import numpy as np


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes



from dataclasses import dataclass
@dataclass
class Solution:
    start_index:int
    end_index:int
    logit:float

def get_predictions(start_logits, end_logits, n_best_size = 5):
    start_indexes = _get_best_indexes(start_logits, n_best_size)
    end_indexes = _get_best_indexes(end_logits, n_best_size)


    solutions = []
    for start_index in start_indexes:
        if start_index < -1:
            continue
        for end_index in end_indexes:
            if end_index < -1 or end_index < start_index:
                continue
            solutions.append(Solution(start_index, end_index,start_logits[start_index]+end_logits[end_index]))
    
    solutions.sort(key=lambda x:-x.logit)
    return solutions[:5]
        

