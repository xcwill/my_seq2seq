# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Definition of a reward and successor network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import numpy as np
import tensorflow as tf
from seq2seq.graph_module import GraphModule
from seq2seq.configurable import Configurable

from tensorflow.python.ops import init_ops

class reward_network(GraphModule, Configurable):
"""
构建reward网络，简单的线性回归模型
"""


    def __init__(self, params, mode, name="attention"):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)

    def _build(self,outputs):
        outputs_size = tf.shape(outputs)[1]
        dtype = outputs.dtype.base_dtype
        # Set up the requested initialization.
        init_mean = 0.0
        init_stddev = 0.0

        reward = tf.contrib.layers.fully_connected(
            inputs=outputs,
            num_outputs= 1,
            weights_initializer=init_ops.random_normal_initializer(init_mean, init_stddev, dtype=dtype),dtype=dtype)
            biases_initializer = init_ops.random_normal_initializer(init_mean, init_stddev, dtype=dtype),\
                               dtype = dtype
            activation_fn=None,
            scope="reward")
        return reward