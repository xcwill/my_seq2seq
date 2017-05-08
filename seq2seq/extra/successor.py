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
Define the model of successor features network
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

class  successor_nn(GraphModule,Configurable):
    def __init__(self):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)

    def _build(self,outputs):
        # TODO: a fully connected network.
        # 后面还需要对输出m_sa求最大，再和reward的权重作乘。
        return m_sa #一个tensor？