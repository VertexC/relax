# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations  # must import to defer parsing of annotations
import sys
import tempfile
import pytest
import tvm
from tvm import relax
from tvm._ffi.base import TVMError
from tvm.relax.transform import OperatorLegalizer

# from tvm.script import relax as R, tir as T

from tvm.script._parser import ir as I, relax as R, tir as T

import numpy as np

@I.ir_module
class Transformer:
    @R.function
    def transformer_block(
            x: R.Tensor(("b", "f", "h"), "float32"), 
            src_indptr: R.Tensor((b + 1,), "int32"), 
            src_data: R.Tesor(("w", f*h), "float32"),
            w_qkv: R.Tensor((3, f*h, f, h), "float32"),
            w_z: R.Tensor((h*f, h*f), "float32"),
            f_scalar: R.Tensor((1,), "float32"),
            ):
        # FIXME: d0 = tir.Var()
        # ragged related api should automatically convert it into RaggedDim
        r_b = dense_fixed(b) 
        r_n = ragged_dim(src_indptr, [d0])
        r_f = dense_fixed(f)
        r_h = dense_fixed(h)
        l_0 = [[r_b, r_n], [r_f, r_h]] # (b*n, f*h)

        x_pos_ragged: R.RaggedTensor(l_0) = ragged_tensor(src_data, dim_group=[l_0]) # -> x_pos: R.Tensor((w, f*h), "float32")
        
        lv0: R.Tensor((h*f, 3, f, h), "float32") = R.transpose(w_qkv, [1, 0, 2, 3])
        lv1: R.Tensor((h*f, f*h*3), "float32") = R.reshape(lv0, (h*f, f*h*3))

        r_fh3 = dense_fixed(f*h*3)
        l_1 = [[r_b, r_n], [3, r_f, r_h]] # (b*n, 3*f*h)
        qkv: R.RaggedTensor(l_1) = R.matmul(x_pos_ragged, lv1) # -> qkv: R.Tensor((w, f*h*3)) = R.matmul(src_data, lv1)

        qkv_split = R.split(qkv, 3, axis=2)

        l_2 = [[r_b, r_n], [r_f, r_h]]
        q_0: R.RaggedTensor(l_2) = relax.TupleGetItem(qkv_split, 0) # -> q_0: R.Tensor((w, f*h)) = relax.TupleGetItem(qkv_split, 0)
        q_1 = ragged_reshape(q_0, [[r_b], [r_n], [r_f], [r_h]]) # -> q_1_ragged = ragged_cast(q_0); TODO: ragged_reshape?
        l_3 = [[r_b], [r_h], [r_n], [r_f]]
        q: R.RaggedTensor(l_3) = ragged.transpose(q_1, [0, 3, 1, 2]) # -> q = ragged_transpose_tir_func(q_1_ragged)
        
        # ... # same for k, v
        
        l_4 = [[r_b], [r_h], [r_f], [r_n]]
        k_t: R.RaggedTensor(l_4, "float32") = ragged.transpose(k, [0, 1, 3, 2]) # -> q = ragged_transpose_tir_func(k_ragged)
        # q @ k_t
        l_5 = [[r_b], [r_h], [r_n], [r_n]]
        score: R.RaggedTensor(l_5, "float32") = ragged.matmul(q, k_t) # -> q = ragged_matmul_tir_func(q, k_t)
        
        scale = R.sqrt(f_scalar)
        score_scaled: R.RaggedTensor(l_5, "float32") = ragged_divide(score, scale) # -> score_scaled = ragged_ewise_div_tir(score_scale)
        score_softmax: R.RaggedTensor(l_5, "float32") = ragged_softmax(score_scaled, axis=3) # -> score_softmax = ragged_softmax_tir(score_scaled, axis=3)

        # score @ v
        l_6 = [[r_b], [r_h], [r_n], [r_f]]
        z_: R.RaggedTensor(l_6) = ragged.matmul(score_softmax, v) # z_ = ragged_matmul_tor(score_softmax, v)
        
        l_7 = [[r_b], [r_n], [r_h], [r_f]]
        l_8 = [[r_b, r_n], [r_h, r_f]]
        z_before_merge: R.RaggedTensor(l_7, "float32") = R.transpose(z_, [0, 2, 1, 3]) # z_before_merge =  ragged_tranpose_tir
        z_merge: R.RaggedTensor(l_8, "float32") = ragged.reshape(z_before_merge, l_8) # z_merge: R.Tensor(w, h*f) = R.matmul(w, h*f)
        z: R.RaggedTensor(l_8, "float32") = R.matmul(z_merge, w_z) # z: Tensor(w, h*f) = R.matmul(z_merge, w_z)
        
        # residual 
        res: RaggedTensor(l_8, "float32") = R.add(z, x_pos) # res: Tensor(w, h*f) = R.add(z, x_pos)

        # # layernorm(x)
        # ln: R.Tensor((b, n, h*f), "float32") = R.layer_norm(res, gamma, beta, axis=[-1])

        return res