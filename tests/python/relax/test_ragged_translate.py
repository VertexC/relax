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
        
        r_b = dense_fixed(b) 
        r_n = ragged_dim(src_indptr, [d0])
        r_f = dense_fixed(f)
        r_h = dense_fixed(h)

        x_pos: R.RaggedTensor([r_b, r_n], [r_f, r_h]) 
            = ragged_tensor(src_data, dim_group=[[r_b, r_n], [r_f, r_h]])
        

        lv0: R.Tensor((h*f, 3, f, h), "float32") = R.transpose(w_qkv, [1, 0, 2, 3])
        lv1: R.Tensor((h*f, f*h*3), "float32") = R.reshape(lv0, (h*f, f*h*3))

        qkv: R.Tensor((w, 3*f*h)) = R.matmul(x_pos, lv1)

        qkv_split = R.split(qkv, 3, axis=2)

        q_0: R.Tensor((w, f*h)) = relax.TupleGetItem(qkv_split, 0) # (b, n, f*h)
        
        indptr_1: R.Tensor(b*h+1) = get_indptr(b, h)

        q: R.RaggedTensor([[r_b * r_h * r_n], [r_f]])
        

        transpose_pf_0(q_0, src_indptr, 
            q.data_src, indptr_1) # ragged.transpose(q_0, [0, 3, 1, 2])
        
        # ... # same for k, v
        
        k_t: R.RaggedTensor([r_b, r_h, r_f, r_n])
        indptr_2: R.Tensor(b*h*f+1)
        transpose_pf_1(k, indptr_1
            k_t.data_src, indptr_2) # ragged.transpose(k, [0, 1, 3, 2])
        
        
        # q @ k_t
        score: R.RaggedTensor([[r_b*r_h], [r_n], [r_n]])
        matmul_pf_0(q, indptr_1
            k_t, indptr_2, 
            score, indptr_1) # ragged_matmul_tir_func(q, k_t)
        
        scale = R.sqrt(f_scalar)
        score_scaled: R.RaggedTensor([[r_b*r_h], [r_n], [r_n]])

        divide_pf_1(
            score_scaled, indptr_1, indptr_1, 
            scale) # -> score_scaled = ragged_ewise_div_tir(score_scale, scale)
        score_softmax: R.RaggedTensor([[r_b*r_h], [r_n], [r_n]])
        softmax_pf_0(
            score_softmax.data_src, indptr_1, indptr_1,
            axis=3) # -> score_softmax = ragged_softmax_tir(score_scaled, axis=3)

        # score @ v
        z_: R.RaggedTensor([[r_b*r_h*r_n], [r_f]])
        matmul_pf_2(
            score_softmax.data_src, indptr_1, indptr_1,
            v.data_src, indptr_1,
            ) # z_ = ragged_matmul(score_softmax, v)
        
        z_before_merge: R.RaggedTensor([r_b, r_n], [r_h], [r_f])
        transpose_pf_2(
            z_.data_src, indptr_1,
            z_before_merge.data_src, src_indptr)
            # R.transpose(z_, [0, 2, 1, 3]) # z_before_merge =  ragged_tranpose_tir
        z_merge: R.RaggedTensor([r_b, r_n], [r_h, r_f]) 
        reshape_pf_0(
            z_merge, indptr_1)
        
        z: R.RaggedTensor([r_b, r_n], [r_h, r_f]) = R.matmul(
            z_merge.data_src, 
            w_z)
        
        # residual 
        res: R.RaggedTensor([r_b, r_n], [r_h, r_f]) = R.add(
            z.data_src, 
            x_pos.data_src)

        return res