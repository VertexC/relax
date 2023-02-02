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
    @T.prim_func
    def infer_indptr_0(
        indptr_0: T.handle,
        indptr_1: T.handle,
        b: T.int64,
        h: T.int64,
    ) -> None:
        T.func_attr({"global_symbol": "infer_indptr_0", "tir.noalias": True})
        # b_plus_1 = T.var("int32")
        # b_times_h_plus_1 = T.var("int32")
        IDX_0 = T.match_buffer(indptr_0, (b+1,), dtype="int32")
        IDX_1 = T.match_buffer(indptr_1, (b*h+1,), dtype="int32")
        # IDX_1 = T.match_buffer(indptr_1, (bh), dtype="int32")



    @R.function
    def transformer_block(
            x: R.Tensor(("b", "f", "h"), "float32"),
            data_x: R.Tensor(("w", f*h), "float32"),
            indptr_0: R.Tensor((b + 1,), "int32"), 
            w_qkv: R.Tensor((3, f*h, f, h), "float32"),
            w_z: R.Tensor((h*f, h*f), "float32"),
            f_scalar: R.Tensor((1,), "float32"),
            ):
        lv0: R.Tensor((h*f, 3, f, h), "float32") = R.transpose(w_qkv, [1, 0, 2, 3])
        lv1: R.Tensor((h*f, f*h*3), "float32") = R.reshape(lv0, (h*f, f*h*3))

        qkv: R.Tensor((w, f*h*3)) = R.matmul(data_x, lv1)
        qkv_split = R.split(qkv, 3, axis=1)

        q_0: R.Tensor((w, f*h)) = relax.TupleGetItem(qkv_split, 0) # (b, n, f*h)

        # q: R.RaggedTensor([[r_b * r_h * r_n], [r_f]])

        # indptr_1 = R.call_tir(infer_indptr_0, 
        #     (indptr_0,), ((b*h)+1,), dtype="int32", tir_vars=(b, h))
        # data_q: R.Tensor((w*h, f), "float32")
        # transpose_pf_0(q_0, indptr_0, 
        #     data_q, indptr_1) # ragged.transpose(q_0, [0, 3, 1, 2])
        # R.call_tir(transpose_pf_0, ())
        
        # # ... # same for k, v
        
        # # k_t: R.RaggedTensor([r_b * r_h * r_f], [r_n])
        # data_k_t = R.Tensor((w*h*f), "float32")
        # indptr_2 = R.Tensor((b*h*f+1), "int32")
        # transpose_pf_1(k, indptr_0
        #     data_k_t, indptr_2) # ragged.transpose(k, [0, 1, 3, 2])
        
        # # q @ k_t
        # # score: R.RaggedTensor([[r_b*r_h], [r_n], [r_n]])
        # # TODO: what is the general rule of getting total length of shape?
        # # TODO: what should be the index pointer for the second r_n?
        # s = sqr_pf0(indptr_0)
        # data_score = R.Tensor((h*s), "float32")
        # indptr_3 = R.Tensor((w*h))
        # matmul_pf_0(q, indptr_1
        #     data_k_t, indptr_2,
        #     data_score, indptr_1, indptr_3) # ragged_matmul_tir_func(q, k_t)
        
        # scale = R.sqrt(f_scalar)
        # # score_scaled: R.RaggedTensor([[r_b*r_h], [r_n], [r_n]])
        # score_scaled_data : R.Tensor((h*s), "float32") = 
        #     R.divide(data_score, scale)

        # # score_softmax: R.RaggedTensor([[r_b*r_h], [r_n], [r_n]])
        # score_softmax_data = R.Tensor((h*s), "float32")
        # softmax_pf_0(
        #     score_softmax_data, indptr_1, indptr_3,
        #     axis=3)

        # # score @ v
        # # z_: R.RaggedTensor([[r_b*r_h*r_n], [r_f]])
        # z_data_ = R.Tensor(w*h, f)
        # matmul_pf_2(
        #     score_softmax.data_src, indptr_1, indptr_1,
        #     v.data_src, indptr_1,
        #     z_data_, indptr_1,
        #     ) # z_ = ragged_matmul(score_softmax, v)
        
        # # z_before_merge: R.RaggedTensor([r_b, r_n], [r_h], [r_f])
        # z_before_merge_data = R.Tensor(w*h, f)
        # transpose_pf_2(
        #     z_data_, indptr_1,
        #     z_before_merge.data_src, indptr_0)
        #     # R.transpose(z_, [0, 2, 1, 3]) # z_before_merge =  ragged_tranpose_tir
        # z_merge: R.Tensor(w, h*f) = R.reshape(w, h*f)
        
        # z: R.Tensor(w, h*f) = R.matmul(
        #     z_merge, w_z)
        
        # # residual 
        # res: R.Tensor(w, h*f) = R.add(
        #     z, 
        #     data_x)

        # return res

def run_cpu(mod, func_name, *input):
    target = tvm.target.Target("llvm")
    mod = OperatorLegalizer(mod).transform()
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*input)

def transformer():
    batch_size = 2
    lens = [3, 4]
    ind_ptr = [0, 3, 7]
    total_len = sum(lens)
    embedding_size = 3
    k_size = embedding_size
    head_num = 2
    g = embedding_size*head_num
    np.random.seed(0)

    shape_holder_np = np.random.rand(
        batch_size,
        embedding_size,
        head_num).astype("float32")
    data_np = np.random.rand(
        total_len, g).astype("float32")
    ind_np = np.asarray(
        ind_ptr).astype("int32")

    w_qkv_np = np.random.rand(
        3,
        g,
        embedding_size,
        head_num).astype("float32")
    w_z_np = np.random.rand(
        g,
        g).astype("float32")
    f_scalar_np = np.array([embedding_size]).astype("float32")
    
    shape_holder = tvm.nd.array(shape_holder_np)
    data = tvm.nd.array(data_np)
    ind = tvm.nd.array(ind_np)
    w_qkv = tvm.nd.array(w_qkv_np)
    w_z = tvm.nd.array(w_z_np)
    f_scalar = tvm.nd.array(f_scalar_np)

    res = run_cpu(Transformer, "transformer_block", 
        shape_holder, data, ind, w_qkv, w_z, f_scalar)
    print(res)
    # np.testing.assert_allclose(qkv.numpy(), expected_qkv, rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(res.numpy(), expected_res, rtol=1e-05, atol=1e-05)
    
    # qkv, q, k, v = run_cpu(Transformer, "transformer_block", data, w_qkv)
    # np.testing.assert_allclose(qkv.numpy(), expected_qkv, rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(q.numpy(), e_q, rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(k.numpy(), e_k, rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(v.numpy(), e_v, rtol=1e-05, atol=1e-05)



if __name__ == "__main__":
    transformer()