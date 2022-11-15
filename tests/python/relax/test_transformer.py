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
            x: R.Tensor(("f", "h"), "float32"), 
            x_pos: R.Tensor(("b", "n", f*h), "float32"),
            w_qkv: R.Tensor((3, f*h, f, h), "float32"),
            w_z: R.Tensor((h*f, h*f), "float32"),
            gamma: R.Tensor((h*f, ), "float32"),
            beta: R.Tensor((h*f, ), "float32"),
            f_scalar: R.Tensor((1,), "float32"),
            ):
        
        lv0: R.Tensor((h*f, 3, f, h), "float32") = R.transpose(w_qkv, [1, 0, 2, 3])
        lv1: R.Tensor((h*f, f*h*3), "float32") = R.reshape(lv0, (h*f, f*h*3))

        qkv: R.Tensor((b, n, f*h*3), "float32") = R.matmul(x_pos, lv1)

        qkv_split = R.split(qkv, 3, axis=2)
    
        q_0: R.Tensor((b, n, f*h), "float32") = relax.TupleGetItem(qkv_split, 0)
        q_1 = R.reshape(q_0, (b, n, f, h))
        q: R.Tensor((b, h, n, f), "float32") = R.transpose(q_1, [0, 3, 1, 2])
        
        k_0: R.Tensor((b, n, f*h), "float32") = relax.TupleGetItem(qkv_split, 1)
        k_1 = R.reshape(k_0, (b, n, f, h))
        k: R.Tensor((b, h, n, f), "float32") = R.transpose(k_1, [0, 3, 1, 2])

        v_0: R.Tensor((b, n, f*h), "float32") = relax.TupleGetItem(qkv_split, 2)
        v_1 = R.reshape(v_0, (b, n, f, h))
        v: R.Tensor((b, h, n, f), "float32") = R.transpose(v_1, [0, 3, 1, 2])

        k_t: R.Tensor((b, h, f, n), "float32") = R.transpose(k, [0, 1, 3, 2])
        # q @ k_t

        score: R.Tensor((b, h, n, n), "float32") = R.matmul(q, k_t)
        
        scale = R.sqrt(f_scalar)
        score_scaled: R.Tensor((b, h, n, n), "float32") = R.divide(score, scale)
        score_softmax: R.Tensor((b, h, n, n), "float32") = R.softmax(score_scaled, axis=3)

        # score @ v
        z_: R.Tensor((b, h, n, f), "float32") = R.matmul(score_softmax, v)
        z_before_merge: R.Tensor((b, n, h, f), "float32") = R.transpose(z_, [0, 2, 1, 3])
        z_merge: R.Tensor((b, n, h*f), "float32") = R.reshape(z_before_merge, (b, n, h*f))
        z: R.Tensor((b, n, h*f), "float32") = R.matmul(z_merge, w_z)
        
        # residual 
        res = R.add(z, x_pos)

        # layernorm(x)
        ln: R.Tensor((b, n, h*f), "float32") = R.layer_norm(res, gamma, beta, axis=[-1])

        return score_softmax

def run_cpu(mod, func_name, *input):
    target = tvm.target.Target("llvm")
    mod = OperatorLegalizer(mod).transform()
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*input)

def expected_attention(data, w_qkv, w_z, b, n, f, h):
    g = f*h

    lv0 = np.transpose(w_qkv, [1, 0, 2, 3])
    lv1 = np.reshape(lv0, (g, -1))
    qkv = np.matmul(data, lv1)
    # q, k, v
    q, k, v = np.split(qkv, 3, axis=2)
    qkv_func = lambda x:  np.transpose(np.reshape(x, (b, n, f, h)), [0, 3, 1, 2])
    q = qkv_func(q)
    k = qkv_func(k)
    v = qkv_func(v)
    score = np.matmul(q, np.transpose(k, [0, 1, 3, 2]))
    score_scaled = np.divide(score, f)
    s_max = np.max(score_scaled, axis=-1).reshape(b, h, n, 1)
    s_exp = np.exp(score_scaled-s_max)
    score_softmax = s_exp / (np.sum(s_exp, axis=-1).reshape(b, h, n, 1))
    z_ = np.matmul(score_softmax, v)
    z_before_merge = np.transpose(z_, [0, 2, 1, 3])
    z_merge = np.reshape(z_before_merge, (b, n, h*f))
    z = np.matmul(z_merge, w_z)

    res = z + data
    return score_softmax

def transformer():
    batch_size = 2
    max_length = 4
    embedding_size = 16
    k_size = embedding_size
    head_num = 8
    g = embedding_size*head_num


    shape_holder_np = data_np = np.random.rand(
        embedding_size,
        head_num).astype("float32")
    data_np = np.random.rand(
        batch_size,
        max_length, 
        g).astype("float32")
    w_qkv_np = np.random.rand(
        3,
        g,
        embedding_size,
        head_num).astype("float32")
    w_z_np = np.random.rand(
        g,
        g).astype("float32")
    gamma_np  = np.random.rand(g, ).astype("float32")
    beta_np  = np.random.rand(g, ).astype("float32")
    f_scalar_np = np.array([embedding_size]).astype("float32")
    
    expected_res = expected_attention(data_np, w_qkv_np, w_z_np,
        batch_size, max_length, embedding_size, head_num)

    shape_holder = tvm.nd.array(shape_holder_np)
    data = tvm.nd.array(data_np)
    w_qkv = tvm.nd.array(w_qkv_np)
    w_z = tvm.nd.array(w_z_np)
    gamma = tvm.nd.array(gamma_np)
    beta = tvm.nd.array(beta_np)
    f_scalar = tvm.nd.array(f_scalar_np)

    res = run_cpu(Transformer, "transformer_block", 
        shape_holder, data, w_qkv, w_z, gamma, beta, f_scalar)
    # np.testing.assert_allclose(qkv.numpy(), expected_qkv, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(res.numpy(), expected_res, rtol=1e-05, atol=1e-05)
    
    # qkv, q, k, v = run_cpu(Transformer, "transformer_block", data, w_qkv)
    # np.testing.assert_allclose(qkv.numpy(), expected_qkv, rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(q.numpy(), e_q, rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(k.numpy(), e_k, rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(v.numpy(), e_v, rtol=1e-05, atol=1e-05)



if __name__ == "__main__":
    transformer()