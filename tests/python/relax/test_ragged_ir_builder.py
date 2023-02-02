
from __future__ import annotations  # must import to defer parsing of annotations
import sys
import tempfile
import pytest
import tvm
from tvm import relax, tir, topi
from tvm._ffi.base import TVMError
from tvm.relax.transform import OperatorLegalizer

# from tvm.script import relax as R, tir as T

from tvm.script._parser import ir as I, relax as R, tir as T
from tvm.relax.testing.ast_printer import dump_ast
import numpy as np

b = tvm.tir.Var("b", "int64")
h = tvm.tir.Var("h", "int64")
b_r = relax.expr.RaggedDim("b_r", False, b, None, None)
h_r = relax.expr.RaggedDim("h_r", False, h, None, None)

ind_ptr = relax.Var("ind_ptr", [b+1,], relax.DynTensorType(1, "int64"))

g = tvm.tir.Var("g", "int64")
k = tvm.tir.Var("k", "int64")

rt_data = relax.Var("rt_data", [g, h], relax.DynTensorType(2, "float32"))
w = relax.Var("w", [h, k], relax.DynTensorType(2, "float32"))
 

bb = relax.BlockBuilder()
with bb.function("main", [rt_data, ind_ptr, w]):
  with bb.dataflow():
    l_r = relax.expr.RaggedDim("l_r", True, None, b_r, ind_ptr)
    l0 = relax.expr.RaggedLayoutExpr([b_r, l_r, h_r], [[0, 1], [2]])
    print(type(l0))
    rt = bb.emit(relax.nn.ragged_tensor_pack(l0, rt_data, 3, "float32"))
    c = bb.emit_output(relax.nn.ragged_matmul(rt, w))
  bb.emit_func_output(c)

mod = bb.get()
print(mod.script())
# func = mod["main"]

# print(dump_ast(func))


passes = [relax.transform.RaggedMatmulToDenseMatmul()]
seq = tvm.transform.Sequential(passes)
mod = seq(mod)

print(mod.script())

# print(func)
B = 3
l = [1, 2, 3]
ind_ptr = [0, 1, 3, 6]
G = sum(ind_ptr)
H = 12
K = 5

data_np = np.random.rand(
        G,
        H).astype("float32")
w_np = np.random.rand(
        H,
        K).astype("float32")
ind_ptr_np = np.asarray(
        ind_ptr, dtype="int64")
expected_res = np.matmul(data_np, w_np)

data = tvm.nd.array(data_np)
w = tvm.nd.array(w_np)
ind_ptr = tvm.nd.array(ind_ptr_np)


def run_cpu(mod, func_name, *input):
  target = tvm.target.Target("llvm")
  mod = OperatorLegalizer(mod).transform()
  # print(mod.script())
  ex = relax.vm.build(mod, target)
  vm = relax.VirtualMachine(ex, tvm.cpu())
  return vm[func_name](*input)

res = run_cpu(mod, "main", data, ind_ptr, w)
print(res)
np.testing.assert_allclose(res.numpy(), expected_res, rtol=1e-05, atol=1e-05)



