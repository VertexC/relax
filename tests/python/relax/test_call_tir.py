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
import tvm.testing

@I.ir_module
class Module:
    @T.prim_func
    def add(var_rxplaceholder: T.handle, var_B: T.handle, m: T.int64):
        T.func_attr({"global_symbol": "add", "tir.noalias": True})
        rxplaceholder = T.match_buffer(var_rxplaceholder, [m], dtype="float32")
        B = T.match_buffer(var_B, [m], dtype="float32")
        for i0 in T.serial(m):
            with T.block("B"):
                i0_1 = T.axis.spatial(m, i0)
                T.reads(rxplaceholder[i0_1])
                T.writes(B[i0_1])
                B[i0_1] = rxplaceholder[i0_1] + T.float32(1)
        
    @R.function
    def main(x: R.Tensor(("n",), "float32")):
        gv = R.call_tir(add, (x,), (n,), dtype="float32", tir_vars=(n,))
        return gv


mod = Module
print(mod.script())
func = tvm.build(mod["add"], target="llvm")

dev = tvm.cpu(0)
exec = relax.vm.build(mod, "llvm")
vm = relax.VirtualMachine(exec, dev)

a_np = np.random.rand(10).astype("float32")
b_np = a_np + 1
a_tvm = tvm.nd.array(a_np, dev)

b_tvm = vm["main"](a_tvm)

tvm.testing.assert_allclose(b_tvm.numpy(), b_np, rtol=1e-6, atol=1e-6)
print("Numerical comparison passed")
