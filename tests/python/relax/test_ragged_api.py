
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

import numpy as np


b = tvm.tir.Var("b", "int64")
b_r = relax.expr.RaggedDim("b_r", False, b, None, None)
# print(b_r)

ind_ptr = relax.Var("ind_ptr", [b+1,], relax.DynTensorType(1, "int64"))

l_r = relax.expr.RaggedDim("l_r", True, None, b_r, ind_ptr)
# print(l_r)

l0 = relax.expr.RaggedLayoutExpr([b_r, l_r], [[0, 1]])
# print(l0) # no ExprFunctor

# rt = relax.Var("rt", l0, relax.RaggedDynTensorType(2, "float32"))
g = tvm.tir.Var("g", "int64")
h = tvm.tir.Var("h", "int64")
k = tvm.tir.Var("k", "int64")
data_ptr = relax.Var("rt_data", [g, h], relax.DynTensorType(2, "float32"))
rt = relax.nn.ragged_tensor_pack(l0, data_ptr, 2)
# print(rt)

w = relax.Var("w", [h, k], relax.DynTensorType(2, "float32"))

res = relax.nn.ragged_matmul(rt, w)
# print(res