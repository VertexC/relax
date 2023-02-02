
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

b_r = relax.expr.RaggedDim("b_r", False, 1, None, None)

@I.ir_module
class RaggedMatmulExample:
  @R.function
  def main(
    x: R.Tensor(("b"), "float32"),
    rt_data: R.Tensor(("g", "h"), "float32"),
    w: R.Tensor((h, "k"), "float32"),
    ind_ptr: R.Tensor((b + 1, ), "int64")
  ):
    with R.dataflow():
      rt = relax.nn.ragged_tensor_pack(relax.expr.RaggedLayoutExpr([
        relax.expr.RaggedDim("b_r", False, b, None, None), 
        relax.expr.RaggedDim("h_r", False, h, None, None), 
        relax.expr.RaggedDim("l_r", True, None, b_r, ind_ptr)], [[0, 1], [2]]), rt_data, 3, "float32")
      out = relax.nn.ragged_matmul(rt, w, "float32")
      R.output(out)
    return out

@I.ir_module
class MatmulExample:
  @R.function
  def main(
    padded_x: R.Tensor(("b", "l", "h"), "float32"),
    w: R.Tensor((h, "k"), "float32"),
  ):
    out: R.Tensor((b, l, k), "float32") = relax.nn.matmul(padded_x, w, "float32")
    return out

import argparse
import timeit
parser = argparse.ArgumentParser()
parser.add_argument('f', type=str)
parser.add_argument('--b', type=int, default=8)
parser.add_argument('--h', type=int, default=8)
parser.add_argument('--k', type=int, default=8)

args = parser.parse_args()
print(args)

def chunks(lst, n, m):
    if m == -1:
        ext = len(lst)
    else:
        ext = min(m*n, len(lst))
    """Yield successive n-sized chunks from lst."""
    for i in range(0, ext, n):
        if i+n <= ext: yield np.array(lst[i:i + n], "int32")

def read_lengths(filename, skip = 0):
    data_lines = [int(line.strip()) for line in open(filename, "r", errors='replace')]
    return data_lines[skip:]

def read_and_chunk_lengths(batch_size, max_batches, lengths_file):
    data_lines = read_lengths(lengths_file)
    return list(chunks(data_lines, batch_size, max_batches)), max(data_lines)


B = args.b
H = args.h
K = args.k
file_name = args.f
# ls = read_lengths("race_sub.txt")
batch_lens, max_l = read_and_chunk_lengths(B, -1, file_name)
print(max_l)


def analysis():
  # flop : (mxp) x (pxn) = mn*(2p-1)
  dense_flop = B*max_l*K*(2*H-1)
  ragged_flops = []
  for batch_len in batch_lens:
    G = sum(batch_len)
    ragged_flop = G*K*(2*H-1)
    ragged_flops.append(ragged_flop) 
  avg_ragged_flops = sum(ragged_flops) / len(ragged_flops)
  print(dense_flop, avg_ragged_flops)
analysis()

def run_cpu_dense(mod, func_name, times=10):
  data_np = np.random.rand(B, max_l, H).astype("float32")
  w_np = np.random.rand(H, K).astype("float32")

  inputs_np = [data_np, w_np]
  inputs_tvm = [tvm.nd.array(d) for d in inputs_np]
  
  target = tvm.target.Target("llvm")
  mod = OperatorLegalizer(mod).transform()
  ex = relax.vm.build(mod, target)
  vm = relax.VirtualMachine(ex, tvm.cpu())
  # print(vm[func_name](*inputs_tvm))
  t = timeit.timeit(lambda : vm[func_name](*inputs_tvm), number=times)
  print(t)

run_cpu_dense(MatmulExample, "main")


def run_cpu_ragged(mod, func_name, times=10):
  # compile
  target = tvm.target.Target("llvm")
  print(mod)
  passes = [relax.transform.RaggedMatmulToDenseMatmul()]
  seq = tvm.transform.Sequential(passes)
  mod = seq(mod)
  mod = OperatorLegalizer(mod).transform()
  ex = relax.vm.build(mod, target)
  vm = relax.VirtualMachine(ex, tvm.cpu())

  # build ragged_data

  ts = []
  for batch_len in batch_lens:
    l = batch_len
    ind_ptr = [0]
    for i in batch_len:
      ind_ptr.append(i + ind_ptr[-1])
    G = sum(l)

    x_np = np.random.rand(B).astype("float32")
    data_np = np.random.rand(
            G,
            H).astype("float32")
    w_np = np.random.rand(
            H,
            K).astype("float32")
    ind_ptr_np = np.asarray(
            ind_ptr, dtype="int64")

    inputs_np = [x_np, data_np, w_np, ind_ptr_np]
    inputs_tvm = [tvm.nd.array(d) for d in inputs_np]

    t = timeit.timeit(lambda : vm[func_name](*inputs_tvm), number=times)
    # print(t)
    ts.append(t)
  print("avg:", sum(ts) / len(ts))
  
run_cpu_ragged(RaggedMatmulExample, "main")





