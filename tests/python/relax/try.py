import tvm
from tvm import te, tir, relax
import tvm.testing
import numpy as np

def te_add(A: te.Tensor) -> te.Tensor:
    # B = te.compute(A.shape, lambda i: A[i] + 1, "B")
    B = te.compute(A.shape, lambda *i: A(*i) + 1, "B")
    return B


n = tir.Var("n", "int64")
x = relax.Var("x", (n,), relax.DynTensorType(ndim=1, dtype="float32"))

bb = relax.BlockBuilder()
with bb.function("main", [x]):
    y = bb.emit_te(te_add, x)
    bb.emit_func_output(y)

mod = bb.get()
print(mod.script())

dev = tvm.cpu(0)
exec = relax.vm.build(mod, "llvm")
vm = relax.VirtualMachine(exec, dev)

a_np = np.random.rand(10).astype("float32")
b_np = a_np + 1
a_tvm = tvm.nd.array(a_np, dev)

b_tvm = vm["main"](a_tvm)

print("b_np", b_np)
print("b_tvm", b_tvm.numpy())
tvm.testing.assert_allclose(b_tvm.numpy(), b_np, rtol=1e-6, atol=1e-6)
print("Numerical comparison passed")