@I.ir_module
class MatmulExample:
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
        relax.expr.RaggedDim("l_r", True, None, b_r, ind_ptr)
        relax.expr.RaggedDim("h_r", False, h, None, None),], [[0, 1], [2]]), rt_data, 3, "float32")
      out = relax.nn.ragged_matmul(rt, w, "float32")
      R.output(out)
    return out

@autobatch("main", )
@I.ir_module
class MatmulExample:
  @R.function

@autobatch("main", ragged=True)
@tvm.script.ir_module
class Module:
    @R.function
    def main(a: Tensor(("l", "h"), "float32"), 
            b: Tensor((h, "k"), "float32")):
        out: Tensor((l, k), "float32") = relax.nn.matmul(a, b, out_dtype="float32")
        return out

@R.function
def main(rt_data: Tensor((g, h), "float32"), w: Tensor((h, k), "float32"), ind_ptr: Tensor(((b + 1),), "int64")) -> RaggedTensor(None, "float32", ndim = 3):
    # block 0
    with R.dataflow():
        rt: RaggedTensor((b_r[0], h_r[0], l_r[1]), "float32") = relax.nn.ragged_tensor_pack((b_r[0], h_r[0], l_r[1]), rt_data, out_dtype="float32", ndim=3, attrs_type_key="relax.attrs.RaggedTensorPackAttrs")
        out: RaggedTensor((b_r[0], h_r[0], test[0]), "float32") = relax.nn.ragged_matmul(rt, w, out_dtype="float32", attrs_type_key="relax.attrs.MatmulAttrs")
        R.output(out)
    return out