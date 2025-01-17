/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "nn.h"
#include "../make_op.h"

namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(DenseAttrs);

RELAX_REGISTER_OP("relax.nn.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<DenseAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight", "2D Tensor", "Weight matrix.")
    .set_attr<FInferShape>("FInferShape", InferShapeDense)
    .set_attr<FInferType>("FInferType", InferTypeDense);

Expr MakeDense(Expr data, Expr weight, PrimExpr units, DataType out_dtype) {
  auto attrs = make_object<DenseAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.nn.dense");

  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.dense").set_body_typed(MakeDense);
TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

Expr MakeSoftmax(Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("relax.nn.softmax");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAX_REGISTER_OP("relax.nn.softmax")
    .describe(R"code(Softmax layer.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeUnaryBroadcast)
    .set_attr<FInferType>("FInferType", InferTypeUnaryBroadcast);

TVM_REGISTER_GLOBAL("relax.op.nn.softmax").set_body_typed(MakeSoftmax);

/* relax.nn.relu */
RELAX_REGISTER_UNARY_OP("nn.relu");

/* relax.nn.gelu */
RELAX_REGISTER_UNARY_OP("nn.gelu");

/* relax.nn.silu */
RELAX_REGISTER_UNARY_OP("nn.silu");

RELAX_REGISTER_OP("relax.nn.flatten")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferShape>("FInferShape", InferShapeFlatten)
    .set_attr<FInferType>("FInferType", InferTypeFlatten);

Expr MakeFlatten(Expr data) {
  static const Op& op = Op::Get("relax.nn.flatten");
  return Call(op, {data}, {}, {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.flatten").set_body_typed(MakeFlatten);

/* relax.nn.batch_norm */
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

RELAX_REGISTER_OP("relax.nn.batch_norm")
    .set_attrs_type<BatchNormAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .add_argument("moving_mean", "Tensor", "Running mean of input.")
    .add_argument("moving_var", "Tensor", "Running variance of input.")
    .set_attr<FInferShape>("FInferShape", InferShapeBatchNorm)
    .set_attr<FInferType>("FInferType", InferTypeBatchNorm);

Expr MakeBatchNorm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,  //
                   int axis, double epsilon, bool center, bool scale) {
  ObjectPtr<BatchNormAttrs> attrs = make_object<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op& op = Op::Get("relax.nn.batch_norm");
  return Call(op,
              {std::move(data), std::move(gamma), std::move(beta), std::move(moving_mean),
               std::move(moving_var)},
              Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.batch_norm").set_body_typed(MakeBatchNorm);

Optional<Expr> InferShapeBatchNorm(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 5) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "BatchNorm op should have 5 arguments, but only " << call->args.size()
                       << "are get.");
  }

  const auto* data_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* mean_shape = call->args[3]->shape().as<ShapeExprNode>();
  const auto* var_shape = call->args[4]->shape().as<ShapeExprNode>();
  if (data_shape == nullptr || mean_shape == nullptr || var_shape == nullptr) {
    return NullOpt;
  }

  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  const auto* gamma_shape = call->args[1]->shape().as<ShapeExprNode>();
  const auto* beta_shape = call->args[2]->shape().as<ShapeExprNode>();
  if (attrs->scale && gamma_shape == nullptr) {
    return NullOpt;
  }
  if (attrs->center && beta_shape == nullptr) {
    return NullOpt;
  }

  return Tuple(
      {GetRef<ShapeExpr>(data_shape), GetRef<ShapeExpr>(mean_shape), GetRef<ShapeExpr>(var_shape)});
}

Type InferTypeBatchNorm(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 5) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "BatchNorm op should have 5 arguments, but only " << call->args.size()
                       << "are get.");
  }

  const auto* data_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* gamma_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  const auto* beta_type = call->args[2]->checked_type().as<DynTensorTypeNode>();
  const auto* mean_type = call->args[3]->checked_type().as<DynTensorTypeNode>();
  const auto* var_type = call->args[4]->checked_type().as<DynTensorTypeNode>();

  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  int axis = attrs->axis;

  if (data_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (data_type->ndim <= axis) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op axis is " << axis << " while the input data tensor only has "
                       << data_type->ndim << " dimensions. Please make sure `axis` is in range [0, "
                       << data_type->ndim << ").");
  }
  if (mean_type == nullptr) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The op input moving mean should has type DynTensorType, but actually it is "
        << call->args[3]->checked_type()->GetTypeKey()
        << ". Please make sure the input has type DynTensorType.");
  } else if (mean_type->ndim != 1 && mean_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input mean should be a 1-dim tensor, while the actual input mean has "
        << mean_type->ndim << " dimensions.");
  }
  if (var_type == nullptr) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The op input moving variance should has type DynTensorType, but actually it is "
        << call->args[4]->checked_type()->GetTypeKey()
        << ". Please make sure the input has type DynTensorType.");
  } else if (var_type->ndim != 1 && var_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input variance should be a 1-dim tensor, while the actual input variance has "
        << var_type->ndim << " dimensions.");
  }
  if (gamma_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input gamma should has type DynTensorType, but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (gamma_type->ndim != 1 && gamma_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input gamma should be a 1-dim tensor, while the actual input gamma has "
        << gamma_type->ndim << " dimensions.");
  }
  if (beta_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input beta should has type DynTensorType, but actually it is "
                       << call->args[2]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (beta_type->ndim != 1 && beta_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input beta should be a 1-dim tensor, while the actual input beta has "
        << beta_type->ndim << " dimensions.");
  }

  return TupleType({GetRef<DynTensorType>(data_type), GetRef<DynTensorType>(mean_type),
                    GetRef<DynTensorType>(var_type)});
  // Todo(ruihang): how to do dtype broadcasting?
}

/* relax.nn.dropout */
TVM_REGISTER_NODE_TYPE(DropoutAttrs);

RELAX_REGISTER_OP("relax.nn.dropout")
    .set_attrs_type<DropoutAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "Input to which dropout will be applied.")
    .set_attr<FInferShape>("FInferShape", InferShapeDropout)
    .set_attr<FInferType>("FInferType", InferTypeDropout);

Expr MakeDropout(Expr data, double rate) {
  ObjectPtr<DropoutAttrs> attrs = make_object<DropoutAttrs>();
  attrs->rate = rate;

  static const Op& op = Op::Get("relax.nn.dropout");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.dropout").set_body_typed(MakeDropout);

Optional<Expr> InferShapeDropout(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Dropout op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  if (shape == nullptr) {
    return Tuple({RuntimeDepShape(), RuntimeDepShape()});
  }

  return Tuple({GetRef<ShapeExpr>(shape), GetRef<ShapeExpr>(shape)});
}

Type InferTypeDropout(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Dropout op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }

  return TupleType({GetRef<DynTensorType>(input_type), GetRef<DynTensorType>(input_type)});
}

/* relax.nn.layer_norm */
TVM_REGISTER_NODE_TYPE(LayerNormAttrs);

RELAX_REGISTER_OP("relax.nn.layer_norm")
    .set_attrs_type<LayerNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_attr<FInferShape>("FInferShape", InferShapeLayerNorm)
    .set_attr<FInferType>("FInferType", InferTypeLayerNorm);

Expr MakeLayerNorm(Expr data, Expr gamma, Expr beta, Array<Integer> axis, double epsilon,
                   bool center, bool scale) {
  ObjectPtr<LayerNormAttrs> attrs = make_object<LayerNormAttrs>();
  attrs->axis = std::move(axis);
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op& op = Op::Get("relax.nn.layer_norm");
  return Call(op, {std::move(data), std::move(gamma), std::move(beta)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.layer_norm").set_body_typed(MakeLayerNorm);

Optional<Expr> InferShapeLayerNorm(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "LayerNorm op should have 3 arguments");
  }

  const auto* data_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* gamma_shape = call->args[1]->shape().as<ShapeExprNode>();
  const auto* beta_shape = call->args[2]->shape().as<ShapeExprNode>();

  const auto* attrs = call->attrs.as<LayerNormAttrs>();

  int n_axis = attrs->axis.size();
  if (gamma_shape != nullptr && static_cast<int>(gamma_shape->values.size()) != n_axis) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "LayerNorm operator expects the input gamma to have the same rank as the "
                          "number of input axes. However, the given gamma has rank "
                       << gamma_shape->values.size() << " while the number of given axes is "
                       << n_axis);
  }
  if (beta_shape != nullptr && static_cast<int>(beta_shape->values.size()) != n_axis) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "LayerNorm operator expects the input beta to have the same rank as the "
                          "number of input axes. However, the given beta has rank "
                       << gamma_shape->values.size() << " while the number of given axes is "
                       << n_axis);
  }

  arith::Analyzer ana;
  if (data_shape == nullptr) {
    if (gamma_shape != nullptr && beta_shape != nullptr) {
      for (int i = 0; i < n_axis; ++i) {
        if (ana.CanProve(gamma_shape->values[i] != beta_shape->values[i])) {
          diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                             << "LayerNorm expects the input gamma and beta to have the same "
                                "shape. However, the given gamma and beta shapes differ on dim "
                             << i);
        }
      }
    }
    return RuntimeDepShape();
  }

  int ndim = data_shape->values.size();
  for (int i = 0; i < n_axis; ++i) {
    int dim = attrs->axis[i]->value;
    if (dim < 0) {
      dim = ndim + dim;
    }
    if (dim < 0 || dim >= ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "LayerNorm expects all the input axis indices are in range [-" << ndim
                         << ", " << ndim << "). However, the given axis index " << i << " is "
                         << attrs->axis[i]->value);
    }
    if (gamma_shape != nullptr && ana.CanProve(gamma_shape->values[i] != data_shape->values[dim])) {
      diag_ctx.EmitFatal(
          Diagnostic::Error(call->span)
          << "LayerNorm expects the input gamma to have compatible shape with the input data with "
             "regard to the input axis indices. However, the gamma dimension "
          << i << " has length " << gamma_shape->values[i] << " while the data dimension " << dim
          << " has length " << data_shape->values[dim]);
    }
    if (beta_shape != nullptr && ana.CanProve(beta_shape->values[i] != data_shape->values[dim])) {
      diag_ctx.EmitFatal(
          Diagnostic::Error(call->span)
          << "LayerNorm expects the input beta to have compatible shape with the input data with "
             "regard to the input axis indices. However, the beta dimension "
          << i << " has length " << beta_shape->values[i] << " while the data dimension " << dim
          << " has length " << data_shape->values[dim]);
    }
  }

  return GetRef<ShapeExpr>(data_shape);
}

Type InferTypeLayerNorm(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 3) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "LayerNorm op should have 3 arguments");
  }

  const auto* data_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* gamma_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  const auto* beta_type = call->args[2]->checked_type().as<DynTensorTypeNode>();
  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  int n_axis = attrs->axis.size();

  if (data_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "LayerNorm operator expects the input data to have type DynTensorType, "
                          "but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }
  if (gamma_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "LayerNorm operator expects the input gamma to have type DynTensorType, "
                          "but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (!gamma_type->IsUnknownNdim() && gamma_type->ndim != n_axis) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "LayerNorm operator expects the input gamma to have the same rank as the "
                          "number of input axes. However, the given gamma has rank "
                       << gamma_type->ndim << " while the number of given axes is " << n_axis);
  }
  if (beta_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "LayerNorm operator expects the input beta to have type DynTensorType, "
                          "but actually it is "
                       << call->args[2]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (!beta_type->IsUnknownNdim() && beta_type->ndim != n_axis) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "LayerNorm operator expects the input beta to have the same rank as the "
                          "number of input axes. However, the given beta has rank "
                       << beta_type->ndim << " while the number of given axes is " << n_axis);
  }

  return GetRef<DynTensorType>(data_type);
}

/* relax.nn.ragged_pack */
TVM_REGISTER_NODE_TYPE(RaggedTensorPackAttrs);

RELAX_REGISTER_OP("relax.nn.ragged_tensor_pack")
    .set_num_inputs(3)
    .add_argument("layout", "Layout", "")
    .add_argument("data_ptr", "Data", "")
    .add_argument("ndim", "Dims", "")
    .set_attr<FInferShape>("FInferShape", InferShapeRaggedTensorPack)
    .set_attr<FInferType>("FInferType", InferTypeRaggedTensorPack);

Expr MakeRaggedTensorPack(Expr layout, Var data_ptr, int ndim, DataType out_dtype) {
  LOG(INFO) << "ragged_tensor_pack " << layout;
  ObjectPtr<RaggedTensorPackAttrs> attrs = make_object<RaggedTensorPackAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->ndim = ndim;

  static const Op& op = Op::Get("relax.nn.ragged_tensor_pack");
  LOG(INFO) << layout;
  return Call(op, {std::move(layout), std::move(data_ptr)}, Attrs(attrs), {});
}

Optional<Expr> InferShapeRaggedTensorPack(const Call& call, DiagnosticContext diag_ctx) {
  return call->args[0];
}

Type InferTypeRaggedTensorPack(const Call& call, DiagnosticContext diag_ctx) {
  const auto* attrs = call->attrs.as<RaggedTensorPackAttrs>();
  return RaggedDynTensorType(attrs->ndim, attrs->out_dtype);
}

TVM_REGISTER_GLOBAL("relax.op.nn.ragged_tensor_pack").set_body_typed(MakeRaggedTensorPack);


/* relax.nn.ragged_matmul */
TVM_REGISTER_NODE_TYPE(RaggedMatmulAttrs);

RELAX_REGISTER_OP("relax.nn.ragged_matmul")
    .set_num_inputs(2)
    .add_argument("a", "Tensor", "The left operand of the matmul.")
    .add_argument("b", "Tensor", "The right operand of the matmul.")
    .set_attr<FInferShape>("FInferShape", InferShapeRaggedMatmul)
    .set_attr<FInferType>("FInferType", InferTypeRaggedMatmul);

/*
  Current raggedmatul only support left as ragged tensor (ndims = 3, 
    the right most dim is dense), 
  right as dense tensor, (ndim = 2)
*/
Expr MakeRaggedMatmul(Expr a, Expr b, DataType out_dtype) {
  ObjectPtr<RaggedMatmulAttrs> attrs = make_object<RaggedMatmulAttrs>();
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.nn.ragged_matmul");
  return Call(op, {std::move(a), std::move(b)}, Attrs(attrs), {});
}

Optional<Expr> InferShapeRaggedMatmul(const Call& call, DiagnosticContext diag_ctx) {
  const auto* a_layout_expr = call->args[0]->shape().as<RaggedLayoutExprNode>();
  const auto* b_shape_expr = call->args[1]->shape().as<ShapeExprNode>();
  LOG(INFO) << a_layout_expr->dims;
  Array<RaggedDim> a_layout = a_layout_expr->dims;
  Array<PrimExpr> b_shape = b_shape_expr->values;

  Array<RaggedDim> out_layout;
  int a_ndim = a_layout.size();

  for(int i=0; i < a_ndim - 1; i++) {
    out_layout.push_back(a_layout[i]);
  }
  // build a RaggedDim
  // TODO (bowenc): check if b_shape is Var, if it is, reuse the name
  RaggedDim dim = RaggedDim("test", false, b_shape[1], NullOpt, NullOpt);
  out_layout.push_back(dim);
  return RaggedLayoutExpr(out_layout, a_layout_expr->group);
}

Type InferTypeRaggedMatmul(const Call& call, DiagnosticContext diag_ctx) {
  const auto* a_type = call->args[0]->checked_type().as<RaggedDynTensorTypeNode>();
  int a_ndim = a_type->ndim;

  return RaggedDynTensorType(a_ndim, a_type->dtype);
}

TVM_REGISTER_GLOBAL("relax.op.nn.ragged_matmul").set_body_typed(MakeRaggedMatmul);


/* relax.nn.matmul */
TVM_REGISTER_NODE_TYPE(MatmulAttrs);

RELAX_REGISTER_OP("relax.nn.matmul")
    .set_num_inputs(2)
    .add_argument("a", "Tensor", "The left operand of the matmul.")
    .add_argument("b", "Tensor", "The right operand of the matmul.")
    .set_attr<FInferShape>("FInferShape", InferShapeMatmul)
    .set_attr<FInferType>("FInferType", InferTypeMatmul);

Expr MakeMatmul(Expr a, Expr b, DataType out_dtype) {
  ObjectPtr<MatmulAttrs> attrs = make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.nn.matmul");
  return Call(op, {std::move(a), std::move(b)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.matmul").set_body_typed(MakeMatmul);

Optional<Expr> InferShapeMatmul(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Matmul operator should have 2 arguments");
  }
  const auto* a_shape_expr = call->args[0]->shape().as<ShapeExprNode>();
  const auto* b_shape_expr = call->args[1]->shape().as<ShapeExprNode>();
  if (a_shape_expr == nullptr || b_shape_expr == nullptr) {
    return RuntimeDepShape();
  }

  Array<PrimExpr> a_shape = a_shape_expr->values;
  Array<PrimExpr> b_shape = b_shape_expr->values;
  int a_ndim = a_shape.size();
  int b_ndim = b_shape.size();

  if (a_ndim == 0 || b_ndim == 0) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "Matmul requires both operands to be have at lease one dimension. However, the operand `"
        << (a_ndim == 0 ? "a" : "b") << "` has zero dimension.");
  }

  bool a_prepended = false;
  bool b_appended = false;
  if (a_ndim == 1) {
    a_shape.insert(a_shape.begin(), tir::make_const(DataType::Int(32), 1));
    a_ndim = 2;
    a_prepended = true;
  }
  if (b_ndim == 1) {
    b_shape.insert(b_shape.end(), tir::make_const(DataType::Int(32), 1));
    b_ndim = 2;
    b_appended = true;
  }

  bool is_a_larger = a_ndim > b_ndim;
  int offset = is_a_larger ? a_ndim - b_ndim : b_ndim - a_ndim;
  int output_ndim = is_a_larger ? a_ndim : b_ndim;
  Array<PrimExpr> output_shape;
  output_shape.reserve(output_ndim);
  for (int i = 0; i < offset; ++i) {
    output_shape.push_back(is_a_larger ? a_shape[i] : b_shape[i]);
  }

  arith::Analyzer ana;
  for (int i = 0; i < output_ndim - offset - 2; ++i) {
    int a_idx, b_idx;
    if (is_a_larger) {
      a_idx = i + offset;
      b_idx = i;
    } else {
      a_idx = i;
      b_idx = i + offset;
    }
    PrimExpr a_dim = a_shape[a_idx];
    PrimExpr b_dim = b_shape[b_idx];
    if (is_a_larger) {
      a_dim = a_shape[i + offset];
      b_dim = b_shape[i];
    } else {
      a_dim = a_shape[i];
      b_dim = b_shape[i + offset];
    }

    if (EqualConstInt(a_dim, 1)) {
      output_shape.push_back(b_dim);
    } else if (EqualConstInt(b_dim, 1)) {
      output_shape.push_back(a_dim);
    } else if (EqualCheck(a_dim, b_dim)) {
      output_shape.push_back(a_dim);
    } else if (ana.CanProve(a_dim != b_dim)) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Matmul expects the input tensors to have broadcastable shapes. "
                            "However, the shape of `a` at dim "
                         << a_idx << " (which is " << a_dim
                         << ") is not compatible with the shape of `b` at dim " << b_idx
                         << " (which is " << b_dim << ")");
    } else {
      // Todo(ruihang): refine this point
      // defer the computation of output shapes to runtime
      // e.g., broadcast Tensor([m, n]), Tensor([k]) -> defer to runtime
      return Call(ExternFunc(String("vm.binary_broadcast_shape_infer")),
                  {call->args[0], call->args[1]}, {}, {});
    }
  }

  if (ana.CanProve(a_shape[a_ndim - 1] != b_shape[b_ndim - 2])) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "Matmul expects the last two dimensions of both operands to be matmul-compatible. "
           "However, the last dimension of `a` is "
        << a_shape[a_ndim - 1]
        << ", which is incompatible with the last but one dimension of `b`, which is "
        << b_shape[b_ndim - 2]);
  }
  // Todo(ruihang): if cannot prove equal, do runtime inference.
  if (!a_prepended) {
    output_shape.push_back(a_shape[a_ndim - 2]);
  }
  if (!b_appended) {
    output_shape.push_back(b_shape[b_ndim - 1]);
  }

  return ShapeExpr(output_shape);
}

Type InferTypeMatmul(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Matmul operator should have 2 arguments");
  }

  const auto* a_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* b_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  const auto* attrs = call->attrs.as<MatmulAttrs>();
  if (a_type == nullptr || b_type == nullptr) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "Matmul expects both operands to have type DynTensorType. However, the operand `"
        << (a_type == nullptr ? "a" : "b") << "` has type "
        << call->args[a_type == nullptr ? 0 : 1]->checked_type()->GetTypeKey());
  }

  DataType output_dtype;
  if (a_type->IsUnknownDtype() || b_type->IsUnknownDtype()) {
    output_dtype = attrs->out_dtype;
  } else if (a_type->dtype != b_type->dtype && attrs->out_dtype.is_void()) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Matmul expects both operands to have the same data type when there is "
                          "no specified output dtype. However, operand `a` has dtype "
                       << a_type->dtype << " while `b` has dtype " << b_type->dtype);
  } else {
    output_dtype = attrs->out_dtype.is_void() ? a_type->dtype : attrs->out_dtype;
  }

  int a_ndim = a_type->ndim;
  int b_ndim = b_type->ndim;

  if (a_ndim == 0 || b_ndim == 0) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "Matmul requires both operands to be have at lease one dimension. However, the operand `"
        << (a_ndim == 0 ? "a" : "b") << "` has zero dimension.");
  }
  if (a_type->IsUnknownNdim() || b_type->IsUnknownNdim()) {
    return DynTensorType(-1, output_dtype);
  }

  bool a_prepended = false;
  bool b_appended = false;
  if (a_ndim == 1) {
    a_ndim = 2;
    a_prepended = true;
  }
  if (b_ndim == 1) {
    b_ndim = 2;
    b_appended = true;
  }
  int output_ndim =
      std::max(a_ndim, b_ndim) - static_cast<int>(a_prepended) - static_cast<int>(b_appended);

  return DynTensorType(output_ndim, output_dtype);
}

}  // namespace relax
}  // namespace tvm
