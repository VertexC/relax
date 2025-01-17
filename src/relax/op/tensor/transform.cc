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

/*!
 * \file transform.cc
 * \brief Transform operators.
 */

#include "transform.h"

#include <unordered_set>

namespace tvm {
namespace relax {

/* relax.transpose */
TVM_REGISTER_NODE_TYPE(TransposeAttrs);

RELAX_REGISTER_OP("relax.transpose")
    .set_attrs_type<TransposeAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "nD Tensor", "input tensor to be transposed")
    .set_attr<FInferShape>("FInferShape", InferShapeTranspose)
    .set_attr<FInferType>("FInferType", InferTypeTranspose);

Expr MakeTranspose(Expr data, Optional<Array<Integer>> axes) {
  ObjectPtr<TransposeAttrs> attrs = make_object<TransposeAttrs>();
  attrs->axes = std::move(axes);

  static const Op& op = Op::Get("relax.transpose");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.transpose").set_body_typed(MakeTranspose);

Optional<Expr> InferShapeTranspose(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Transpose op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<TransposeAttrs>();
  if (shape == nullptr) {
    return NullOpt;
  }

  int ndim = shape->values.size();
  if (attrs->axes.defined() && ndim != static_cast<int>(attrs->axes.value().size())) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Transpose op expects the input axis indices to be a permutation of 0 to "
                       << ndim - 1 << ". However, the length of the given indices is not " << ndim);
  }

  Array<PrimExpr> out_shape;
  std::unordered_set<int> used_axis;
  out_shape.resize(ndim);
  used_axis.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    int dim = attrs->axes.defined() ? attrs->axes.value()[i]->value : (ndim - i - 1);
    if (dim < 0) {
      dim = ndim + dim;
    }

    if (dim < 0 || dim >= ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Transpose expects all axis indices to be in range [-" << ndim << ", "
                         << ndim << "). However, the given indices on axis " << i << " is " << dim);
    }
    if (used_axis.count(dim)) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Transpose expects all axis indices not to duplicate. However the "
                            "given indices has duplicate "
                         << dim);
    }

    out_shape.Set(i, shape->values[dim]);
    used_axis.insert(dim);
  }
  return ShapeExpr(out_shape);
}

Type InferTypeTranspose(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Transpose op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }
  return GetRef<DynTensorType>(input_type);
}

/* relax.reshape */
RELAX_REGISTER_OP("relax.reshape")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("new_shape", "ShapeExpr", "The input new shape.")
    .set_attr<FInferShape>("FInferShape", InferShapeReshape)
    .set_attr<FInferType>("FInferType", InferTypeReshape);

Expr MakeReshape(Expr data, Expr new_shape) {
  static const Op& op = Op::Get("relax.reshape");
  return Call(op, {std::move(data), std::move(new_shape)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.reshape").set_body_typed(MakeReshape);

Optional<Expr> InferShapeReshape(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Reshape op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* new_shape = call->args[1].as<ShapeExprNode>();

  // If we have no knowledge on the input data shape or the input new shape, just return the input
  // new shape.
  if (shape == nullptr || new_shape == nullptr) {
    return call->args[1];
  }

  int ndim = shape->values.size();
  PrimExpr shape_prod = tir::make_const(tvm::DataType::Int(32), 1);
  for (int i = 0; i < ndim; ++i) {
    shape_prod = shape_prod * shape->values[i];
  }

  int dim_to_infer = -1;
  int new_ndim = new_shape->values.size();
  PrimExpr new_shape_prod = tir::make_const(tvm::DataType::Int(32), 1);
  arith::Analyzer ana;
  for (int i = 0; i < new_ndim; ++i) {
    PrimExpr dim_len = new_shape->values[i];
    if (ana.CanProveEqual(dim_len, tir::make_const(tvm::DataType::Int(32), -1))) {
      if (dim_to_infer != -1) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "Reshape op accepts at most one \"-1\" in the new shape. However, "
                              "the new shape on dimension "
                           << dim_to_infer << " and " << i << " are both \"-1\"");
      }
      dim_to_infer = i;
    } else if (ana.CanProveEqual(dim_len, tir::make_const(tvm::DataType::Int(32), 0))) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Reshape op does not accept \"0\" in the new shape. However, the new "
                            "shape on dimension "
                         << i << " is \"0\"");
    } else {
      new_shape_prod = new_shape_prod * dim_len;
    }
  }

  // Todo(ruihang): need a runtime reshape inference function

  Array<PrimExpr> new_shape_arr = new_shape->values;
  if (dim_to_infer != -1) {
    new_shape_arr.Set(dim_to_infer, shape_prod / new_shape_prod);
  }
  return ShapeExpr(new_shape_arr);
}

Type InferTypeReshape(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Reshape op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }
  const auto* new_shape_type = call->args[1]->checked_type().as<ShapeTypeNode>();
  if (new_shape_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The new shape of the reshape operator should has type ShapeTypeNode, "
                          "but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the new shape has type ShapeType.");
  }

  // Todo(ruihang): add ndim to ShapeType
  // return DynTensorType(new_shape_type->ndim, input_type->dtype);

  const auto* new_shape = call->args[1].as<ShapeExprNode>();
  if (new_shape != nullptr) {
    return DynTensorType(new_shape->values.size(), input_type->dtype);
  } else {
    return DynTensorType(-1, input_type->dtype);
  }
}

/* relax.expand_dims */
TVM_REGISTER_NODE_TYPE(ExpandDimsAttrs);

RELAX_REGISTER_OP("relax.expand_dims")
    .set_num_inputs(1)
    .set_attrs_type<ExpandDimsAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeExpandDims)
    .set_attr<FInferType>("FInferType", InferTypeExpandDims);

Expr MakeExpandDims(Expr data, Array<Integer> axis) {
  ObjectPtr<ExpandDimsAttrs> attrs = make_object<ExpandDimsAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.expand_dims");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.expand_dims").set_body_typed(MakeExpandDims);

Optional<Expr> InferShapeExpandDims(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "ExpandDims op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  if (shape == nullptr) {
    return NullOpt;
  }

  int ndim = shape->values.size();
  int n_new_dim = attrs->axis.size();
  int output_ndim = ndim + n_new_dim;

  Array<PrimExpr> output_shape;
  output_shape.resize(output_ndim);
  for (int i = 0; i < n_new_dim; ++i) {
    int dim = attrs->axis[i]->value;
    if (dim < 0) {
      dim = output_ndim + dim;
    }

    if (dim < 0 || dim >= output_ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The index \"" << attrs->axis[i]->value << "\" at the position " << i
                         << " of the new axis indices of operator ExpandDim is out of range.");
    }
    if (output_shape[dim].defined()) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The new axis indices of operator ExpandDim contain duplicate indices "
                            "- at least two indices refers to dim "
                         << dim << ". Please make sure the indices do not duplicate.");
    }
    output_shape.Set(dim, tvm::tir::make_const(tvm::DataType::Int(32), 1));
  }

  for (int i = 0, p = 0; i < output_ndim; ++i) {
    if (output_shape[i].defined()) {
      continue;
    }
    output_shape.Set(i, shape->values[p]);
    ++p;
  }

  return ShapeExpr(output_shape);
}

Type InferTypeExpandDims(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "ExpandDims op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }

  if (input_type->ndim == -1) {
    return GetRef<DynTensorType>(input_type);
  } else {
    return DynTensorType(input_type->ndim + attrs->axis.size(), input_type->dtype);
  }
}

/* relax.squeeze */
TVM_REGISTER_NODE_TYPE(SqueezeAttrs);

RELAX_REGISTER_OP("relax.squeeze")
    .set_num_inputs(1)
    .set_attrs_type<SqueezeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeSqueeze)
    .set_attr<FInferType>("FInferType", InferTypeSqueeze);

Expr MakeSqueeze(Expr data, Optional<Array<Integer>> axis) {
  ObjectPtr<SqueezeAttrs> attrs = make_object<SqueezeAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.squeeze");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.squeeze").set_body_typed(MakeSqueeze);

Optional<Expr> InferShapeSqueeze(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Squeeze op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  if (shape == nullptr) {
    return NullOpt;
  }

  int ndim = shape->values.size();
  arith::Analyzer ana;
  std::unordered_set<int> removed_axis;
  removed_axis.reserve(ndim);

  if (attrs->axis.defined()) {
    for (int i = 0; i < static_cast<int>(attrs->axis.value().size()); ++i) {
      int dim = attrs->axis.value()[i]->value;
      if (dim < 0) {
        dim = ndim + dim;
      }

      if (dim < 0 || dim >= ndim) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "The axis index \"" << attrs->axis.value()[i]->value
                           << "\" at the position " << i
                           << " of the axis indices of operator squeeze is out of range.");
      }
      if (ana.CanProve(shape->values[dim] != tir::make_const(DataType::Int(32), 1))) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "Squeeze expects all axis indices to correspond to axes with "
                              "dimension length 1. However, the input data on given axis index "
                           << dim << " is " << shape->values[i] << ", which cannot be 1.");
      }
      removed_axis.insert(dim);
    }
  } else {
    for (int i = 0; i < ndim; ++i) {
      bool is_one = ana.CanProveEqual(shape->values[i], tir::make_const(DataType::Int(32), 1));
      bool isnt_one = ana.CanProve(shape->values[i] != tir::make_const(DataType::Int(32), 1));
      if (!is_one && !isnt_one) {
        return NullOpt;
      } else if (is_one) {
        removed_axis.insert(i);
      }
    }
  }

  Array<PrimExpr> new_shape;
  new_shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    if (!removed_axis.count(i)) {
      new_shape.push_back(shape->values[i]);
    }
  }
  return ShapeExpr(new_shape);
}

Type InferTypeSqueeze(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Squeeze op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }

  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  if (attrs->axis.defined()) {
    if (input_type->ndim != -1) {
      return DynTensorType(input_type->ndim - attrs->axis.value().size(), input_type->dtype);
    } else {
      return GetRef<DynTensorType>(input_type);
    }
  } else {
    Optional<Expr> out_shape = InferShapeSqueeze(call, diag_ctx);
    if (out_shape.defined()) {
      const auto* shape = out_shape.value().as<ShapeExprNode>();
      ICHECK_NOTNULL(shape);
      return DynTensorType(shape->values.size(), input_type->dtype);
    } else {
      return DynTensorType(-1, input_type->dtype);
    }
  }
}

/* relax.concatenate */
TVM_REGISTER_NODE_TYPE(ConcatenateAttrs);

RELAY_REGISTER_OP("relax.concatenate")
    .set_attrs_type<ConcatenateAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input list of tensors.")
    .set_attr<FInferShape>("FInferShape", InferShapeConcatenate)
    .set_attr<FInferType>("FInferType", InferTypeConcatenate);

Expr MakeConcatenate(Expr data, Optional<Integer> axis) {
  ObjectPtr<ConcatenateAttrs> attrs = make_object<ConcatenateAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.concatenate");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.concatenate").set_body_typed(MakeConcatenate);

Optional<Expr> InferShapeConcatenate(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Concatenate op should have 1 argument");
  }

  const auto* tuple_shape = call->args[0]->shape().as<TupleNode>();
  if (tuple_shape == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Concatenate operator expects the input to be a tuple or a list of "
                          "tensors, indicating that the input shape should be a tuple of "
                          "ShapeExpr. However, the given input has shape "
                       << call->args[0]->shape()->GetTypeKey());
  }
  const auto* attrs = call->attrs.as<ConcatenateAttrs>();

  int n_tensor = tuple_shape->fields.size();
  if (n_tensor == 0) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Concatenate operator expects the input to have at least one tensor. "
                          "However, the given tensor tuple is empty");
  }

  int output_ndim = -1;
  arith::Analyzer ana;

  bool runtime_dep_shape = false;
  for (int i = 0; i < n_tensor; ++i) {
    const auto* shape = tuple_shape->fields[i].as<ShapeExprNode>();
    if (shape == nullptr) {
      if (!tuple_shape->fields[i]->IsInstance<RuntimeDepShapeNode>()) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "Invalid shape type " << tuple_shape->fields[i]->GetTypeKey());
      }
      runtime_dep_shape = true;
      continue;
    }

    if (!attrs->axis.defined() && shape->values.size() != 1) {
      diag_ctx.EmitFatal(
          Diagnostic::Error(call->span)
          << "Concatenate operator expects all input tensors to be 1-dim tensors when not given a "
             "specific concatenation axis. However, the input tensor "
          << i << " has " << shape->values.size() << " dimensions");
    }

    if (output_ndim == -1) {
      output_ndim = shape->values.size();
    } else if (static_cast<int>(shape->values.size()) != output_ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Concatenate operator expects all input tensors to have the same rank. "
                            "However, one input tensor has rank "
                         << output_ndim << " while another has rank " << shape->values.size());
    }
  }

  if (runtime_dep_shape) {
    return RuntimeDepShape();
  }
  ICHECK_NE(output_ndim, -1);

  int concat_axis = attrs->axis.defined() ? attrs->axis.value()->value : 0;
  if (concat_axis < 0) {
    concat_axis = output_ndim + concat_axis;
  }
  if (concat_axis < 0 || concat_axis >= output_ndim) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Concatenate operator expects the axis to be concatenated is in range ["
                       << -output_ndim << ", " << output_ndim
                       << "). However, the given axis index is " << attrs->axis.value()->value
                       << ", which is out of range");
  }

  Array<PrimExpr> output_shape;
  output_shape.reserve(output_ndim);

  for (int dim = 0; dim < output_ndim; ++dim) {
    if (dim == concat_axis) {
      PrimExpr concat_dim_len = tir::make_const(DataType::Int(32), 0);
      for (int i = 0; i < n_tensor; ++i) {
        PrimExpr dim_len = ana.Simplify(Downcast<ShapeExpr>(tuple_shape->fields[i])->values[dim]);
        concat_dim_len = concat_dim_len + dim_len;
      }
      output_shape.push_back(concat_dim_len);
    } else {
      int static_len = -1;
      PrimExpr symbolic_len{nullptr};
      bool runtime_dep_dim = false;
      for (int i = 0; i < n_tensor; ++i) {
        PrimExpr dim_len = ana.Simplify(Downcast<ShapeExpr>(tuple_shape->fields[i])->values[dim]);
        const int64_t* cur_dim_len = tir::as_const_int(dim_len);
        if (cur_dim_len != nullptr) {
          if (static_len == -1) {
            static_len = *cur_dim_len;
          } else if (*cur_dim_len != static_len) {
            diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                               << "Concatenate operator expects all input tensors to have the "
                                  "same shape except on the specified axis. However, the given "
                                  "tensors don't have the same dimension length on axis "
                               << dim);
          }
        } else if (!runtime_dep_dim) {
          if (!symbolic_len.defined()) {
            symbolic_len = dim_len;
          } else if (!ana.CanProveEqual(dim_len, symbolic_len)) {
            runtime_dep_dim = true;
          }
        }
      }
      if (static_len != -1) {
        output_shape.push_back(tir::make_const(DataType::Int(32), static_len));
      } else if (!runtime_dep_dim) {
        ICHECK(symbolic_len.defined());
        output_shape.push_back(symbolic_len);
      } else {
        return RuntimeDepShape();
      }
    }
  }

  return ShapeExpr(output_shape);
}

Type InferTypeConcatenate(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Concatenate op should have 1 argument");
  }

  const auto* tuple_type = call->args[0]->checked_type().as<TupleTypeNode>();
  if (tuple_type == nullptr) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "Concatenate operator expects the input to be a tuple or a list of tensors, indicating "
           "that the input type should be TupleType ShapeExpr. However, the given input has type "
        << call->args[0]->checked_type()->GetTypeKey());
  }
  const auto* attrs = call->attrs.as<ConcatenateAttrs>();

  int n_tensor = tuple_type->fields.size();
  if (n_tensor == 0) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Concatenate operator expects the input to have at least one tensor. "
                          "However, the given tensor tuple is empty");
  }

  int output_ndim = -1;
  DataType dtype = DataType::Void();
  for (int i = 0; i < n_tensor; ++i) {
    const auto* tensor_type = tuple_type->fields[i].as<DynTensorTypeNode>();
    if (tensor_type == nullptr) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Concatenate operator expects the input tuple has elements of type "
                            "DynTensorType. However, the element "
                         << i << " has type " << tuple_type->fields[i]->GetTypeKey());
    }
    if (!tensor_type->IsUnknownNdim()) {
      if (!attrs->axis.defined() && tensor_type->ndim != 1) {
        diag_ctx.EmitFatal(
            Diagnostic::Error(call->span)
            << "Concatenate operator expects all input tensors to be 1-dim tensors when not given "
               "a specific concatenation axis. However, the input tensor "
            << i << " has " << tensor_type->ndim << " dimensions");
      }
      if (output_ndim == -1) {
        output_ndim = tensor_type->ndim;
      } else if (tensor_type->ndim != output_ndim) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "Concatenate operator expects all input tensors to have the same "
                              "rank. However, one input tensor has rank "
                           << output_ndim << " while another has rank " << tensor_type->ndim);
      }
    }
    if (!tensor_type->IsUnknownDtype()) {
      if (dtype.is_void()) {
        dtype = tensor_type->dtype;
      } else if (tensor_type->dtype != dtype) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "Concatenate operator expects all input tensors to have the same "
                              "dtype. However, one input tensor has dtype "
                           << dtype << " while another has dtype " << tensor_type->dtype);
      }
    }
  }

  return DynTensorType(output_ndim, dtype);
}

/* relax.cumsum */
TVM_REGISTER_NODE_TYPE(CumsumAttrs);

RELAX_REGISTER_OP("relax.cumsum")
    .set_attrs_type<CumsumAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeCumsum)
    .set_attr<FInferType>("FInferType", InferTypeCumsum);

Expr MakeCumsum(Expr data, Optional<Integer> axis) {
  ObjectPtr<CumsumAttrs> attrs = make_object<CumsumAttrs>();
  attrs->axis = axis;

  static const Op& op = Op::Get("relax.cumsum");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.cumsum").set_body_typed(MakeCumsum);

Optional<Expr> InferShapeCumsum(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Cumsum op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<CumsumAttrs>();
  if (shape == nullptr) {
    return RuntimeDepShape();
  }

  if (attrs->axis.defined()) {
    return GetRef<ShapeExpr>(shape);
  }

  PrimExpr prod = tir::make_const(DataType::Int(32), 1);
  for (const PrimExpr& shape_dim : shape->values) {
    prod = prod * shape_dim;
  }
  return ShapeExpr({prod});
}

Type InferTypeCumsum(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Cumsum op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }

  const auto* attrs = call->attrs.as<CumsumAttrs>();
  if (attrs->axis.defined()) {
    return GetRef<DynTensorType>(input_type);
  } else {
    return DynTensorType(/*ndim=*/1, input_type->dtype);
  }
}

/* relax.trilu */
TVM_REGISTER_NODE_TYPE(TriluAttrs);

RELAX_REGISTER_OP("relax.trilu")
    .set_attrs_type<TriluAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeTrilu)
    .set_attr<FInferType>("FInferType", InferTypeTrilu);

Expr MakeTrilu(Expr data, int k, bool is_upper) {
  auto attrs = make_object<TriluAttrs>();
  attrs->k = k;
  attrs->is_upper = is_upper;

  static const Op& op = Op::Get("relax.trilu");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.trilu").set_body_typed(MakeTrilu);

Optional<Expr> InferShapeTrilu(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Trilu op should have 1 argument");
  }

  return call->args[0]->shape();
}

Type InferTypeTrilu(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Trilu op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Trilu operator requires the input data to have type DynTensorType. "
                          "However, the type of the given input is "
                       << call->args[0]->checked_type()->GetTypeKey());
  }

  return GetRef<DynTensorType>(input_type);
}

/* relax.cast */
TVM_REGISTER_NODE_TYPE(CastAttrs);

RELAX_REGISTER_OP("relax.cast")
    .set_attrs_type<CastAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferShape>("FInferShape", InferShapeCast)
    .set_attr<FInferType>("FInferType", InferTypeCast);

Expr MakeCast(Expr data, DataType dtype) {
  ObjectPtr<CastAttrs> attrs = make_object<CastAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.cast");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.cast").set_body_typed(MakeCast);

Optional<Expr> InferShapeCast(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Cast op should have 1 argument");
  }
  return call->args[0]->shape();
}

Type InferTypeCast(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Cast op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }
  const auto* attrs = call->attrs.as<CastAttrs>();
  return DynTensorType(input_type->ndim, attrs->dtype);
}

/* relax.take */
TVM_REGISTER_NODE_TYPE(TakeAttrs);

RELAX_REGISTER_OP("relax.take")
    .set_attrs_type<TakeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeTake)
    .set_attr<FInferType>("FInferType", InferTypeTake);

Expr MakeTake(Expr data, Expr indices, Optional<Integer> axis, int batch_dims, String mode) {
  ObjectPtr<TakeAttrs> attrs = make_object<TakeAttrs>();
  attrs->axis = std::move(axis);
  attrs->batch_dims = batch_dims;
  attrs->mode = std::move(mode);

  static const Op& op = Op::Get("relax.take");
  return Call(op, {std::move(data), std::move(indices)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.take").set_body_typed(MakeTake);

Optional<Expr> InferShapeTake(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Take op should have 2 arguments");
  }

  const auto* data_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* indices_shape = call->args[1]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<TakeAttrs>();

  if (indices_shape == nullptr) {
    return RuntimeDepShape();
  } else if (!attrs->axis.defined()) {
    return GetRef<ShapeExpr>(indices_shape);
  } else if (data_shape == nullptr) {
    return RuntimeDepShape();
  }

  int axis = attrs->axis.value()->value;
  int ndim_data = data_shape->values.size();
  if (axis < 0) {
    axis = ndim_data + axis;
  }
  if (axis < 0 || axis >= ndim_data) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Take operator expects the input axis to be in range [" << -ndim_data
                       << ", " << ndim_data << "). However, the given axis is "
                       << attrs->axis.value()->value << ", which is out of range");
  }

  Array<PrimExpr> output_shape = data_shape->values;
  output_shape.erase(output_shape.begin() + axis);
  output_shape.insert(output_shape.begin() + axis, indices_shape->values.begin(),
                      indices_shape->values.end());
  return ShapeExpr(output_shape);
}

Type InferTypeTake(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Take op should have 2 arguments");
  }

  const auto* data_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* indices_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  const auto* attrs = call->attrs.as<TakeAttrs>();
  if (data_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }
  if (indices_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input indices should has type DynTensorType, but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input indices has type DynTensorType.");
  }
  if (!indices_type->IsUnknownDtype() && !indices_type->dtype.is_int()) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Take operator expects the input indices to have integer dtype. However, "
                          "the given indices has dtype "
                       << indices_type->dtype);
  }

  if (indices_type->IsUnknownNdim()) {
    return DynTensorType(-1, data_type->dtype);
  } else if (!attrs->axis.defined()) {
    return DynTensorType(indices_type->ndim, data_type->dtype);
  } else if (data_type->IsUnknownNdim()) {
    return DynTensorType(-1, data_type->dtype);
  } else {
    return DynTensorType(data_type->ndim - 1 + indices_type->ndim, data_type->dtype);
  }
}

/* Initialization operators */
TVM_REGISTER_NODE_TYPE(FullAttrs);

/* relax.full */
RELAX_REGISTER_OP("relax.full")
    .set_attrs_type<FullAttrs>()
    .set_num_inputs(2)
    .add_argument("fill_value", "Tensor", "The scalar tensor, denoting the value to fill.")
    .add_argument("shape", "ShapeExpr", "The shape of the created tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeFull)
    .set_attr<FInferType>("FInferType", InferTypeFull);

Expr MakeFull(Expr fill_value, Expr shape, DataType dtype) {
  ObjectPtr<FullAttrs> attrs = make_object<FullAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.full");
  return Call(op, {std::move(fill_value), std::move(shape)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.full").set_body_typed(MakeFull);

Optional<Expr> InferShapeFull(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Full op should have 2 arguments");
  }

  const auto* fill_value_shape = call->args[0]->shape().as<ShapeExprNode>();
  if (fill_value_shape != nullptr && fill_value_shape->values.size() != 0) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Full operator expects the input fill value to be a scalar tensor "
                          "(0-rank tensor). However, the input fill value has rank "
                       << fill_value_shape->values.size());
  }

  return call->args[1];
}

Type InferTypeFull(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Full op should have 2 arguments");
  }

  const auto* fill_value_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* shape_type = call->args[1]->checked_type().as<ShapeTypeNode>();
  if (fill_value_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The input fill value should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }
  if (!fill_value_type->IsUnknownNdim() && fill_value_type->ndim != 0) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Full operator expects the input fill value to be a scalar tensor "
                          "(0-rank tensor). However, the input fill value has rank "
                       << fill_value_type->ndim);
  }
  if (shape_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The input shape should has type ShapeType, but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type ShapeType.");
  }

  // Todo(ruihang): add ndim to ShapeType
  int ndim = -1;
  const auto* shape = call->args[1].as<ShapeExprNode>();
  if (shape != nullptr) {
    ndim = shape->values.size();
  }

  const auto* attrs = call->attrs.as<FullAttrs>();
  return DynTensorType(ndim, attrs->dtype.is_void() ? fill_value_type->dtype : attrs->dtype);
}

/* relax.split */
TVM_REGISTER_NODE_TYPE(SplitAttrs);

RELAX_REGISTER_OP("relax.split")
    .set_attrs_type<SplitAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeSplit)
    .set_attr<FInferType>("FInferType", InferTypeSplit);

Expr MakeSplit(Expr data, ObjectRef indices_or_sections, int axis) {
  ObjectPtr<SplitAttrs> attrs = make_object<SplitAttrs>();
  attrs->indices_or_sections = indices_or_sections;
  if (const auto* n_section = indices_or_sections.as<IntImmNode>()) {
    CHECK(n_section->value > 0) << "Split operator expects the input number of sections to be a "
                                   "positive integer. However, the given number of sections is "
                                << n_section->value;
  }
  attrs->axis = axis;

  static const Op& op = Op::Get("relax.split");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.split").set_body_typed(MakeSplit);

Optional<Expr> InferShapeSplit(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Split op should have 1 argument");
  }

  const auto* input_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<SplitAttrs>();
  if (input_shape == nullptr) {
    return RuntimeDepShape();
  }

  int ndim = input_shape->values.size();
  int axis = attrs->axis;
  if (axis < 0) {
    axis = ndim + axis;
  }
  if (axis < 0 || axis >= ndim) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Split operator expects the input axis to be in range [" << -ndim << ", "
                       << ndim << "). However, the given axis is " << attrs->axis
                       << ", which is out of range");
  }

  Array<Expr> output_shape;
  PrimExpr len_axis = input_shape->values[axis];
  if (const auto* p_indices = attrs->indices_or_sections.as<ArrayNode>()) {
    Array<PrimExpr> indices = GetRef<Array<PrimExpr>>(p_indices);
    PrimExpr zero = tir::make_const(DataType::Int(32), 0);

    output_shape.reserve(indices.size() + 1);
    indices.insert(indices.begin(), zero);
    indices.insert(indices.end(), len_axis);

    for (int i = 0; i + 1 < static_cast<int>(indices.size()); ++i) {
      PrimExpr l = tvm::max(zero, indices[i]);
      PrimExpr r = tvm::min(len_axis, indices[i + 1]);
      PrimExpr len = tvm::max(zero, r - l);
      Array<PrimExpr> shape = input_shape->values;
      shape.erase(shape.begin() + axis);
      shape.insert(shape.begin() + axis, len);
      output_shape.push_back(ShapeExpr(shape));
    }
  } else {
    const auto* p_n_section = attrs->indices_or_sections.as<IntImmNode>();
    ICHECK_NOTNULL(p_n_section);
    int n_section = p_n_section->value;
    if (n_section <= 0) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Split operator expects the input number of sections to be a positive "
                            "integer. However, the given number of sections is "
                         << n_section);
    }
    if (const int64_t* len_axis_value = tir::as_const_int(len_axis)) {
      if (*len_axis_value % n_section != 0) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "Split operator expects the length of the input axis is divisible by "
                              "the input number of section. However, the axis has length "
                           << *len_axis_value << " while the given number of section is "
                           << n_section << ", which does not result in an equal division.");
      }
    }
    // Todo(relax-team): need runtime divisibility check for the cases where `len_axis` is symbolic

    PrimExpr n_section_expr = tir::make_const(DataType::Int(32), n_section);
    Array<PrimExpr> shape = input_shape->values;
    shape.erase(shape.begin() + axis);
    shape.insert(shape.begin() + axis,
                 arith::Analyzer().Simplify(tvm::floordiv(len_axis, n_section_expr)));
    // shape.insert(shape.begin() + axis, tvm::floordiv(len_axis, n_section_expr));
    for (int i = 0; i < n_section; ++i) {
      output_shape.push_back(ShapeExpr(shape));
    }
  }
  return Tuple(output_shape);
}

Type InferTypeSplit(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Split op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* attrs = call->attrs.as<SplitAttrs>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }

  int n_tensor = -1;
  if (const auto* p_indices = attrs->indices_or_sections.as<ArrayNode>()) {
    n_tensor = p_indices->size() + 1;
  } else {
    const auto* p_n_section = attrs->indices_or_sections.as<IntImmNode>();
    ICHECK_NOTNULL(p_n_section);
    n_tensor = p_n_section->value;
  }

  Array<Type> output_type;
  output_type.reserve(n_tensor);
  for (int i = 0; i < n_tensor; ++i) {
    output_type.push_back(GetRef<DynTensorType>(input_type));
  }
  return TupleType(output_type);
}

/* relax.broadcast_to */
RELAX_REGISTER_OP("relax.broadcast_to")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "ShapeExpr", "The shape of the created tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeBroadcastTo)
    .set_attr<FInferType>("FInferType", InferTypeBroadcastTo);

Expr MakeBroadcastTo(Expr data, Expr shape) {
  const static Op& op = Op::Get("relax.broadcast_to");
  return Call(op, {std::move(data), std::move(shape)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.broadcast_to").set_body_typed(MakeBroadcastTo);

Optional<Expr> InferShapeBroadcastTo(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "BroadcastTo op should have 2 arguments");
  }

  const auto* data_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* new_shape = call->args[1].as<ShapeExprNode>();
  if (data_shape == nullptr || new_shape == nullptr) {
    // Todo: need runtime shape broadcast compatibility check
    return call->args[1];
  }

  int data_ndim = data_shape->values.size();
  int new_ndim = new_shape->values.size();
  if (new_ndim < data_ndim) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The broadcast_to operator expects the input new shape to have at least "
                          "as many dimensions as the input data. However, the given data has ndim "
                       << data_ndim << " while the given shape has ndim " << new_ndim);
  }

  arith::Analyzer ana;
  for (int i = 1; i <= data_ndim; ++i) {
    PrimExpr prev_len = data_shape->values[data_ndim - i];
    PrimExpr new_len = new_shape->values[new_ndim - i];
    if (tir::is_const_int(prev_len, 1)) {
      continue;
    } else if (ana.CanProve(prev_len != new_len)) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The broadcast_to operator expects the input new shape is broadcast "
                            "compatible with the shape of the input data. However, on the last but "
                         << i << " dimension, the input data shape has length " << prev_len
                         << " while the nwe shape has length " << new_len
                         << ", which are not compatible");
    }
  }
  return GetRef<ShapeExpr>(new_shape);
}

Type InferTypeBroadcastTo(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "BroadcastTo op should have 2 arguments");
  }

  const auto* data_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* shape_type = call->args[1]->checked_type().as<ShapeTypeNode>();
  if (data_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }
  if (shape_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input new shape should has type ShapeType, but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type ShapeType.");
  }

  // Todo(ruihang): add ndim to ShapeType
  int ndim = -1;
  if (const auto* shape = call->args[1].as<ShapeExprNode>()) {
    ndim = shape->values.size();
  }
  return DynTensorType(ndim, data_type->dtype);
}

/* relax.strided_slice */
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);

RELAX_REGISTER_OP("relax.strided_slice")
    .set_attrs_type<StridedSliceAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeStridedSlice)
    .set_attr<FInferType>("FInferType", InferTypeStridedSlice);

Expr MakeStridedSlice(Expr data,                          //
                      Array<PrimExpr> begin,              //
                      Array<PrimExpr> end,                //
                      Optional<Array<PrimExpr>> strides,  //
                      Optional<Array<Integer>> axes,      //
                      String slice_mode) {
  CHECK(slice_mode == "end" || slice_mode == "size")
      << "Operator strided_slice expects the input `slice_mode` to be either \"end\" or \"size\". "
         "However, the given `slice_mode` is "
      << slice_mode;

  ObjectPtr<StridedSliceAttrs> attrs = make_object<StridedSliceAttrs>();
  attrs->begin = std::move(begin);
  attrs->end = std::move(end);
  attrs->strides = std::move(strides);
  attrs->axes = std::move(axes);
  attrs->slice_mode = std::move(slice_mode);

  const static Op& op = Op::Get("relax.strided_slice");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.strided_slice").set_body_typed(MakeStridedSlice);

Optional<Expr> InferShapeStridedSlice(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "StridedSlice op should have 1 argument");
  }

  const auto* input_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  if (input_shape == nullptr) {
    return RuntimeDepShape();
  }

  int ndim = input_shape->values.size();
  Array<Integer> axes;
  if (attrs->axes.defined()) {
    axes = attrs->axes.value();
  } else {
    axes.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      axes.push_back(Integer(i));
    }
  }

  int n_axis = axes.size();
  Array<PrimExpr> begins = attrs->begin;
  Array<PrimExpr> ends = attrs->end;
  Array<PrimExpr> strides;
  if (attrs->strides.defined()) {
    strides = attrs->strides.value();
  } else {
    strides.reserve(n_axis);
    for (int i = 0; i < n_axis; ++i) {
      strides.push_back(tir::make_const(DataType::Int(32), 1));
    }
  }

  if (static_cast<int>(begins.size()) != n_axis) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The strided_slice operator expects the input begin values to have the same length as "
           "the number of input axes. However, the input axes length is  "
        << n_axis << " while the length of begin values is " << begins.size());
  }
  if (static_cast<int>(ends.size()) != n_axis) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The strided_slice operator expects the input end values to have the same length as "
           "the number of input axes. However, the input axes length is  "
        << n_axis << " while the length of end values is " << ends.size());
  }
  if (static_cast<int>(strides.size()) != n_axis) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The strided_slice operator expects the input stride values to have the same length as "
           "the number of input axes. However, the input axes length is  "
        << n_axis << " while the length of stride values is " << strides.size());
  }

  arith::Analyzer ana;
  Array<PrimExpr> output_shape = input_shape->values;
  std::unordered_set<int> specified_axes;
  specified_axes.reserve(axes.size());
  for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
    int axis = axes[i]->value;
    if (axis < 0) {
      axis = ndim + axis;
    }
    if (axis < 0 || axis >= ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Operator strided_slice expects the input axis to be in range ["
                         << -ndim << ", " << ndim << "). However, the given axis " << i << " is "
                         << axes[i]->value << ", which is out of range");
    }
    if (specified_axes.count(axis)) {
      diag_ctx.EmitFatal(
          Diagnostic::Error(call->span)
          << "Operator strided_slice expects the input axes not to duplicate. However, axis "
          << axis << " occurs twice");
    }
    specified_axes.insert(axis);

    PrimExpr begin = begins[i];
    PrimExpr end{nullptr};
    PrimExpr stride = strides[i];

    if (attrs->slice_mode == "size") {
      stride = tir::make_const(DataType::Int(32), 1);
      end = begin + ends[i];
    } else {
      if (attrs->slice_mode != "end") {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "The strided_slice operator expects the input `slice_mode` to be "
                              "either \"end\" or \"size\". However, the given `slice_mode` is "
                           << attrs->slice_mode);
      }
      end = tvm::min(input_shape->values[axis], ends[i]);
    }
    if (ana.CanProveLess(stride, 0)) {
      output_shape.Set(axis, tvm::ceildiv(begin - end, -stride));
    } else {
      output_shape.Set(axis, tvm::ceildiv(end - begin, stride));
    }
  }

  return ShapeExpr(output_shape);
}

Type InferTypeStridedSlice(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "StridedSlice op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }

  return GetRef<DynTensorType>(input_type);
}

}  // namespace relax
}  // namespace tvm
