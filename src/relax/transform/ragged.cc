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
 * \file tvm/relax/transform/normalize.cc
 * \brief Pass for transforming Relax IR to normal form, i.e., the expressions are normalized(no
 * nesting and hence the AST is in ANF), and all checked_type_ and shape_ of expressions are
 * available.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/analysis.h>
#include "../op/make_op.h"

namespace tvm {
namespace relax {

class RaggedMatmulToDenseMutator : public ExprMutator {
 public:
  // RaggedMatmulToDenseMutator() { builder_ = BlockBuilder::Create(NullOpt); }
  static Function Mutate(Function f) {
    Function new_f = Downcast<Function>(RaggedMatmulToDenseMutator().VisitExpr(f));
    new_f = RemoveAllUnused(new_f);
    return new_f;
  }

  Expr VisitExpr_(const CallNode* call) {
    static const Op& ragged_tensor_pack_op = Op::Get("relax.nn.ragged_tensor_pack");
    static const Op& ragged_matmul_op = Op::Get("relax.nn.ragged_matmul");

    if (call->op == ragged_matmul_op) {
      const auto* b_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
      LOG(INFO) << a_dense_ << " " << call->args[1] << " " << b_type->dtype; 
      auto matmul_op = MakeMatmul(a_dense_, 
            call->args[1], b_type->dtype);
      Var tensor =
          builder_->Emit(matmul_op, "out");
      return tensor;
    } else if (call->op == ragged_tensor_pack_op) {
      a_dense_ = call->args[1];
    }

    return GetRef<Expr>(call);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : op->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      all_params_unchanged &= param.same_as(new_param);
    }

    Expr body = this->VisitWithNewScope(op->body);

    Type ret_type = body->checked_type();
    Expr ret_shape = body->shape();
    LOG(INFO) << ret_type << " " << ret_shape;
    if (all_params_unchanged && ret_type.same_as(op->ret_type) && body.same_as(op->body) &&
        ret_shape.same_as(op->ret_shape)) {
      return GetRef<Expr>(op);
    } else {
      return Function(params, body, ret_type, ret_shape, op->attrs);
    }
  }


 private:
  /*! \brief Internal block builder to emit bindings during rewriting. */
  Expr a_dense_;
};  // namespace relax

Expr RaggedMatmulToDense(const Function& e) { return RaggedMatmulToDenseMutator::Mutate(e); }


class AutobatchMutator : public ExprMutator {
 public:

  Expr VisitExpr_(const FunctionNode* op) final {


    Var a = op->params[0];
    ShapeExpr old_shape = Downcast<ShapeExpr>(a->shape());
    Array<PrimExpr> new_shape_values;
    tvm::tir::Var batch_dim = tvm::tir::Var("batch_dim", DataType::Int(32));
    new_shape_values.push_back(batch_dim);
    for(PrimExpr v: old_shape->values) {
      new_shape_values.push_back(v);
    }

    ShapeExpr new_shape(new_shape_values);
    DynTensorType old_type = Downcast<DynTensorType>(a->checked_type_);
    DynTensorType new_type(old_type->ndim, old_type->dtype, old_type->span);
    Var new_var = Var(a->vid, new_shape, new_type, a->span);

    this->var_remap_[a->vid] = new_var;

    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    params.push_back(new_var);
    for (size_t i=1; i<op->params.size(); i++) {
      Var param = op->params[i];
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      all_params_unchanged &= param.same_as(new_param);
    }


    Type ret_type = this->VisitType(op->ret_type);
    Expr ret_shape = this->VisitExpr(op->ret_shape);
    Expr body = this->VisitWithNewScope(op->body);

    if (all_params_unchanged && ret_type.same_as(op->ret_type) && body.same_as(op->body) &&
        ret_shape.same_as(op->ret_shape)) {
      return GetRef<Expr>(op);
    } else {
      return Function(params, body, ret_type, ret_shape, op->attrs);
    }
  }


 private:
  /*! \brief Internal block builder to emit bindings during rewriting. */
  Expr a_dense_;
};  // namespace relax

Expr Autobatch(const Function& e) { return AutobatchMutator().VisitExpr(e); }

namespace transform {

Pass AutobatchMatmul() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(Autobatch(f)); };
  return CreateFunctionPass(pass_func, 0, "AutobatchMatmul", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AutobatchMatmul").set_body_typed(AutobatchMatmul);


Pass RaggedMatmulToDenseMatmul() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(RaggedMatmulToDense(f)); };
  return CreateFunctionPass(pass_func, 0, "RaggedMatmulToDenseMatmul", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RaggedMatmulToDenseMatmul").set_body_typed(RaggedMatmulToDenseMatmul);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
