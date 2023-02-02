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
# pylint: disable=invalid-name, unused-import
"""The type nodes of the Relax language."""
from typing import Any, List, Optional, Union
import tvm._ffi
from tvm.ir import Type, TensorType, TupleType, FuncType, Span

from . import _ffi_api
from .. import relay
from ..tir import PrimExpr

Expr = relay.Expr

@tvm._ffi.register_object("relax.ShapeType")
class ShapeType(Type):
    """The type of shape in Relax."""

    def __init__(self, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeType, span)


@tvm._ffi.register_object("relax.ObjectType")
class ObjectType(Type):
    """A type that corresponds to tvm::runtime::Object, is base of all possible object
    values in TVM."""

    def __init__(self, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ObjectType, span)


@tvm._ffi.register_object("relax.DynTensorType")
class DynTensorType(Type):
    """A dynamic tensor type in Relax.

    This is the type assigned to tensors with a known dtype and unknown shape.

    Parameters
    ----------
    ndim : Optional[int]
        The ndim of the Tensor

    dtype : Optional[str]
        The content data type.
    """

    def __init__(self, ndim=-1, dtype="float32", span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DynTensorType, ndim, dtype, span)


@tvm._ffi.register_object("relax.DimType")
class DimType(Type):
    """The type of indices/shape dimensions in Relax."""

    def __init__(self, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DimType, span)


@tvm._ffi.register_object("relax.RaggedDynTensorType")
class RaggedDynTensorType(Type):
    def __init__(self, ndim=-1, dtype="float32", span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.RaggedDynTensorType, ndim, dtype, span)


# @tvm._ffi.register_object("relax.RaggedDimType")
# class RaggedDimType(Type):
#     def __init__(self, is_ragged: bool, fixed_bound: Optional[Var], 
#             parent: Optional[Var], ind_ptr: Optional[Var], span: Span = None) -> None:
#         self.__init_handle_by_constructor__(_ffi_api.RaggedDimType, 
#             fixed_bound, parent, ind_ptr, is_ragged, span)


def is_base_of(base: Type, derived: Type) -> bool:
    """Check the subtype relationship between base and derived.

    Parameters
    ----------
    base : Type
        The base type.

    derived : Type
        The derived type.


    Returns
    -------
    ret : bool
        If derived is a subtype of base or if both are the same type, returns true.
        Otherwise returns false.
    """
    return _ffi_api.IsBaseOf(base, derived)
