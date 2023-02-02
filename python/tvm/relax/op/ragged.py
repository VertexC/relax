from typing import Union, List, Optional, Tuple, Int

import tvm
from tvm.runtime.object import Object

from . import _ffi_api
from ..expr import Expr, ShapeExpr, Tuple, Call, ExternFunc
from ..ty import DynTensorType, TupleType
from ...ir import Array

py_print = print  # pylint: disable=invalid-name


