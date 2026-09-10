#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Dict, List, Tuple, Union

from . import handshake
from ._handshake_ops_gen import *
from ._handshake_ops_gen import _Dialect

from ..dialects._ods_common import _cext as _ods_cext
from ..ir import ArrayAttr, Attribute, FunctionType, StringAttr, Type, TypeAttr

from ._ods_common import (
    equally_sized_accessor as _ods_equally_sized_accessor,
    get_default_loc_context as _ods_get_default_loc_context,
    get_op_result_or_op_results as _get_op_result_or_op_results,
    get_op_result_or_value as _get_op_result_or_value,
    get_op_results_or_values as _get_op_results_or_values,
    segmented_accessor as _ods_segmented_accessor,
)

_ods_ir = _ods_cext.ir


@_ods_cext.register_operation(_Dialect, replace=True)
class FuncOp(FuncOp):

  @staticmethod
  def create(sym_name: Union[StringAttr, str],
             args: List[Tuple[str, Type]],
             results: List[Tuple[str, Type]],
             attributes: Dict[str, Attribute] = {},
             loc=None,
             ip=None) -> FuncOp:
    if isinstance(sym_name, str):
      sym_name = StringAttr.get(sym_name)
    input_types = [t for _, t in args]
    res_types = [t for _, t in results]
    func_type = FunctionType.get(input_types, res_types)
    func_type_attr = TypeAttr.get(func_type)
    funcop = FuncOp(func_type_attr, loc=loc, ip=ip)
    for k, v in attributes.items():
      funcop.attributes[k] = v
    funcop.attributes["sym_name"] = sym_name
    funcop.attributes["argNames"] = ArrayAttr.get(
        [StringAttr.get(name) for name, _ in args])
    funcop.attributes["resNames"] = ArrayAttr.get(
        [StringAttr.get(name) for name, _ in results])
    return funcop

  def add_entry_block(self):
    self.body.blocks.append(*self.function_type.value.inputs)
    return self.body.blocks[0]
