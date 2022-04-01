#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import _obj_to_value
from .pycde_types import dim as _dim
from .value import Value as _Value
from circt.support import get_value as _get_value
from circt.dialects import msft as _msft, hw as _hw
import mlir.ir as _ir


def SystolicArray(pe_output_type, row_inputs, col_inputs, pe_builder):
  """Build a systolic array."""

  row_inputs_type = _hw.ArrayType(row_inputs.type)
  col_inputs_type = _hw.ArrayType(col_inputs.type)
  sa_result_type = _dim(pe_output_type, col_inputs_type.size,
                        row_inputs_type.size)
  array = _msft.SystolicArrayOp(sa_result_type, _get_value(row_inputs),
                                _get_value(col_inputs))
  pe = array.pe.blocks.append(row_inputs_type.element_type,
                              col_inputs_type.element_type)
  with _ir.InsertionPoint(pe):
    result = pe_builder(_Value.get(pe.arguments[0]),
                        _Value.get(pe.arguments[1]))
    value = _obj_to_value(result, pe_output_type)
    _msft.PEOutputOp(value.value)
  return array.peOutputs
