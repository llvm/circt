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


def SystolicArray(row_inputs, col_inputs, pe_builder):
  """Build a systolic array."""

  row_inputs_type = _hw.ArrayType(row_inputs.type)
  col_inputs_type = _hw.ArrayType(col_inputs.type)

  dummy_op = _ir.Operation.create("dummy", regions=1)
  pe_block = dummy_op.regions[0].blocks.append(row_inputs_type.element_type,
                                               col_inputs_type.element_type)
  with _ir.InsertionPoint(pe_block):
    result = pe_builder(_Value.get(pe_block.arguments[0]),
                        _Value.get(pe_block.arguments[1]))
    value = _obj_to_value(result, result.type)
    pe_output_type = value.type
    _msft.PEOutputOp(value.value)

  sa_result_type = _dim(pe_output_type, col_inputs_type.size,
                        row_inputs_type.size)
  array = _msft.SystolicArrayOp(sa_result_type, _get_value(row_inputs),
                                _get_value(col_inputs))
  _msft.move_first_block(dummy_op, array)
  dummy_op.operation.erase()

  return array.peOutputs
