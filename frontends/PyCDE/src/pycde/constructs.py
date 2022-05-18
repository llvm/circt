#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import _obj_to_value
from .pycde_types import dim
from .value import Value
from circt.support import get_value
from circt.dialects import msft, hw
import mlir.ir as ir


def SystolicArray(row_inputs, col_inputs, pe_builder):
  """Build a systolic array."""

  row_inputs_type = hw.ArrayType(row_inputs.type)
  col_inputs_type = hw.ArrayType(col_inputs.type)

  dummy_op = ir.Operation.create("dummy", regions=1)
  pe_block = dummy_op.regions[0].blocks.append(row_inputs_type.element_type,
                                               col_inputs_type.element_type)
  with ir.InsertionPoint(pe_block):
    result = pe_builder(Value(pe_block.arguments[0]),
                        Value(pe_block.arguments[1]))
    value = _obj_to_value(result, result.type)
    pe_output_type = value.type
    msft.PEOutputOp(value.value)

  sa_result_type = dim(pe_output_type, col_inputs_type.size,
                       row_inputs_type.size)
  array = msft.SystolicArrayOp(sa_result_type, get_value(row_inputs),
                               get_value(col_inputs))
  dummy_op.regions[0].blocks[0].append_to(array.regions[0])
  dummy_op.operation.erase()

  return array.peOutputs
