#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .pycde_types import PyCDEType, dim
from .value import BitVectorValue, ListValue, Value
from circt.support import get_value, BackedgeBuilder
from circt.dialects import msft, hw
import mlir.ir as ir

import typing


def Wire(type: PyCDEType):
  """Declare a wire. Used to create backedges. Must assign exactly once."""

  class WireValue(type._get_value_class()):

    def __init__(self):
      self._backedge = BackedgeBuilder.create(type, "wire", None)
      super().__init__(self._backedge.result, type)

    def assign(self, new_value: Value):
      if self._backedge is None:
        raise ValueError("Cannot assign value to Wire twice.")
      if new_value.type != self.type:
        raise TypeError(
            f"Cannot assign {new_value.value.type} to {self.value.type}")

      msft.replaceAllUsesWith(self._backedge.result, new_value.value)
      self._backedge.erase()
      self._backedge = None
      self.value = new_value.value

  return WireValue()


def Reg(type: PyCDEType,
        clk: Value = None,
        rst: Value = None,
        rst_value: Value = None):
  """Declare a register. Must assign exactly once."""

  class RegisterValue(type._get_value_class()):

    def assign(self, new_value: Value):
      if self._wire is None:
        raise ValueError("Cannot assign value to Reg twice.")
      self._wire.assign(new_value)
      self._wire = None

  # Create a wire and register it.
  wire = Wire(type)
  value = RegisterValue(wire.reg(clk, rst, rst_value), type)
  value._wire = wire
  return value


def Mux(sel: BitVectorValue, *data_inputs: typing.List[Value]):
  """Create a single mux from a list of values."""
  num_inputs = len(data_inputs)
  if num_inputs == 0:
    raise ValueError("'Mux' must have 1 or more data input")
  if num_inputs == 1:
    return data_inputs[0]
  if sel.type.width != (num_inputs - 1).bit_length():
    raise TypeError("'Sel' bit width must be clog2 of number of inputs")
  return ListValue(data_inputs)[sel]


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
    value = Value(result)
    pe_output_type = value.type
    msft.PEOutputOp(value.value)

  sa_result_type = dim(pe_output_type, col_inputs_type.size,
                       row_inputs_type.size)
  array = msft.SystolicArrayOp(sa_result_type, get_value(row_inputs),
                               get_value(col_inputs))
  dummy_op.regions[0].blocks[0].append_to(array.regions[0])
  dummy_op.operation.erase()

  return Value(array.peOutputs)
