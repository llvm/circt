#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .pycde_types import PyCDEType, dim
from .value import BitVectorValue, ListValue, Value
from circt.support import get_value
from circt.dialects import msft, hw
import mlir.ir as ir

import typing


class Wire(Value):
  """Declare a wire. Used to create backedges and they disappear when assigned
  to."""

  def __init__(self, type: PyCDEType):
    self.value = ir.Operation.create("builtin.reinterpret_cast", [type])
    self.type = type
    self.assigned = False

  def __new__(cls, type, name=None):
    return super(Value, cls).__new__(cls)

  def assign(self, new_value: Value):
    if self.assigned:
      raise ValueError("Cannot assign to a wire that has already been assigned")
    if new_value.type != self.type:
      raise TypeError(
          f"Cannot assign {new_value.value.type} to {self.value.type}")

    self.assigned = True
    msft.replaceAllUsesWith(self.value.result, new_value.value)
    self.value.operation.erase()
    self.value = new_value.value


class Reg:
  """Declare a register. Must be assigned at some point."""

  def __init__(self,
               type: PyCDEType,
               clk: Value = None,
               rst: Value = None,
               rst_value: Value = None):
    self.wire = Wire(type)
    self.value = self.wire.reg(clk, rst, rst_value)
    self.type = type

  def __new__(cls, type, name=None):
    return super(Value, cls).__new__(cls)

  def assign(self, new_value: Value):
    if self.wire is None:
      raise ValueError("Cannot assign to a reg that has already been assigned")

    self.wire.assign(new_value)
    self.wire = None


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
