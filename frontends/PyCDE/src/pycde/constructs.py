#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .pycde_types import dim
from .value import BitVectorValue, ListValue, Value
from circt.support import get_value
from pycde.dialects import comb
from circt.dialects import msft, hw
import mlir.ir as ir

import ctypes
from contextvars import ContextVar
import inspect
from typing import Dict, List, Tuple


def Mux(sel: BitVectorValue, *data_inputs: List[Value]):
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


_current_if_stmt = ContextVar("current_pycde_if_stmt")


class If:
  """Syntactic sugar for creation of muxes with if-then-else-ish behavioral
  syntax.

  ```
  @module
  class IfDemo:
    cond = Input(types.i1)
    out = Output(types.i8)

    @generator
    def build(ports):
      with If(ports.cond):
        with Then:
          v = types.i8(1)
        with Else:
          v = types.i8(0)
      ports.out = v
  ```"""

  def __init__(self, cond: BitVectorValue):
    if (cond.type.width != 1):
      raise TypeError("'Cond' bit width must be 1")
    self._cond = cond
    self._muxes: Dict[str, Tuple[Value, Value]] = {}

  @staticmethod
  def current():
    return _current_if_stmt.get(None)

  def __enter__(self):
    self._old_system_token = _current_if_stmt.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_if_stmt.reset(self._old_system_token)

    s = inspect.stack()[1][0]
    new_locals: Dict[str, Value] = {}
    for (varname, (else_value, then_value)) in self._muxes.items():
      if then_value is None or else_value is None:
        continue
      if then_value.type != else_value.type:
        raise TypeError(
            f"'Then' and 'Else' values must have same type for {varname}" +
            f" ({then_value.type} vs {else_value.type})")
      then_value.name = f"{varname}_thenvalue"
      else_value.name = f"{varname}_elsevalue"
      mux = comb.MuxOp(self._cond, then_value, else_value)
      mux.name = varname
      new_locals[varname] = mux
    s.f_locals.update(new_locals)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))


class _IfBlock:

  def __init__(self, is_then: bool):
    self._is_then = is_then

  def __enter__(self):
    s = inspect.stack()[1][0]
    self._scope = dict(s.f_locals)

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_val is not None:
      return

    if_stmt = If.current()
    s = inspect.stack()[1][0]
    lcls_to_del = set()
    new_lcls: Dict[str, Value] = {}
    for (varname, value) in s.f_locals.items():
      if not isinstance(value, Value):
        continue
      if varname in self._scope and self._scope[varname] is value:
        continue
      if varname not in if_stmt._muxes:
        if_stmt._muxes[varname] = (None, None)
      m = if_stmt._muxes[varname]

      if self._is_then:
        if m[1] is not None:
          raise Exception(
              f"Multiple assignments to '{varname}' in 'then' block")
        if_stmt._muxes[varname] = (m[0], value)

      if not self._is_then:
        if m[0] is not None:
          raise Exception(
              f"Multiple assignments to '{varname}' in 'else' block")
        if_stmt._muxes[varname] = (value, m[1])

      if varname in self._scope:
        new_lcls[varname] = self._scope[varname]
      else:
        lcls_to_del.add(varname)

    for varname in lcls_to_del:
      del s.f_locals[varname]
      ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s),
                                            ctypes.c_int(1))
    s.f_locals.update(new_lcls)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))


Then = _IfBlock(True)
Else = _IfBlock(False)
