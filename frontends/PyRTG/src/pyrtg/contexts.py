#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .rtgtest import rtgtest
from .core import Value, Type
from .base import ir
from .support import _FromCirctValue, _collect_values_recursively
from .sequences import Sequence

import ctypes
import sys
import inspect
from contextvars import ContextVar
from typing import Union, Dict, Set

_context_stack = ContextVar("context_stack", default=[])
_context_seq_num = ContextVar("context_seq_num", default=0)


def _get_next_seq_num():
  num = _context_seq_num.get()
  _context_seq_num.set(num + 1)
  return str(num)


class CPUCore(Value):
  """
  Represents a CPU core in the test environment. A CPU core can be specified
  either by its hardware thread ID (hartid) or can represent all available
  cores. This class allows operations to target specific cores or all cores
  when generating randomized tests.
  """

  def __init__(self, hartid: Union[int, ir.Attribute, ir.Value]) -> CPUCore:
    """
    Creates a CPUCore instance for a specific hardware thread ID.
    """

    if isinstance(hartid, ir.Value):
      self._attr = None
      self._value = hartid
      return

    self._attr = hartid if isinstance(
        hartid, ir.Attribute) else rtgtest.CPUAttr.get(hartid)
    self._value = self._attr

  @staticmethod
  def any() -> CPUCore:
    """
    Creates a CPUCore instance referring to any core. This is useful when
    specifying a sequence to use for switching from one core to another.
    """

    return CPUCore(rtg.AnyContextAttr.get(rtgtest.CPUType.get()))

  @staticmethod
  def register_switch(from_core: CPUCore, to_core: CPUCore,
                      seq: Sequence) -> None:
    """
    Register a sequence with arguments of types [CPUCore, CPUCore, Sequence] to
    be usable for switching from context 'from_core' to context 'to_core' and back.
    """

    assert from_core._attr is not None and to_core._attr is not None, "must have attribute available"
    rtg.ContextSwitchOp(from_core._attr, to_core._attr, seq)

  def __enter__(self):
    # TODO: just adding all variables in the context is not particularly nice.
    # Maybe we can analyze the Python AST and only add the variables actually
    # used in the nested block in the future.
    args = []
    arg_names = []
    visited = set()
    for varname, obj in sys._getframe(1).f_locals.items():
      _collect_values_recursively(obj, varname, args, arg_names, visited)

    curr = ir.InsertionPoint.current.block.owner.operation
    while curr.name != 'builtin.module':
      curr = curr.parent

    arg_types = [arg.get_type()._codegen() for arg in args]
    # TODO: we currently just assume "_context_seq" is a reserved prefix, would
    # be good to do proper name uniquing in the future
    seq_name = "_context_seq_" + _get_next_seq_num()
    seq_type = rtg.SequenceType.get(arg_types)
    with ir.InsertionPoint(curr.regions[0].blocks[0]):
      seq_decl = rtg.SequenceOp(seq_name, ir.TypeAttr.get(seq_type))
      block = ir.Block.create_at_start(seq_decl.regions[0], arg_types)

    seq = rtg.GetSequenceOp(seq_type, seq_name)
    seq = rtg.SubstituteSequenceOp(seq, args)
    rtg.OnContextOp(self, seq)

    s = inspect.stack()[1][0]
    _context_stack.get().append((block, dict(s.f_locals), arg_names, args))

    ir.InsertionPoint(block).__enter__()

    for path, arg in zip(arg_names, block.arguments):
      if '.' in path:
        parts = path.split('.')
        obj = s.f_locals[parts[0]]
        for part in parts[1:-1]:
          obj = getattr(obj, part)
        setattr(obj, parts[-1], _FromCirctValue(arg))
      else:
        s.f_locals[path] = _FromCirctValue(arg)

    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))

    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    if exc_value is not None:
      return

    s = inspect.stack()[1][0]
    block, scope, arg_names, args = _context_stack.get().pop()

    for path, arg in zip(arg_names, args):
      if '.' in path:
        parts = path.split('.')
        base_name = parts[0]
        obj = s.f_locals[base_name]
        for part in parts[1:-1]:
          obj = getattr(obj, part)
        setattr(obj, parts[-1], arg)
      else:
        s.f_locals[path] = arg

    ir.InsertionPoint(block).__exit__(exc_type, exc_value, exc_traceback)

    for (varname, _) in s.f_locals.items():
      if varname not in scope or scope[varname] is None:
        s.f_locals[varname] = None

    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, ir.Attribute):
      self = rtg.ConstantOp(self._value)
    return self._value

  def get_type(self) -> Type:
    return CPUCoreType()


class CPUCoreType(Type):
  """
  Represents the type of CPU cores in the test environment.

  This type is used for values that represent CPU cores, which can be
  specified by hardware thread ID or represent all available cores.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, CPUCoreType)

  def _codegen(self) -> ir.Type:
    return rtgtest.CPUType.get()
