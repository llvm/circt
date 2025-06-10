#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .integers import Bool, Integer
from .arrays import Array
from .core import Value
from .base import ir
from .scf import scf
from .support import _FromCirctValue, _collect_values_recursively

import ctypes
from contextvars import ContextVar
import inspect
import sys
from typing import Dict, Optional, Set, Union

_current_if_stmt = ContextVar("current_pyrtg_if_stmt")


class If:
  """
  This class can be used to generate if-else statements with RTG 'Bool' values
  as condition. They will be captured in the generated IR unlike regular Python
  if-else statements.

  Example:
  ```
  with If(cond):
    v = Integer(1)
  with Else():
    v = Integer(0)
  EndIf()
  use(v)
  ```
  """

  def __init__(self, cond: Bool):
    self._cond = cond
    self._then_results: Dict[str, Value] = {}
    self._else_results: Dict[str, Value] = {}
    self._defaults: Dict[str, Value] = {}
    self._hasElse = False

  @staticmethod
  def current() -> Optional[If]:
    return _current_if_stmt.get(None)

  def __enter__(self):
    self._old_system_token = _current_if_stmt.set(self)
    # Keep all the important logic in the _IfBlock class so we can share it with
    # 'Else'.
    self._op = scf.IfOp(self._cond._get_ssa_value(), hasElse=True)
    self.then = _IfBlock(True)
    self.then.__enter__(stack_level=2)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    self.then.__exit__(exc_type, exc_value, traceback, stack_level=2)

  def _finalize(self):
    _current_if_stmt.reset(self._old_system_token)

    results = sorted(
        set(x for x, _ in list(self._then_results.items()) +
            list(self._else_results.items())
            if (x in self._then_results or x in self._defaults) and
            (x in self._else_results or x in self._defaults)))

    then_list = []
    else_list = []
    for varname in results:
      then_val = self._then_results.get(varname, self._defaults[varname])
      else_val = self._else_results.get(varname, self._defaults[varname])
      if then_val.get_type() != else_val.get_type():
        raise TypeError(
            f"'Then' and 'Else' values must have same type for {varname} ({then_val.get_type()} vs {else_val.get_type()})"
        )

      then_list.append(then_val)
      else_list.append(else_val)

    with ir.InsertionPoint(self._op.then_block):
      scf.YieldOp(then_list)

    with ir.InsertionPoint(self._op.else_block):
      scf.YieldOp(else_list)

    # FIXME: this is very ugly because the MLIR python bindings do now allow us
    # to delete blocks from regions and the IfOp class directly adds blocks to
    # the region in its constructor.
    hasElse = self._hasElse or len(else_list) > 0
    new_if = scf.IfOp(self._cond._get_ssa_value(),
                      [v.get_type()._codegen() for v in then_list],
                      hasElse=hasElse)
    for op in self._op.then_block.operations:
      new_if.operation.regions[0].blocks[0].append(op)
    if hasElse:
      for op in self._op.else_block.operations:
        new_if.operation.regions[1].blocks[0].append(op)
    self._op.erase()
    self._op = new_if

    # Update the stack frame with the new if-statement results as locals
    s = inspect.stack()[2][0]

    for i, varname in enumerate(results):
      s.f_locals[varname] = _FromCirctValue(self._op.results[i])

    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))


class _IfBlock:
  """
  Track additions and changes to the current stack frame.
  """

  def __init__(self, is_then: bool):
    self._is_then = is_then

  # Capture the current scope of the stack frame and store a copy.
  def __enter__(self, stack_level=1):
    if_stmt = If.current()
    if self._is_then:
      self._ip = ir.InsertionPoint(if_stmt._op.then_block)
    else:
      self._ip = ir.InsertionPoint(if_stmt._op.else_block)
      if_stmt._hasElse = True

    s = inspect.stack()[stack_level][0]
    self._scope = dict(s.f_locals)

    self._ip.__enter__()

  def __exit__(self, exc_type, exc_val, exc_tb, stack_level=1):
    # Don't do nothing if an exception was thrown.
    if exc_val is not None:
      return

    if_stmt = If.current()
    s = inspect.stack()[stack_level][0]
    new_lcls: Dict[str, Value] = {}
    lcls_to_del: Set[str] = set()
    for (varname, value) in s.f_locals.items():
      # Only operate on Values.
      if not isinstance(value, Value):
        continue

      # If the value was in the original scope and it hasn't changed, don't
      # touch it.
      if varname in self._scope and self._scope[varname] is value:
        continue

      # If that variable exists in the outer scope, use it as a default.
      if varname not in (if_stmt._then_results if self._is_then else
                         if_stmt._else_results) and varname in self._scope:
        if_stmt._defaults[varname] = self._scope[varname]

      # If a variable changes, we need to keep track of it to return the new
      # value.
      if self._is_then:
        if_stmt._then_results[varname] = value
      else:
        if_stmt._else_results[varname] = value

      if varname in self._scope:
        new_lcls[varname] = self._scope[varname]
      else:
        lcls_to_del.add(varname)

    # TODO: probably better to delete the variable
    for varname in lcls_to_del:
      s.f_locals[varname] = None

    # Restore existing locals to their original values.
    s.f_locals.update(new_lcls)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))

    self._ip.__exit__(exc_type, exc_val, exc_tb)


def Else():
  return _IfBlock(False)


def EndIf():
  c = If.current()
  assert c, "EndIf() called without matching If()"
  c._finalize()


class For:
  """
  This class represents a for loop construct that will be captured in the generated IR,
  unlike regular Python for loops. Similarly to ranges in Python, it has a
  start, stop, and step argument to indicate the start value of the induction
  variable (inclusive), the upper bound (exclusive), and the step to be added
  each iteration.
  Variables declared outside the body of this For-loop and modified inside will
  be added as loop carried values.

  Example:
  ```
  with For(lower, upper, step) as i:
    v += Integer(1)  # v is an iteration argument
  use(v)  # v contains the final value after all iterations
  ```
  """

  def __init__(self,
               start_or_stop: Union[int, Integer],
               stop: Optional[Union[int, Integer]] = None,
               step: Union[int, Integer] = 1):
    if stop is None:
      stop = start_or_stop
      start = Integer(0)
    else:
      start = start_or_stop if isinstance(start_or_stop,
                                          Integer) else Integer(start_or_stop)

    self._lower = start
    self._upper = stop if isinstance(stop, Integer) else Integer(stop)
    self._step = step if isinstance(step, Integer) else Integer(step)

  def _return_on_enter(self) -> Value:
    return self._index

  def __enter__(self, stack_level=1) -> Value:
    visited = set()
    self._init_args = []
    self._init_arg_names = []
    for varname, obj in sys._getframe(1).f_locals.items():
      if varname != "_":
        _collect_values_recursively(obj, varname, self._init_args,
                                    self._init_arg_names, visited)

    self._op = scf.ForOp(self._lower._get_ssa_value(),
                         self._upper._get_ssa_value(),
                         self._step._get_ssa_value(),
                         [x._get_ssa_value() for x in self._init_args])
    self._ip = ir.InsertionPoint(self._op.body)

    s = inspect.stack()[stack_level][0]
    self._scope = dict(s.f_locals)

    self._ip.__enter__()

    all = [_FromCirctValue(arg) for arg in self._op.body.arguments]
    self._index = all[0]
    self._iter_args = all[1:]

    for path, arg in zip(self._init_arg_names, self._iter_args):
      if '.' in path:
        parts = path.split('.')
        obj = s.f_locals[parts[0]]
        for part in parts[1:-1]:
          obj = getattr(obj, part)
        setattr(obj, parts[-1], arg)
      else:
        s.f_locals[path] = arg

    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))

    return self._return_on_enter()

  def __exit__(self, exc_type, exc_value, traceback, stack_level=1):
    if exc_value is not None:
      return

    s = inspect.stack()[stack_level][0]
    results = len(self._op.results) * [None]

    for i, path in enumerate(self._init_arg_names):
      arg = _FromCirctValue(self._op.results[i])
      if '.' in path:
        parts = path.split('.')
        base_name = parts[0]
        obj = s.f_locals[base_name]
        for part in parts[1:-1]:
          obj = getattr(obj, part)
        results[i] = getattr(obj, parts[-1])
        setattr(obj, parts[-1], arg)
      else:
        results[i] = s.f_locals[path]
        s.f_locals[path] = arg

    scf.YieldOp(results)

    self._ip.__exit__(exc_type, exc_value, traceback)

    for varname, _ in s.f_locals.items():
      if varname not in self._scope or self._scope[varname] is None:
        s.f_locals[varname] = None

    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))


class Foreach(For):
  """
  Iterates over one or more arrays simultaneously, providing the current index
  and array values for each iteration. All arrays must have the same size.
  The accessed values are passed by value, i.e., assigning to them does not
  update the value in the original array.

  Example:
  
  ```
  with Foreach(arr0, [Integer(1) for _ in range(3)]) as (i, a0, a1):
    v += a0 + a1 + i
  consumer(v)
  ```

  - arr0: first input array to iterate over
  - [Integer(1) for _ in range(3)]: second input array (automatically converted
    to an RTG array)
  - i: current iteration index
  - a0: current value from first array
  - a1: current value from second array
  """

  def __init__(self, *args: Array):
    if len(args) == 0:
      raise ValueError("must provide at least one argument")

    self._args = args
    super().__init__(self._args[0].size())

  def _return_on_enter(self) -> Value:
    accesses = [self._index] + [arr[self._index] for arr in self._args]
    return tuple(accesses)
