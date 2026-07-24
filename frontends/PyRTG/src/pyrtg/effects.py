#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from .base import ir
from .rtg import rtg
from .core import CodeGenObject, Value, Type
from .support import _FromCirctValue, _FromCirctType

# ---------------------------------------------------------------------------
# Continuation types / values
# ---------------------------------------------------------------------------


class ContinuationType(Type):
  """Type of a first-class continuation value.

  resume_type is the type passed back by rtg.resume (or VoidType / NoneType
  for void effects).
  """

  def __init__(self, resume_type: Type):
    self.resume_type = resume_type

  def __eq__(self, other) -> bool:
    return (isinstance(other, ContinuationType) and
            self.resume_type == other.resume_type)

  def _codegen(self) -> ir.Type:
    return rtg.ContinuationType.get(self.resume_type._codegen())


class VoidType(Type):
  """Elaboration-time void (no value returned by handler)."""

  def __eq__(self, other) -> bool:
    return isinstance(other, VoidType)

  def _codegen(self) -> ir.Type:
    return ir.NoneType.get()


class Continuation(Value):
  """A first-class continuation value (received by control handlers)."""

  def __init__(self, value: ir.Value):
    self._value = value

  def resume(self, result: Optional[Value] = None) -> None:
    """Resume this continuation, optionally passing a result value."""
    if result is not None:
      rtg.ResumeOp(self._value, value=result._get_ssa_value())
    else:
      rtg.ResumeOp(self._value)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    return _FromCirctType(self._value.type)


# ---------------------------------------------------------------------------
# Effect declaration
# ---------------------------------------------------------------------------


class EffectDeclaration(CodeGenObject):
  """A declared algebraic effect.

  Created by the @effect decorator.  No MLIR is emitted by the constructor;
  rtg.effect is emitted once when _codegen() is called.
  """

  def __init__(self, name: str, inputs: list[Type], result: Type):
    self.name = name
    self.inputs = inputs  # list of Type
    self.result = result  # Type (VoidType if void)

  def _codegen(self) -> None:
    input_types = [t._codegen() for t in self.inputs]
    result_types = ([] if isinstance(self.result, VoidType) else
                    [self.result._codegen()])
    fn_type = ir.FunctionType.get(input_types, result_types)
    rtg.EffectOp(self.name, ir.TypeAttr.get(fn_type))

  def codegen_depends_on(self):
    return []


def effect(inputs: list[Type], result: Type = None):
  """Decorator that declares an algebraic effect.

  Args:
    inputs: Types of the values the performer passes to the handler.
    result: Type of the value the handler passes back via resume.
             Defaults to VoidType() if omitted.

  Example::

    @effect(inputs=[IntegerType(64)], result=IntegerType(64))
    def choose(candidates) -> Integer: ...
  """
  if result is None:
    result = VoidType()

  def wrapper(func):
    decl = EffectDeclaration(func.__name__, inputs, result)
    decl.register()
    return decl

  return wrapper


# ---------------------------------------------------------------------------
# perform
# ---------------------------------------------------------------------------


def perform(decl: EffectDeclaration, *args: Value) -> Optional[Value]:
  """Perform an effect.

  Dispatches to the innermost matching handler installed by an enclosing
  effect_handler scope.  If no handler is in scope, elaboration fails.

  Args:
    decl: The EffectDeclaration produced by @effect.
    *args: Values to pass to the handler (must match decl.inputs).

  Returns:
    The value passed back by the handler, or None for void effects.
  """
  decl.register()
  operand_values = [a._get_ssa_value() for a in args]
  is_void = isinstance(decl.result, VoidType)
  result_type = None if is_void else decl.result._codegen()
  op = rtg.PerformOp(result_type, decl.name, operand_values)
  if is_void:
    return None
  return _FromCirctValue(op._get_ssa_value())


# ---------------------------------------------------------------------------
# effect_handler scope
# ---------------------------------------------------------------------------


class HandlerScope:
  """Context manager that installs one or more effect handlers.

  Handlers are functions whose arity determines whether they are
  *simple* (auto-resume) or *control* (explicit Continuation):

  - Simple: arity == len(effect.inputs)  — framework calls k.resume(return_val)
  - Control: arity == len(effect.inputs) + 1 (last arg is Continuation)

  Usage::

    with effect_handler(choose, my_choose_handler):
        result = perform(choose, candidates)

    with effect_handler([(choose, ch), (log, lh)]):
        ...
  """

  def __init__(
      self,
      entries: "EffectDeclaration | list[tuple[EffectDeclaration, Callable]]",
      handler: "Optional[Callable]" = None,
  ):
    if isinstance(entries, EffectDeclaration):
      # Single-effect shorthand: effect_handler(decl, handler_fn)
      assert handler is not None, "handler function required"
      self._entries = [(entries, handler)]
    else:
      assert handler is None, "handler must be None when passing a list"
      self._entries = entries

  def __enter__(self):
    # Validate arities before emitting any IR.
    for decl, fn in self._entries:
      nparams = len(inspect.signature(fn).parameters)
      expected_simple = len(decl.inputs)
      expected_control = expected_simple + 1
      if nparams not in (expected_simple, expected_control):
        raise TypeError(
            f"Handler for @{decl.name} has {nparams} parameters; expected "
            f"{expected_simple} (simple) or {expected_control} (control).")

    # Ensure effect ops are in the module before the handle op.
    for decl, _ in self._entries:
      decl.register()

    # Build the list of effect symbol attributes.
    effect_syms = [
        ir.FlatSymbolRefAttr.get(decl.name) for decl, _ in self._entries
    ]
    effects_attr = ir.ArrayAttr.get(effect_syms)

    # Create the rtg.effect_handlers op; insert body block and set insertion point.
    self._handle_op = rtg.WithHandlersOp(effects_attr, len(self._entries))
    op = self._handle_op.operation
    body_block = ir.Block.create_at_start(op.regions[0], [])
    self._body_ip = ir.InsertionPoint(body_block)
    self._body_ip.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    # Terminate the body region.
    rtg.YieldOp([])
    self._body_ip.__exit__(exc_type, exc_value, traceback)

    # Emit each handler region.
    op = self._handle_op.operation
    for i, (decl, fn) in enumerate(self._entries):
      # Compute block argument types: effect inputs + continuation.
      arg_types = [t._codegen() for t in decl.inputs]
      is_void = isinstance(decl.result, VoidType)
      resume_type = (ir.NoneType.get() if is_void else decl.result._codegen())
      cont_type = rtg.ContinuationType.get(resume_type)
      arg_types.append(cont_type)

      handler_block = ir.Block.create_at_start(op.regions[i + 1], arg_types)
      with ir.InsertionPoint(handler_block):
        # Convert block args to Python values.
        py_args = [
            _FromCirctValue(handler_block.arguments[j])
            for j in range(len(decl.inputs))
        ]
        cont_val = Continuation(handler_block.arguments[len(decl.inputs)])

        nparams = len(inspect.signature(fn).parameters)
        is_control = nparams == len(decl.inputs) + 1

        if is_control:
          fn(*py_args, cont_val)
        else:
          # Simple handler: auto-resume with return value.
          ret = fn(*py_args)
          if is_void:
            cont_val.resume()
          else:
            assert ret is not None, \
                f"Simple handler for @{decl.name} must return a value"
            cont_val.resume(ret)

        rtg.YieldOp([])


def effect_handler(
    entries: "EffectDeclaration | list[tuple[EffectDeclaration, Callable]]",
    handler: "Optional[Callable[..., Any]]" = None,
) -> HandlerScope:
  """Install one or more effect handlers for a body scope.

  Returns a HandlerScope context manager.  Use as::

    with effect_handler(my_effect, my_handler):
        perform(my_effect, arg)
  """
  return HandlerScope(entries, handler)
