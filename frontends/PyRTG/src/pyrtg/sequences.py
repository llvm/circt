#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import CodeGenContext, CodeGenObject, Value, Type
from .support import _FromCirctType
from .base import ir
from .rtg import rtg


class SequenceDeclaration(CodeGenObject):
  """
  This class is responsible for managing and generating RTG sequences. It
  encapsulates the sequence function, its argument types, and the source
  location where it was defined.
  """

  def __init__(self, sequence_func, arg_types: list[Type]):
    self.sequence_func = sequence_func
    self.arg_types = arg_types

  @property
  def name(self) -> str:
    return self.sequence_func.__name__

  def get(self) -> Sequence:
    """
    Returns a sequence value referring to this sequence declaration. It can be
    used for substitution, randomization, or passed as a value to other
    functions.
    """

    self.register()
    return Sequence(self._get_ssa_value(), self.get_type())

  def substitute(self, *args: Value) -> Sequence:
    """
    Creates a new sequence with the given arguments substituted.
    
    Args:
      *args: Values to substitute for the sequence's parameters.
    """

    return self.get().substitute(*args)

  def randomize(self, *args: Value) -> RandomizedSequence:
    """
    Randomizes this sequence, i.e., replaces all randomization constructs with
    concrete values.

    Args:
      *args: Values to substitute for the sequence's parameters.
    """

    return self.get().randomize(*args)

  def __call__(self, *args: Value) -> None:
    """
    Convenience method to substitute, randomize, and embed this sequence in one
    go.
    
    Args:
      *args: Values to substitute for the sequence's parameters.
    """

    self.get()(*args)

  def _codegen(self, context: CodeGenContext) -> None:
    self.context = context

    mlir_arg_types = [arg._codegen() for arg in self.arg_types]
    seq = ir.Operation.create(
        "rtg.sequence",
        attributes={
            "sym_name": ir.StringAttr.get(self.name),
            "sequenceType": ir.TypeAttr.get(rtg.SequenceType.get(mlir_arg_types)),
        },
        regions=1)
    block = ir.Block.create_at_start(seq.regions[0], mlir_arg_types)
    with ir.InsertionPoint(block):
      self.sequence_func(*[
          arg_type._wrap(arg)
          for arg_type, arg in zip(self.arg_types, block.arguments)
      ])

  def _get_ssa_value(self) -> ir.Value:
    self.register()
    return ir.Operation.create(
        "rtg.get_sequence",
        attributes={"sequence": ir.FlatSymbolRefAttr.get(self.name)},
        results=[self.get_type()._codegen()],
        regions=0).result

  def get_type(self) -> Type:
    return SequenceType(self.arg_types)


def sequence(args: list[Type], **kwargs):
  """
  Decorator for defining RTG sequence functions.

  Args:
    args: The types of the sequence's parameters.
  """

  def wrapper(func):
    return SequenceDeclaration(func, args)

  return wrapper


class Sequence(Value):
  """
  Represents a sequence value that can be substituted and randomized (i.e., all
  randomization constructs are replaced with concrete values). Once it is
  randomized it can be embedded into a test or another sequence.
  """

  def __init__(self, value: ir.Value, type: SequenceType = None) -> Sequence:
    """
    Intended for library internal usage only.
    """

    self._value = value
    self._type = type

  def substitute(self, *args: Value) -> Sequence:
    """
    Creates a new sequence with the given arguments substituted.
    
    Args:
      *args: Values to substitute for the sequence's parameters.
    """

    element_types = self.element_types
    if len(args) == 0:
      raise ValueError("At least one argument must be provided")

    if len(args) > len(element_types):
      raise ValueError(
          f"Expected at most {len(element_types)} arguments, got {len(args)}")

    for arg, expected_type in zip(args, element_types):
      if arg.get_type() != expected_type:
        raise TypeError(
            f"Expected argument of type {expected_type}, got {arg.get_type()}")

    result_type = SequenceType(element_types[len(args):])
    return Sequence(ir.Operation.create(
        "rtg.substitute_sequence",
        operands=[self._value] + [arg._get_ssa_value() for arg in args],
        results=[result_type._codegen()],
        regions=0).result, result_type)

  def randomize(self, *args: Value) -> RandomizedSequence:
    """
    Creates a randomized version (i.e., all randomization constructs are
    replaced with concrete values) of this sequence.

    Args:
      *args: Values to substitute for the sequence's parameters.
    """

    value = self
    element_types = self.element_types
    if len(element_types) > 0:
      if len(args) != len(element_types):
        raise TypeError(
            f"Expected {len(element_types)} arguments, got {len(args)}")

      for arg, expected_type in zip(args, element_types):
        if arg.get_type() != expected_type:
          raise TypeError(
              f"Expected argument of type {expected_type}, got {arg.get_type()}"
          )

      value = self.substitute(*args)

    return RandomizedSequence(ir.Operation.create(
        "rtg.randomize_sequence",
        operands=[value._get_ssa_value()],
        results=[rtg.RandomizedSequenceType.get()],
        regions=0).result)

  def __call__(self, *args: Value) -> None:
    """
    Convenience method to substitute, randomize, and embed this sequence in one
    go.
    
    Args:
      *args: Values to substitute for the sequence's parameters.
    """

    self.randomize(*args).embed()

  @property
  def element_types(self) -> list[Type]:
    """
    Returns the list of elements types for this sequence.
    """

    return self.get_type().element_types

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    if self._type is not None:
      return self._type
    return _FromCirctType(self._value.type)


class SequenceType(Type):
  """
  Represents the type of statically typed sequences.

  Fields:
    element_types: list[Type]
  """

  def __init__(self, element_types: list[Type]):
    self.element_types = element_types

  def __eq__(self, other) -> bool:
    return isinstance(
        other, SequenceType) and self.element_types == other.element_types

  def _codegen(self):
    return rtg.SequenceType.get([ty._codegen() for ty in self.element_types])

  def _wrap(self, value: ir.Value) -> Sequence:
    return Sequence(value, self)


class RandomizedSequence(Value):
  """
  Represents a randomized sequence value where all randomization constructs have
  been replaced with concrete values. It can be embedded into a test or another
  sequence.
  """

  def __init__(self, value: ir.Value,
               type: RandomizedSequenceType = None) -> RandomizedSequence:
    """
    Intended for library internal usage only.
    """

    self._value = value
    self._type = type

  def embed(self) -> None:
    """
    Embeds this randomized sequence at the current position in the test or
    sequence.
    """

    ir.Operation.create(
        "rtg.embed_sequence",
        operands=[self._value],
        regions=0)

  def __call__(self) -> None:
    """
    Convenience method to embed this sequence. Takes no arguments since the
    sequence is already fully sustituted.
    
    Args:
      *args: Must be empty, since randomized sequences cannot take arguments.
    """

    self.embed()

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    if self._type is not None:
      return self._type
    return _FromCirctType(self._value.type)


class RandomizedSequenceType(Type):
  """
  Represents the type of randomized sequences.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, RandomizedSequenceType)

  def _codegen(self) -> ir.Type:
    return rtg.RandomizedSequenceType.get()

  def _wrap(self, value: ir.Value) -> RandomizedSequence:
    return RandomizedSequence(value, self)
