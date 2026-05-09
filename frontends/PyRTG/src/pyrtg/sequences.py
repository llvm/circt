#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import CodeGenObject, Value, Type
from .support import _FromCirctValue, _FromCirctType
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
    return Sequence(self._get_ssa_value())

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

  def _codegen(self) -> None:
    mlir_arg_types = [arg._codegen() for arg in self.arg_types]
    seq = rtg.SequenceOp(self.name,
                         ir.TypeAttr.get(rtg.SequenceType.get(mlir_arg_types)))
    block = ir.Block.create_at_start(seq.bodyRegion, mlir_arg_types)
    with ir.InsertionPoint(block):
      self.sequence_func(*[_FromCirctValue(arg) for arg in block.arguments])

  def _get_ssa_value(self) -> ir.Value:
    self.register()
    return rtg.GetSequenceOp(self.get_type()._codegen(),
                             self.name)._get_ssa_value()

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

  def __init__(self, value: ir.Value) -> Sequence:
    """
    Intended for library internal usage only.
    """

    self._value = value

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

    return rtg.SubstituteSequenceOp(self, args)

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

    return rtg.RandomizeSequenceOp(value)

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


class RandomizedSequence(Value):
  """
  Represents a randomized sequence value where all randomization constructs have
  been replaced with concrete values. It can be embedded into a test or another
  sequence.
  """

  def __init__(self, value: ir.Value) -> RandomizedSequence:
    """
    Intended for library internal usage only.
    """

    self._value = value

  def embed(self) -> None:
    """
    Embeds this randomized sequence at the current position in the test or
    sequence.
    """

    rtg.EmbedSequenceOp(self)

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
    return _FromCirctType(self._value.type)


class RandomizedSequenceType(Type):
  """
  Represents the type of randomized sequences.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, RandomizedSequenceType)

  def _codegen(self) -> ir.Type:
    return rtg.RandomizedSequenceType.get()
