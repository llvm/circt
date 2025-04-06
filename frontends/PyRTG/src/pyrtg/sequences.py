#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import CodeGenRoot, Value
from .support import _FromCirctValue
from .base import ir, support
from .rtg import rtg


class SequenceDeclaration(CodeGenRoot):
  """
  This class is responsible for managing and generating RTG sequences. It
  encapsulates the sequence function, its argument types, and the source
  location where it was defined.
  """

  def __init__(self, sequence_func, arg_types: list[ir.Type]):
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
    seq = rtg.SequenceOp(self.name,
                         ir.TypeAttr.get(rtg.SequenceType.get(self.arg_types)))
    block = ir.Block.create_at_start(seq.bodyRegion, self.arg_types)
    with ir.InsertionPoint(block):
      self.sequence_func(*[_FromCirctValue(arg) for arg in block.arguments])

  def _get_ssa_value(self) -> ir.Value:
    return rtg.GetSequenceOp(rtg.SequenceType.get(self.arg_types),
                             self.name)._get_ssa_value()

  def get_type(self):
    return rtg.SequenceType.get(self.arg_types)


def sequence(*args: ir.Type, **kwargs):
  """
  Decorator for defining RTG sequence functions.

  Args:
    *args: The types of the sequence's parameters.
  """

  def wrapper(func):
    return SequenceDeclaration(func, list(args))

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
    assert len(args) > 0, "At least one argument must be provided"
    assert len(args) <= len(
        element_types
    ), f"Expected at most {len(element_types)} arguments, got {len(args)}"
    for arg, expected_type in zip(args, element_types):
      assert arg.get_type(
      ) == expected_type, f"Expected argument of type {expected_type}, got {arg.get_type()}"

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
      assert len(args) == len(
          element_types
      ), f"Expected {len(element_types)} arguments, got {len(args)}"
      for arg, expected_type in zip(args, element_types):
        assert arg.get_type(
        ) == expected_type, f"Expected argument of type {expected_type}, got {arg.get_type()}"

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

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  @property
  def element_types(self) -> list[ir.Type]:
    """
    Returns the list of elements types for this sequence.
    """

    type = support.type_to_pytype(self.get_type())
    return [type.get_element(i) for i in range(type.num_elements)]

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(*args: ir.Type) -> ir.Type:
    """
    Returns the sequence type with the given argument types.
    """

    return rtg.SequenceType.get(list(args))


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

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(*args: ir.Type) -> ir.Type:
    """
    Returns the randomized sequence type.
    """

    assert len(
        args) == 0, "RandomizedSequence type does not take type arguments"
    return rtg.RandomizedSequenceType.get()
