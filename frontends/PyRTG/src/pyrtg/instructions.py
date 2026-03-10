#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import Value, Type
from .sequences import SequenceDeclaration, Sequence

from typing import List, Tuple, Callable, Union, Optional
from enum import Enum


class SideEffect(Enum):
  """
  Enumeration of side effects for instruction operands.

  Defines how an instruction interacts with its operands:
  - UNSPECIFIED: The side effect is unknown.
  - PURE: The operand has no side effects (e.g., immediates)
  - READ: The operand is read from (e.g., source register)
  - WRITE: The operand is written to (e.g., destination register)
  - READ_WRITE: The operand is both read from and written to (e.g., combined source/destination register)
  - JUMP: The operand is a (conditional) jump target (e.g., label, immediate)
  """

  UNSPECIFIED = -1,
  PURE = 0,
  READ = 1,
  WRITE = 2,
  READ_WRITE = 3,
  JUMP = 4,


class Instruction(SequenceDeclaration):
  """
  Represents an instruction with typed operands and side effects.

  An Instruction is a specialized SequenceDeclaration that associates each operand
  with a type and a side effect. This allows the instruction to automatically
  allocate destination registers if desired and allowes the user to determine
  which operands are source or destination registers.
  """

  def __init__(self, sequence_func, return_val_func: Callable[[Type], Value],
               arg_types_and_side_effects: List[Tuple[Type, SideEffect]]):
    super().__init__(sequence_func, [t for t, _ in arg_types_and_side_effects])
    self.return_val_func = return_val_func
    self.side_effects = [se for _, se in arg_types_and_side_effects]

  def num_read_effects(self) -> int:
    """
    Returns the number of operands of this instruction.
    """

    return self.side_effects.count(SideEffect.READ) + self.side_effects.count(
        SideEffect.READ_WRITE)

  def __repr__(self):
    return f"Instruction<{self.name}, {self.arg_types}, {self.side_effects}>"

  def __call__(self, *args: Value) -> Optional[Union[Value, List[Value]]]:
    """
    Embeds the instruction into the instruction stream with the given arguments.

    This method supports two calling conventions:

    1. If the number of arguments matches the total number of operands (including
       destination registers), all operands are passed directly to the sequence
       function and no value is returned.
    2. If fewer arguments are provided (only source operands), destination registers
       are automatically allocated, and the allocated registers are returned.

    :param args: Operands for the instruction. The number of arguments should
                 match the number of READ and READ_WRITE side effects, unless all
                 operands (including WRITE operands) are being passed explicitly.
    :return: None if all operands were passed explicitly, otherwise returns the allocated
             destination register(s). Returns a single Value if there's one destination,
             or a List[Value] if there are multiple destinations.
    """

    # Destination registers are passed.
    if len(args) == len(self.arg_types):
      self.sequence_func(*args)
      return

    num_non_write_operands = sum(
        1 for se in self.side_effects if se != SideEffect.WRITE)
    if len(args) != num_non_write_operands:
      raise ValueError(
          f"Expected {num_non_write_operands} arguments (excluding WRITE operands), but got {len(args)}"
      )

    new_args = []
    results = []
    args_iter = iter(args)
    for arg_type, se in zip(self.arg_types, self.side_effects):
      if se == SideEffect.WRITE:
        reg = self.return_val_func(arg_type)
        new_args.append(reg)
        results.append(reg)
      else:
        new_args.append(next(args_iter))

    self.sequence_func(*new_args)

    if len(results) == 1:
      return results[0]
    return results

  def as_seq(self) -> Sequence:
    """
    Returns the instruction sequence.

    :return: The instruction sequence.
    """

    return super().get()


def instruction(return_val_func: Callable[[Type], Value],
                args: List[Tuple[Type, SideEffect]], **kwargs):
  """
  Decorator for defining instructions.

  :param args: The types of the instruction's operands and the side-effects on them.
  """

  def wrapper(func):
    return Instruction(func, return_val_func, args)

  return wrapper
