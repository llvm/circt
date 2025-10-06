#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import Type
from .sequences import SequenceDeclaration

from typing import List, Tuple
from enum import Enum


class SideEffects(Enum):
  NONE = 0,
  READ = 1,
  WRITE = 2,
  READ_WRITE = 3,


class Instruction(SequenceDeclaration):
  """
  Represents an instruction sequence.
  """

  def __init__(self, sequence_func, arg_types_and_side_effects: List[Tuple[Type, SideEffects]]):
    super().__init__(sequence_func, [t for t, _ in arg_types_and_side_effects])
    self.side_effects = [se for _, se in arg_types_and_side_effects]

  def num_read_effects(self) -> int:
    """
    Returns the number of operands of this instruction.
    """

    return self.side_effects.count(SideEffects.READ)

  def __repr__(self):
    return f"Instruction<{self.name}, {self.arg_types}, {self.side_effects}>"


def instruction(args: List[Tuple[Type, SideEffects]], **kwargs):
  """
  Decorator for defining instructions.

  Args:
    args: The types of the instruction's operands and the side-effects on them.
  """

  def wrapper(func):
    return Instruction(func, args)

  return wrapper
