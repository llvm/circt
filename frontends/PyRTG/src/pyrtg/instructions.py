#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .core import Value, Type
from .sequences import SequenceDeclaration, Sequence

from typing import List, Tuple, Callable, Union, Optional
from enum import Enum


class SideEffect(Enum):
  NONE = 0,
  READ = 1,
  WRITE = 2,
  READ_WRITE = 3,


class Instruction(SequenceDeclaration):
  """
  Represents an instruction sequence.
  """

  def __init__(self, sequence_func, return_val_func: Callable[[Type], Value], arg_types_and_side_effects: List[Tuple[Type, SideEffect]]):
    super().__init__(sequence_func, [t for t, _ in arg_types_and_side_effects])
    self.return_val_func = return_val_func
    self.side_effects = [se for _, se in arg_types_and_side_effects]

  def num_read_effects(self) -> int:
    """
    Returns the number of operands of this instruction.
    """

    return self.side_effects.count(SideEffect.READ) + self.side_effects.count(SideEffect.READ_WRITE)

  def __repr__(self):
    return f"Instruction<{self.name}, {self.arg_types}, {self.side_effects}>"

  def __call__(self, *args: Value) -> Optional[Union[Value, List[Value]]]:
    # Destination registers are passed (reference-style).
    if len(args) == len(self.arg_types):
      self.sequence_func(*args)
      return

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
    return super().get()


def instruction(return_val_func: Callable[[Type], Value], args: List[Tuple[Type, SideEffect]], **kwargs):
  """
  Decorator for defining instructions.

  Args:
    args: The types of the instruction's operands and the side-effects on them.
  """

  def wrapper(func):
    return Instruction(func, return_val_func, args)

  return wrapper
