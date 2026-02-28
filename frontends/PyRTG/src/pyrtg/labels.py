#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .core import Value, Type
from .rtg import rtg
from .integers import Integer
from .strings import String

from typing import Union


class Label(Value):
  """
  Represents an ISA Assembly label. It can be declared and then passed around
  like every value. To place a label at a specific location in a sequence call
  'place'. It is the user's responsibility to place a label such that if the
  label is used by an instruction in the fully randomized test, there exists
  exactly one placement of the label to not end up with ambiguity or usage of
  an undeclared label.
  """

  def __init__(self, value: ir.Value):
    self._value = value

  def declare(string: Union[str, String]) -> Label:
    """
    Declares a label with a fixed name. Labels returned by different calls to
    this function but with the same arguments refer to the same label.
    """

    if isinstance(string, str):
      return rtg.ConstantOp(rtg.LabelAttr.get(string))

    return rtg.StringToLabelOp(string)

  def declare_unique(string: Union[str, String]) -> Label:
    """
    Declares a unique label. This means, all usages of the value returned by this
    function will refer to the same label, but no other label declarations can
    conflict with this label, including labels returned by other calls to this
    function or fixed labels declared with 'declare_label'.
    """

    if isinstance(string, str):
      string = String(string)

    return rtg.LabelUniqueDeclOp(string)

  def place(
      self,
      visibility: rtg.LabelVisibility = rtg.LabelVisibility.LOCAL) -> None:
    """
    Places a declared label in a sequence or test.
    """

    return rtg.LabelOp(rtg.LabelVisibilityAttr.get(visibility), self._value)

  def get_type(self) -> Type:
    return LabelType()

  def _get_ssa_value(self) -> ir.Value:
    return self._value


class LabelType(Type):
  """
  Represents the type of label values.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, LabelType)

  def _codegen(self) -> ir.Type:
    return rtg.LabelType.get()
