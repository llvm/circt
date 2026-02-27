#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .core import Value, Type
from .rtg import rtg

from typing import Union


class String(Value):
  """
  Represents an immutable string value.
  """

  def __init__(self, value: Union[ir.Value, str]):
    if isinstance(value, str):
      self._value = rtg.ConstantOp(
          ir.StringAttr.get_typed(rtg.StringType.get(),
                                  value))._get_ssa_value()
    else:
      self._value = value

  def __add__(self, other: String) -> String:
    """
    String concatenation.
    """

    return rtg.StringConcatOp([self, other])

  @staticmethod
  def format(*args, delimiter: str = ' ') -> String:
    """
    Format and concatenate values into a string.
    
    Converts each argument to a String and concatenates them together using the
    provided delimiter (single space by default).
    Values are converted using their `to_string()` method if available or raise
    an exception.
    Python values are formatted using Python's built-in format function.
    
    :param args: Values to format and concatenate
    :param delimiter: Delimiter to use between argument values
    :return: Concatenated string result
    """

    def convert_to_string(val):
      if isinstance(val, String):
        return val
      if isinstance(val, Value):
        if not hasattr(val, "to_string"):
          raise TypeError(
              f"Value of type {type(val)} cannot be converted to a String")
        return val.to_string()
      return String(format(val))

    if not args:
      raise ValueError("at least one argument must be provided")

    delimiter_str = String(delimiter)
    result = []
    for i, arg in enumerate(args):
      if i > 0:
        result.append(delimiter_str)
      result.append(convert_to_string(arg))

    return rtg.StringConcatOp(result)

  def get_type(self) -> Type:
    return StringType()

  def _get_ssa_value(self) -> ir.Value:
    return self._value


class StringType(Type):
  """
  Represents the type of string values.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, StringType)

  def _codegen(self) -> ir.Type:
    return rtg.StringType.get()
