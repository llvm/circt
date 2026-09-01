# ===============================================================================
# Magma Release License
# ===============================================================================
# MIT License

# Copyright (C) 2016 Stanford University.
# All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
""" This code includes modifications based on code licensed under the MIT license. 
Based on magma bit_pattern
https://github.com/phanrahan/magma/blob/master/magma/types/bit_pattern.py
"""

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import string

from .types import Bits
from .signals import BitVectorSignal, BitsSignal
from .constructs import Mux


class BitPat:

  def __init__(self, pattern):
    """
    Parses a bit pattern string `pattern` into the attributes `bits`,
    `mask`, `width`
    
    * `bits`  - the literal value, with don't cares being 0
    * `mask`  - the mask bits, with don't cares being 0 and cares being 1
    * `width` - the number of bits in the literal, including values and
                don't cares, but not including the white space and
                underscores
    """
    if pattern[0] != "b":
      raise ValueError("BitPat must be in binary and prefixed with "
                       "'b'")
    bits = 0
    mask = 0
    count = 0
    for digit in pattern[1:]:
      if digit == '_' or digit in string.whitespace:
        continue
      if digit not in "01?":
        raise ValueError(
            f"BitPat {pattern} contains illegal character: {digit}")
      mask = (mask << 1) + (0 if digit == "?" else 1)
      bits = (bits << 1) + (1 if digit == "1" else 0)
      count += 1
    self.bits = Bits(count)(bits)
    self.mask = Bits(count)(mask)
    self.width = count
    self.const = mask == ((1 << count) - 1)
    self.__hash = hash(pattern)

  def __eq__(self, other):
    if not isinstance(other, BitVectorSignal):
      raise TypeError("BitPat can only be compared to BitVectorSignal")
    if not isinstance(other, BitsSignal):
      other = other.as_bits(self.width)
    return self.mask == (other & self.mask)

  def __ne__(self, other):
    if not isinstance(other, BitVectorSignal):
      raise TypeError("BitPat can only be compared to BitVectorSignal")
    if not isinstance(other, BitsSignal):
      other = other.as_bits(self.width)
    return self.bits != (other & self.mask)

  def as_bits(self):
    if not self.const:
      raise TypeError("Can only convert BitPat with no don't cares to Bits")
    return self.bits

  def __hash__(self):
    return self.__hash


def dict_lookup(dict_, select, default):
  """
    Use `select` as an index into `dict` (similar to a case statement)

    `default` is used when `select` does not match any of the keys
  """
  output = default
  for key, value in dict_.items():
    output = Mux(key == select, output, value)
  output.name = "dict_lookup"
  return output
