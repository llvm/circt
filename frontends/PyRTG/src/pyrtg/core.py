#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import ir


class CodeGenRoot:
  """
  This is the base class for classes that have to be visited by the RTG tool
  during codegen.
  """

  def _codegen(self):
    assert False, "must be implemented by the subclass"


class Value:
  """
  This class wraps around MLIR SSA values to provide a more Python native
  experience. Instead of having a value class that stores the type, classes
  deriving from this class represent specific types of values. Operations on
  those values can then be exposed as methods that can support more convenient
  bridging between Python values and MLIR values (e.g., accepting a Python
  integer and automatically building a ConstantOp in MLIR).
  """

  def get_type(self) -> ir.Type:
    assert False, "must be implemented by subclass"

  def type(*args: ir.Type) -> ir.Type:
    assert False, "must be implemented by subclass"

  def _get_ssa_value(self) -> ir.Value:
    assert False, "must be implemented by subclass"
