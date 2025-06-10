#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import ir


class CodeGenRoot:
  """
  This is the base class for classes that have to be visited by the RTG tool
  during codegen.
  """

  _already_generated: bool = False

  def _codegen(self):
    assert False, "must be implemented by the subclass"


class Type:
  """
  This is the base class for classes representing types of 'Value's.
  Those are essentially wrappers around corresponding MLIR types and allow
  constructing types and querying type properties without having an MLIR
  context registered.
  """

  def _codegen(self) -> ir.Type:
    assert False, "must be implemented by subclass"


class Value:
  """
  This class wraps around MLIR SSA values to provide a more Python native
  experience. Instead of having a value class that stores the type, classes
  deriving from this class represent specific types of values. Operations on
  those values can then be exposed as methods that can support more convenient
  bridging between Python values and MLIR values (e.g., accepting a Python
  integer and automatically building a ConstantOp in MLIR).
  """

  def get_type(self) -> Type:
    assert False, "must be implemented by subclass"

  def _get_ssa_value(self) -> ir.Value:
    assert False, "must be implemented by subclass"
