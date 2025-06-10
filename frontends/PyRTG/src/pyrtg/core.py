#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir


class CodeGenObject:
  """
  This is the base class for classes that have to be visited during
  codegen. Not all such objects have to be root objects but codegen may depend
  on other objects having been visited.
  """

  _registry: list[CodeGenObject] = []

  def register(self) -> None:
    CodeGenObject._registry.append(self)

  def codegen_depends_on(self) -> list[CodeGenObject]:
    return []

  @classmethod
  def _codegen_all_instances(cls) -> None:
    processed: set[CodeGenObject] = set()

    def do_codegen(obj):
      for dependent in obj.codegen_depends_on():
        if dependent not in processed:
          do_codegen(dependent)
          processed.add(dependent)

      if obj not in processed:
        obj._codegen()
        processed.add(obj)

    for obj in cls._registry:
      do_codegen(obj)

  def _codegen(self):
    assert False, "must be implemented by the subclass"


class CodeGenRoot(CodeGenObject):
  """
  This is the base class for classes that have to be visited by the RTG tool
  during codegen.
  """

  def __new__(cls, *args, **kwargs):
    inst = super().__new__(cls)
    inst.register()
    return inst


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
