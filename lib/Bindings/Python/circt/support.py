#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import AbstractContextManager
from typing import List


class UnconnectedSignalError(RuntimeError):
  def __init__(self, module: str, port_names: List[str]):
    super().__init__(
        f"Ports {port_names} unconnected in design module {module}.")


class BackedgeBuilder(AbstractContextManager):

  def __init__(self):
    self.edges = []
    self.builders = {}

  def create(self, type, instance_builder):
    from mlir.ir import Operation

    edge = Operation.create("TemporaryBackedge", [type]).result
    self.edges.append(edge)
    self.builders[id(edge)] = instance_builder
    return edge

  def remove(self, edge):
    self.edges.remove(edge)
    self.builders.pop(id(edge))
    edge.owner.erase()

  def __exit__(self, exc_type, exc_value, traceback):
    errors = []
    for edge in self.edges:
      # Build a nice error message about the uninitialized port.
      builder = self.builders[id(edge)]
      instance = builder.operation
      module = builder.module

      for i in range(len(instance.operands)):
        if instance.operands[i] == edge:
          from mlir.ir import ArrayAttr, StringAttr
          arg_names = ArrayAttr(module.attributes["argNames"])
          port_name = "%" + StringAttr(arg_names[i]).value

      assert port_name, "Could not look up port name for backedge"

      # TODO: Make this use `UnconnectedSignalError`.
      msg = "Port:     " + port_name + "\n"
      msg += "Module:   " + str(module).split(" {")[0] + "\n"
      msg += "Instance: " + str(instance)

      # Clean up the IR and Python references.
      instance.erase()
      self.remove(edge)

      errors.append(msg)

    if errors:
      errors.insert(0, "Uninitialized ports remain in circuit!")
      raise RuntimeError("\n".join(errors))


class BuilderValue:
  """Class that holds a value, as well as builder and index of this value in
     the operand or result list. This can represent an OpOperand and index into
     OpOperandList or a OpResult and index into an OpResultList"""

  def __init__(self, value, builder, index):
    self.value = value
    self.builder = builder
    self.index = index
