#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import AbstractContextManager


class BackedgeBuilder(AbstractContextManager):

  def __init__(self):
    self.edges = []
    self.builders = {}

  def create(self, type, instance_builder):
    from mlir.ir import Operation

    edge = Operation.create("TemporaryBackedge", [type]).result
    self.edges.append(edge)
    self.builders[repr(edge)] = instance_builder
    return edge

  def remove(self, edge):
    self.edges.remove(edge)
    self.builders.pop(repr(edge))
    edge.owner.erase()

  def __exit__(self, exc_type, exc_value, traceback):
    errors = []
    for edge in self.edges:
      # Build a nice error message about the uninitialized port.
      builder = self.builders[repr(edge)]
      instance = builder.operation
      module = builder.module

      import re

      value_ident = re.search("(%\w+) =", str(edge))[1]
      module_decl = re.search("(rtl.module @.+\))", str(module))[1]

      msg = "Port:     " + value_ident + "\n"
      msg += "Instance: " + str(instance) + "\n"
      msg += "Module:   " + module_decl

      # Clean up the IR and Python references.
      instance.erase()
      self.remove(edge)

      errors.append(msg)

    if errors:
      errors.insert(0, "Uninitialized ports remain in circuit!")
      raise RuntimeError("\n".join(errors))
