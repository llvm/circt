#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class BackedgeBuilder:

  def __init__(self):
    self.edges = []
    self.builders = {}

  def create(self, type, instance_builder):
    from mlir.ir import Operation

    edge = Operation.create(f"TemporaryBackedge", [type]).result
    self.edges.append(edge)
    self.builders[repr(edge)] = instance_builder
    return edge

  def remove(self, edge):
    self.edges.remove(edge)
    del self.builders[repr(edge)]
    edge.owner.destroy()

  def check(self):
    for edge in self.edges:
      # Build a nice error message about the uninitialized port.
      builder = self.builders[repr(edge)]
      instance = builder.operation
      module = builder.module

      import re
      value_ident = re.search("(%\w+) =", str(edge))[1]
      module_decl = re.search("(rtl.module @.+\))", str(module))[1]

      msg = "Uninitialized ports remain in circuit!\n"
      msg += "Port:     " + value_ident + "\n"
      msg += "Instance: " + str(instance) + "\n"
      msg += "Module:   " + module_decl + "\n"

      # Clean up the IR and Python references.
      self.remove(edge)
      instance.destroy()
      del builder
      del instance
      del edge

      raise RuntimeError(msg)
