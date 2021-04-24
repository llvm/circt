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
      builder = self.builders[repr(edge)]
      value = str(edge)
      instance = str(builder.operation)
      module = str(builder.module)

      import re
      value_ident = re.search("(%\w+) =", value)[1]
      module_decl = re.search("(rtl.module @.+\))", module)[1]

      msg = "Uninitialized ports remain in circuit!\n"
      msg += "Port:     " + value_ident + "\n"
      msg += "Instance: " + instance + "\n"
      msg += "Module:   " + module_decl + "\n"
      raise RuntimeError(msg)
