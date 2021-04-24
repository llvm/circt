#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class BackedgeBuilder:

  def __init__(self):
    self.edges = []
    self.n = 0
    print(f"initialized")

  def create(self, type):
    print(f"adding, n = {self.n}")
    from mlir.ir import Operation

    edge = Operation.create(f"TemporaryBackedge{self.n}", [type]).result
    self.n += 1
    self.edges.append(edge)

    print(
        f"added {edge.owner.name}, remaining edges = {len(self.edges)}, n = {self.n}"
    )
    return edge

  def remove(self, edge):
    print(f"removing {edge.owner.name}, n = {self.n}")
    self.edges.remove(edge)
    edge.owner.destroy()
    print(
        f"removed {edge.owner.name}, remaining edges = {len(self.edges)}, n = {self.n}"
    )

  def check(self):
    print(f"checking, remaining edges = {len(self.edges)}, n = {self.n}")
