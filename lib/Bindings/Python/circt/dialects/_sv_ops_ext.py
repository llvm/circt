#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from mlir.ir import OpView, Attribute
from circt.dialects import sv


class IfDefOp:

  def __init__(self, cond: Attribute, *, loc=None, ip=None):
    operands = []
    results = []
    attributes = {"cond": cond}
    regions = 2
    super().__init__(
        self.build_generic(attributes=attributes,
                           results=results,
                           operands=operands,
                           successors=None,
                           regions=regions,
                           loc=loc,
                           ip=ip))
    self.regions[0].blocks.append()
    self.regions[1].blocks.append()


class WireOp:

  def __init__(self, data_type, *, loc=None, ip=None):
    OpView.__init__(
        self,
        self.build_generic(attributes={},
                           results=[data_type],
                           operands=[],
                           successors=None,
                           regions=0,
                           loc=loc,
                           ip=ip))

  @staticmethod
  def create(data_type):
    return sv.WireOp(data_type)


class AssignOp:

  @staticmethod
  def create(dest, src):
    return sv.AssignOp(dest=dest, src=src)
