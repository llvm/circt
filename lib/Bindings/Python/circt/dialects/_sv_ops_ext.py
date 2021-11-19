#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.ir as _ir


class IfDefOp:

  def __init__(self, cond: _ir.Attribute, *, loc=None, ip=None):
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
