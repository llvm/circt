#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from mlir.passmanager import PassManager
import mlir.ir

from _circt._esi import *
import circt

import sys
import os


class System(CppSystem):

  mod = None
  passes = [
      "lower-esi-ports", "lower-esi-to-physical", "lower-esi-to-rtl",
      "rtl-legalize-names", "rtl.module(rtl-cleanup)"
  ]
  passed = False

  def __init__(self):
    self.mod = mlir.ir.Module.create()
    super().__init__(self.mod)

  @property
  def body(self):
    return self.mod.body

  def print(self):
    self.mod.operation.print()

  def run_passes(self):
    if self.passed:
      return
    pm = PassManager.parse(",".join(self.passes))
    pm.run(self.mod)
    self.passed = True

  def print_verilog(self):
    self.run_passes()
    circt.export_verilog(self.mod, sys.stdout)
