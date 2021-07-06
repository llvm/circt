#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .types import types

import mlir
import mlir.ir
import mlir.passmanager

import circt
from circt.dialects import hw

import sys
import typing


class System:

  mod = None
  passes = ["lower-seq-to-sv", "hw-legalize-names", "hw.module(hw-cleanup)"]
  passed = False

  def __init__(self):
    self.mod = mlir.ir.Module.create()
    self.get_types()

    with mlir.ir.InsertionPoint(self.mod.body):
      self.declare_externs()
      hw.HWModuleOp(name='top',
                    input_ports=self.inputs,
                    output_ports=self.outputs,
                    body_builder=self.build)

  def declare_externs(self):
    pass

  def get_types(self):
    pass

  @property
  def body(self):
    return self.mod.body

  def print(self):
    self.mod.operation.print()

  def graph(self, short_names=True):
    import mlir.all_passes_registration
    pm = mlir.passmanager.PassManager.parse("view-op-graph{short-names=" +
                                            ("1" if short_names else "0") + "}")
    pm.run(self.mod)

  def generate(self, generator_names=[]):
    pm = mlir.passmanager.PassManager.parse("run-generators{generators=" +
                                            ",".join(generator_names) + "}")
    pm.run(self.mod)

  def run_passes(self):
    if self.passed:
      return
    pm = mlir.passmanager.PassManager.parse(",".join(self.passes))
    pm.run(self.mod)
    types.declare_types(self.mod)
    self.passed = True

  def print_verilog(self, out_stream: typing.TextIO = sys.stdout):
    self.run_passes()
    circt.export_verilog(self.mod, out_stream)
