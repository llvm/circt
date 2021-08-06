#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .pycde_types import types
from .module import ModuleDefinition

import mlir
import mlir.ir
import mlir.passmanager

import circt
import circt.support
from circt.dialects import hw

import gc
import sys
import typing


class System:

  mod = None
  passes = [
      "lower-seq-to-sv", "hw-legalize-names", "hw.module(prettify-verilog)",
      "hw.module(hw-cleanup)"
  ]
  passed = False

  def __init__(self, modules=[]):
    self.modules = modules
    if not hasattr(self, "name"):
      self.name = "__pycde_temp_top"

    self.mod = mlir.ir.Module.create()
    self.system_mod = None

    self.build()

  def build(self):
    if self.system_mod is not None:
      return

    with mlir.ir.InsertionPoint(self.mod.body):
      self.system_mod = ModuleDefinition(modcls=None,
                                         name=self.name,
                                         input_ports=[],
                                         output_ports=[])

    # Add the module body. Don't use the `body_builder` to avoid using the
    # `BackedgeBuilder` it creates.
    bb = circt.support.BackedgeBuilder()
    with mlir.ir.InsertionPoint(self.system_mod.add_entry_block()), bb:
      self.mod_ops = set([m().operation.name for m in self.modules])
      hw.OutputOp([])
      # We don't care about the backedges since this module is supposed to be
      # temporary.
      bb.edges.clear()

  @property
  def body(self):
    return self.mod.body

  def print(self, *argv, **kwargs):
    self.mod.operation.print(*argv, **kwargs)

  def graph(self, short_names=True):
    import mlir.all_passes_registration
    pm = mlir.passmanager.PassManager.parse("view-op-graph{short-names=" +
                                            ("1" if short_names else "0") + "}")
    pm.run(self.mod)

  def generate(self, generator_names=[]):
    pm = mlir.passmanager.PassManager.parse("run-generators{generators=" +
                                            ",".join(generator_names) + "}")
    pm.run(self.mod)
    if self.system_mod is None:
      return
    sys_mod_block = self.system_mod.operation.regions[0].blocks[0]
    if all([op.operation.name not in self.mod_ops for op in sys_mod_block]):
      self.system_mod.operation.erase()
      self.system_mod = None
      gc.collect()

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
