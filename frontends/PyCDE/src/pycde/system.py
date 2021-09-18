#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .pycde_types import types
from .instance import Instance

import mlir
import mlir.ir as ir
import mlir.passmanager

import circt
import circt.support
from circt.dialects import hw
from circt import msft

from contextvars import ContextVar
import sys
import typing

_current_system = ContextVar("current_pycde_system")


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
      self.name = "__pycde_system_mod_instances"

    self.mod = ir.Module.create()
    self._generate_queue = []

    self.build()

  def _get_ip(self):
    return ir.InsertionPoint(self.mod.body)

  @staticmethod
  def current():
    bb = _current_system.get(None)
    if bb is None:
      raise RuntimeError("No PyCDE system currently active!")
    return bb

  def __enter__(self):
    self.old_system_token = _current_system.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_system.reset(self.old_system_token)

  def build(self):
    with self:
      [m._pycde_mod.create() for m in self.modules]

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

  def generate(self, generator_names=[], iters=100):
    with self:
      for i in range(iters):
        if len(self._generate_queue) == 0:
          return
        m = self._generate_queue.pop()
        m.generate()

  def get_module(self, mod_name: str) -> hw.HWModuleOp:
    """Find the hw.module op with the specified name."""
    for op in self.mod.body:
      if not isinstance(op, (hw.HWModuleOp, hw.HWModuleExternOp)):
        continue
      op_modname = ir.StringAttr(op.attributes["sym_name"]).value
      if op_modname == mod_name:
        return op

  def get_instance(self, mod_name: str) -> Instance:
    assert self.passed
    root_mod = self.get_module(mod_name)
    return Instance(root_mod, None, None, self)

  def walk_instances(self, mod_name: str, callback) -> None:
    """Walk the instance hierachy, calling 'callback' on each instance."""
    assert self.passed
    root_mod = self.get_module(mod_name)
    inst = Instance(root_mod, None, None, self)
    inst.walk_instances(callback)

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

  def print_tcl(self, top_module: str, out_stream: typing.TextIO = sys.stdout):
    self.run_passes()
    msft.export_tcl(self.get_module(top_module).operation, out_stream)
