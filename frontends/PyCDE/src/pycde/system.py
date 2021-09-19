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

  __slots__ = [
      "mod", "passed", "_old_system_token", "_symbols", "_generate_queue"
  ]

  passes = [
      "lower-msft-to-hw", "lower-seq-to-sv", "hw-legalize-names",
      "hw.module(prettify-verilog)", "hw.module(hw-cleanup)"
  ]

  def __init__(self, modules):
    self.passed = False
    self.mod = ir.Module.create()
    self._symbols: typing.Set[str] = None
    self._generate_queue = []

    with self:
      [m._pycde_mod.create() for m in modules]

  def _get_ip(self):
    return ir.InsertionPoint(self.mod.body)

  @property
  def symbols(self) -> typing.Set[str]:
    if self._symbols is None:
      self._symbols = set()
      for op in self.mod.operation.regions[0].blocks[0]:
        if "sym_name" in op.attributes:
          self._symbols.add(mlir.ir.StringAttr(op.attributes["sym_name"]).value)
    return self._symbols

  def create_symbol(self, basename: str) -> str:
    ctr = 0
    ret = basename
    while ret in self.symbols:
      ctr += 1
      ret = basename + "_" + str(ctr)
    self.symbols.add(ret)
    return ret

  @staticmethod
  def current():
    bb = _current_system.get(None)
    if bb is None:
      raise RuntimeError("No PyCDE system currently active!")
    return bb

  def __enter__(self):
    self._old_system_token = _current_system.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_system.reset(self._old_system_token)

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

  def generate(self, generator_names=[], iters=None):
    i = 0
    with self:
      while len(self._generate_queue) > 0 and (iters is None or i < iters):
        m = self._generate_queue.pop()
        m.generate()
        i += 1

  def get_instance(self, mod_name: str) -> Instance:
    assert self.passed
    root_mod = self.get_module(mod_name)
    return Instance(root_mod, None, None, self)

  def walk_instances(self, root_mod, callback) -> None:
    """Walk the instance hierachy, calling 'callback' on each instance."""
    assert self.passed
    inst = Instance(root_mod, None, None, self)
    inst.walk_instances(callback)

  def run_passes(self):
    if self.passed:
      return
    pm = mlir.passmanager.PassManager.parse(",".join(self.passes))
    # Invalidate the symbol cache
    self._symbols = None
    pm.run(self.mod)
    types.declare_types(self.mod)
    self.passed = True

  def print_verilog(self, out_stream: typing.TextIO = sys.stdout):
    self.run_passes()
    circt.export_verilog(self.mod, out_stream)

  def print_tcl(self, top_module: str, out_stream: typing.TextIO = sys.stdout):
    self.run_passes()
    msft.export_tcl(self.get_module(top_module).operation, out_stream)
