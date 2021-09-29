#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import builtins

from pycde.module import module
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
  """The 'System' contains the user's design and some private bookkeeping. On
  construction, specify a list of 'root' modules which you wish to generate.
  Upon generation, the design will be fleshed out and all the dependent modules
  will be created.

  'System' also has methods to run through the CIRCT lowering, output tcl, and
  output SystemVerilog."""

  __slots__ = [
      "mod", "modules", "passed", "_module_symbols", "_old_system_token",
      "_symbols", "_generate_queue"
  ]

  passes = [
      "lower-msft-to-hw", "lower-seq-to-sv", "hw-legalize-names",
      "hw.module(prettify-verilog)", "hw.module(hw-cleanup)"
  ]

  def __init__(self, modules):
    self.passed = False
    self.mod = ir.Module.create()
    self.modules = list(modules)
    self._module_symbols = {}
    self._symbols: typing.Set[str] = None
    self._generate_queue = []

    with self:
      [m._pycde_mod.create() for m in modules]

  def _get_ip(self):
    return ir.InsertionPoint(self.mod.body)

  # TODO: Return a read-only proxy.
  @property
  def symbols(self) -> typing.Dict[str, ir.Operation]:
    """Get the set of top level symbols in the design. Read from a cache which
    will be invalidated whenever control is given to CIRCT."""
    if self._symbols is None:
      self._symbols = dict()
      for op in self.mod.operation.regions[0].blocks[0]:
        if "sym_name" in op.attributes:
          self._symbols[mlir.ir.StringAttr(
              op.attributes["sym_name"]).value] = op
    return self._symbols

  def create_symbol(self, basename: str, module_cls=None) -> str:
    """Create a unique symbol and add it to the cache. If it is to be preserved,
    the caller must use it as the symbol on a top-level op."""
    ctr = 0
    ret = basename
    while ret in self.symbols:
      ctr += 1
      ret = basename + "_" + str(ctr)
    self.symbols[ret] = None
    if module_cls is not None:
      self._module_symbols[ret] = module_cls
    return ret

  def _notify_symbol(self, symbol: str, op: ir.Operation):
    assert symbol in self._symbols
    assert self._symbols[symbol] is None
    self._symbols[symbol] = op

  @staticmethod
  def current():
    """Get the top-most system in the stack created by `with System()`."""
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
    """Fully generate the system unless iters is specified. Iters specifies the
    number of generators to run. Useful for debugging. Maybe."""
    i = 0
    with self:
      while len(self._generate_queue) > 0 and (iters is None or i < iters):
        m = self._generate_queue.pop()
        m.generate()
        i += 1
    return len(self._generate_queue)

  def get_module(self, symbol):
    return self._module_symbols[symbol]

  def get_instance(self, mod_cls: object) -> Instance:
    assert self.passed
    return Instance(mod_cls, None, None, self)

  def run_passes(self):
    if self.passed:
      return
    if len(self._generate_queue) > 0:
      print("WARNING: running lowering passes on partially generated design!",
            file=sys.stderr)
    pm = mlir.passmanager.PassManager.parse(",".join(self.passes))
    # Invalidate the symbol cache
    self._symbols = None
    pm.run(self.mod)
    types.declare_types(self.mod)
    self.passed = True

    # Run through all the known modules and re-assign the circt_mod in
    # _SpecializedModule.
    for (symbol, mod) in self._module_symbols.items():
      if symbol in self.symbols:
        mod._pycde_mod.circt_mod = self.symbols[symbol]
      else:
        mod._pycde_mod.circt_mod = None

  def print_verilog(self, out_stream: typing.TextIO = sys.stdout):
    self.run_passes()
    circt.export_verilog(self.mod, out_stream)

  def print_tcl(self, top_module: type, out_stream: typing.TextIO = sys.stdout):
    self.run_passes()
    msft.export_tcl(top_module._pycde_mod.circt_mod, out_stream)
