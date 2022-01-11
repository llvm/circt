#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pycde.devicedb import PrimitiveDB, PhysicalRegion

from .module import _SpecializedModule
from .pycde_types import types
from .instance import Instance

import mlir
import mlir.ir as ir
import mlir.passmanager

import circt
import circt.dialects.msft
import circt.support

from contextvars import ContextVar
import os
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
      "mod", "modules", "name", "passed", "_module_symbols", "_symbol_modules",
      "_old_system_token", "_symbols", "_generate_queue", "_primdb",
      "_output_directory", "files"
  ]

  PASSES = """
    msft-partition, msft-wire-cleanup,
    lower-msft-to-hw{{tops={tops} verilog-file={verilog_file} tcl-file={tcl_file}}},
    lower-seq-to-sv,hw.module(prettify-verilog),hw.module(hw-cleanup)
  """

  def __init__(self,
               modules,
               primdb: PrimitiveDB = None,
               name: str = "PyCDESystem",
               output_directory: str = None):
    self.passed = False
    self.mod = ir.Module.create()
    self.modules = list(modules)
    self.name = name
    self._module_symbols: dict[_SpecializedModule, str] = {}
    self._symbol_modules: dict[str, _SpecializedModule] = {}
    self._symbols: typing.Set[str] = None
    self._generate_queue = []
    self.files: typing.Set[str] = set()

    self._primdb = primdb

    if output_directory is None:
      output_directory = os.path.join(os.getcwd(), self.name)
    self._output_directory = output_directory

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

  def create_symbol(self, basename: str) -> str:
    """Create a unique symbol and add it to the cache. If it is to be preserved,
    the caller must use it as the symbol on a top-level op."""
    ctr = 0
    ret = basename
    while ret in self.symbols:
      ctr += 1
      ret = basename + "_" + str(ctr)
    self.symbols[ret] = None
    return ret

  def create_physical_region(self, name: str = None):
    with self._get_ip():
      physical_region = PhysicalRegion(name)
    return physical_region

  def _create_circt_mod(self, spec_mod: _SpecializedModule, create_cb):
    """Wrapper for a callback (which actually builds the CIRCT op) which
    controls all the bookkeeping around CIRCT module ops."""
    if spec_mod in self._module_symbols:
      return
    symbol = self.create_symbol(spec_mod.name)
    # Bookkeeping.
    self._module_symbols[spec_mod] = symbol
    self._symbol_modules[symbol] = spec_mod
    # Build the correct op.
    op = create_cb(symbol)
    # Install the op in the cache.
    self._symbols[symbol] = op
    # Add to the generation queue, if necessary.
    if isinstance(op, circt.dialects.msft.MSFTModuleOp):
      self._generate_queue.append(spec_mod)
      file_name = spec_mod.modcls.__name__ + ".sv"
      self.files.add(os.path.join(self._output_directory, file_name))
      op.fileName = ir.StringAttr.get(file_name)

  def _get_symbol_module(self, symbol):
    """Get the _SpecializedModule for a symbol."""
    return self._symbol_modules[symbol]

  def _get_module_symbol(self, spec_mod):
    """Get the symbol for a module or its associated _SpecializedModule."""
    if not isinstance(spec_mod, _SpecializedModule):
      if not hasattr(spec_mod, "_pycde_mod"):
        raise TypeError("Expected _SpecializedModule or pycde module")
      spec_mod = spec_mod._pycde_mod
    if spec_mod not in self._module_symbols:
      return None
    return self._module_symbols[spec_mod]

  def _get_circt_mod(self, spec_mod):
    """Get the CIRCT module op for a PyCDE module."""
    return self.symbols[self._get_module_symbol(spec_mod)]

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

  @property
  def _passes(self):
    tops = ",".join(self.symbols.keys())
    verilog_file = self.name + ".sv"
    tcl_file = self.name + ".tcl"
    self.files.add(os.path.join(self._output_directory, verilog_file))
    self.files.add(os.path.join(self._output_directory, tcl_file))
    return self.PASSES.format(tops=tops,
                              verilog_file=verilog_file,
                              tcl_file=tcl_file).strip()

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

  def get_instance(self, mod_cls: object) -> Instance:
    return Instance(mod_cls, None, None, self, self._primdb)

  class DevNull:

    def write(self, *_):
      pass

  def run_passes(self):
    if self.passed:
      return
    if len(self._generate_queue) > 0:
      print("WARNING: running lowering passes on partially generated design!",
            file=sys.stderr)
    pm = mlir.passmanager.PassManager.parse(self._passes)
    # Invalidate the symbol cache
    self._symbols = None
    pm.run(self.mod)
    types.declare_types(self.mod)
    self.passed = True

  def emit_outputs(self):
    self.run_passes()
    circt.export_split_verilog(self.mod, self._output_directory)
