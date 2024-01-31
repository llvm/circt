#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .constructs import Wire
from .module import Module, ModuleBuilder, PortProxyBase, generator
from .system import System
from .types import (Bits, Bundle, BundledChannel, ChannelDirection, Channel,
                    ClockType, StructType, Type)

from .circt.dialects import hw
from .circt.ir import FlatSymbolRefAttr, InsertionPoint, StringAttr
from .circt.support import attribute_to_var

from pathlib import Path
from typing import Callable, Dict, List, Optional, get_type_hints

import os
import shutil

SvSupportPath = None
if "IBIS_SVSUPPORT" in os.environ:
  SvSupportPath = Path(os.environ["IBIS_SVSUPPORT"])


class IbisMethod:

  def __init__(self, name: Optional[str], arg_types: Dict[str, Type],
               return_type: Optional[Type]):
    self.name = name
    self.arg_types = arg_types
    if return_type is None:
      return_type = Bits(0)
    self.return_type = return_type

  def get_esi_bundle_type(self) -> Bundle:
    args = StructType(self.arg_types)
    return Bundle([
        BundledChannel("arg", ChannelDirection.FROM, args),
        BundledChannel("result", ChannelDirection.TO, self.return_type)
    ])


def method(func: Callable):
  """Decorator to mark a function as an Ibis function."""

  type_hints = get_type_hints(func)
  arg_types: Dict[str, Type] = {}
  return_type: Optional[Type] = None
  for name, type in type_hints.items():
    if not isinstance(type, Type):
      raise TypeError(
          f"Argument {name} of method {func.__name__} is not a CIRCT type")
    if name == "return":
      return_type = type
    else:
      arg_types[name] = type
  return IbisMethod(None, arg_types, return_type)


class IbisClassBuilder(ModuleBuilder):

  @staticmethod
  def package(sys: System):
    """Copy in the necessary support files."""
    if SvSupportPath is None:
      raise RuntimeError(
          "IBIS_SVSUPPORT environment variable not set. Cannot copy in "
          "necessary support files.")

    # TODO: make this configurable.
    platform = "SimOnly"
    outdir = sys.hw_output_dir
    files = SvSupportPath.glob("*.sv")
    for idx, f in enumerate(files):
      shutil.copy(SvSupportPath / f, outdir / f"{idx}_{f}")
    for f in (SvSupportPath / "HAL" / platform).glob("*.sv"):
      shutil.copy(f, outdir)

  @property
  def circt_mod(self):
    """Get the raw CIRCT operation for the module definition. DO NOT store the
    returned value!!! It needs to get reaped after the current action (e.g.
    instantiation, generation). Memory safety when interacting with native code
    can be painful."""

    from .system import System
    sys: System = System.current()
    ret = sys._op_cache.get_circt_mod(self)
    if ret is None:
      return sys._create_circt_mod(self)
    return ret

  def scan_cls(self) -> None:
    """Scan the class for Ibis methods and register them."""

    self.methods: List[IbisMethod] = []
    for name, value in self.cls_dct.items():
      if isinstance(value, IbisMethod):
        value.name = name
        self.methods.append(value)

    self.src_file = None
    if hasattr(self.modcls, "src_file"):
      self.src_file = Path(self.modcls.src_file).resolve()

    self.clocks = {0}
    self.resets = {1}
    self.inputs = [
        ("clk", ClockType()),
        ("rst", Bits(1)),
    ]
    self.outputs = [(m.name, m.get_esi_bundle_type())
                    for m in self.methods
                    if m.name is not None]
    # self.generators = [lambda ports, self=self: self.generate(ports)]
    self.generators = {"default": generator(self.generate_wrapper)}

  def create_op(self, sys, symbol):
    """Creation callback for creating an Ibis method wrapper."""

    if hasattr(self.modcls, "metadata"):
      meta = self.modcls.metadata
      self.add_metadata(sys, symbol, meta)
    else:
      self.add_metadata(sys, symbol, None)

    if self.src_file is None:
      raise RuntimeError(
          f"Could not find source file for module {self.modcls.name}")

    imported = sys.import_mlir(open(self.src_file).read())
    if self.modcls.__name__ not in imported:
      raise RuntimeError(
          f"Could not find module {self.modcls.name} in {self.src_file}")
    self.imported_mod = imported[self.modcls.__name__]

    return hw.HWModuleOp(
        symbol,
        [(n, t._type) for (n, t) in self.inputs],
        [(n, t._type) for (n, t) in self.outputs],
        attributes=self.attributes,
        loc=self.loc,
        ip=sys._get_ip(),
    )

  def generate_wrapper(self, ports):
    System.current().add_packaging_step(IbisClassBuilder.package)

    # Instantiate the Ibis module and connect its outputs.
    exclude_ports = {
        "clk", "clk1", "rst_in", "stall_rate_in", "inspection_value_in",
        "stall_rate_valid_in"
    }
    ibis_inputs = {
        n: Wire(t, n)
        for (n, t) in self.imported_mod.inputs()
        if n not in exclude_ports
    }
    ibis_instance = self.imported_mod(
        clk=ports.clk.to_bit(),
        clk1=ports.clk.to_bit(),
        rst_in=ports.rst,
        stall_rate_in=None,
        inspection_value_in=None,
        stall_rate_valid_in=None,
        **ibis_inputs,
    )
    inst_outputs = ibis_instance.outputs()

    for m in self.methods:
      assert m.name is not None
      mname = m.name + "_"

      # Return side is FIFO.
      return_out_untyped = inst_outputs[mname + "result_out"]
      return_out = return_out_untyped.bitcast(m.return_type)

      empty_out = inst_outputs[mname + "empty_out"]
      return_chan, rdy = Channel(m.return_type).wrap(return_out, ~empty_out)
      ibis_inputs[mname + "rden_in"].assign(rdy & ~empty_out)

      # Pack the bundle and set it on my output port.
      bundle_type = m.get_esi_bundle_type()
      bundle, [(_, arg_chan)] = bundle_type.pack(result=return_chan)
      setattr(ports, m.name, bundle)

      # Call side is ready/valid.
      args_rdy = inst_outputs[mname + "rdy_out"]
      args, args_valid = arg_chan.unwrap(args_rdy)
      ibis_inputs[mname + "valid_in"].assign(args_valid)
      for name, type in m.arg_types.items():
        input_wire = ibis_inputs[mname + name + "_in"]
        input_wire.assign(args[name].bitcast(input_wire.type))

  @property
  def name(self) -> str:
    ibis_cls_name = super().name
    return ibis_cls_name + "_esi_wrapper"


class IbisClass(Module):
  """Base class to be extended for describing an Ibis class methodj."""

  BuilderType = IbisClassBuilder
