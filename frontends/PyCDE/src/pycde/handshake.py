#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Any, List, Optional, Set, Tuple, Dict
import typing

from .module import Module, ModuleLikeBuilderBase, PortError
from .signals import BitsSignal, ChannelSignal, ClockSignal, Signal
from .system import System
from .support import (get_user_loc, _obj_to_attribute, obj_to_typed_attribute,
                      create_const_zero)
from .types import Channel

from .circt.dialects import handshake as raw_handshake
from .circt import ir


class FuncBuilder(ModuleLikeBuilderBase):
  """Defines how an ESI `PureModule` gets built."""

  @property
  def circt_mod(self):
    sys: System = System.current()
    ret = sys._op_cache.get_circt_mod(self)
    if ret is None:
      return sys._create_circt_mod(self)
    return ret

  def create_op(self, sys: System, symbol):
    if hasattr(self.modcls, "metadata"):
      meta = self.modcls.metadata
      self.add_metadata(sys, symbol, meta)
    else:
      self.add_metadata(sys, symbol, None)

    # If there are associated constants, add them to the manifest.
    if len(self.constants) > 0:
      constants_dict: Dict[str, ir.Attribute] = {}
      for name, constant in self.constants.items():
        constant_attr = obj_to_typed_attribute(constant.value, constant.type)
        constants_dict[name] = constant_attr
      with ir.InsertionPoint(sys.mod.body):
        from .dialects.esi import esi
        esi.SymbolConstantsOp(symbolRef=ir.FlatSymbolRefAttr.get(symbol),
                              constants=ir.DictAttr.get(constants_dict))

    assert len(self.generators) > 0

    if hasattr(self, "parameters") and self.parameters is not None:
      self.attributes["pycde.parameters"] = self.parameters
    # If this Module has a generator, it's a real module.
    return raw_handshake.FuncOp.create(
        symbol,
        [(p.name, p.type._type) for p in self.inputs],
        [(p.name, p.type._type) for p in self.outputs],
        attributes=self.attributes,
        loc=self.loc,
        ip=sys._get_ip(),
    )

  def generate(self):
    """Fill in (generate) this module. Only supports a single generator
    currently."""
    if len(self.generators) != 1:
      raise ValueError("Must have exactly one generator.")
    g: Generator = list(self.generators.values())[0]

    entry_block = self.circt_mod.add_entry_block()
    ports = self.generator_port_proxy(entry_block.arguments, self)
    with self.GeneratorCtxt(self, ports, entry_block, g.loc):
      outputs = g.gen_func(ports)
      if outputs is not None:
        raise ValueError("Generators must not return a value")

      ports._check_unconnected_outputs()
      raw_handshake.ReturnOp([o.value for o in ports._output_values])

  def instantiate(self, module_inst, inputs, instance_name: str):
    """"Instantiate this Func from ESI channels. Check that the input types
    match expectations."""

    port_input_lookup = {port.name: port for port in self.inputs}
    circt_inputs: List[Optional[ir.Value]] = [None] * len(self.inputs)
    remaining_inputs = set(port_input_lookup.keys())
    clk = None
    rst = None
    for name, signal in inputs.items():
      if name == "clk":
        if not isinstance(signal, ClockSignal):
          raise PortError("'clk' must be a clock signal")
        clk = signal.value
        continue
      elif name == "rst":
        if not isinstance(signal, BitsSignal):
          raise PortError("'rst' must be a Bits(1)")
        rst = signal.value
        continue

      if name not in port_input_lookup:
        raise PortError(f"Input port {name} not found in module")
      port = port_input_lookup[name]
      if isinstance(signal, ChannelSignal):
        # If the input is a channel signal, the types must match.
        if signal.type.inner_type != port.type:
          raise ValueError(
              f"Wrong type on input signal '{name}'. Got '{signal.type}',"
              f" expected '{port.type}'")
        assert port.idx is not None
        circt_inputs[port.idx] = signal.value
        remaining_inputs.remove(name)
      elif isinstance(signal, Signal):
        raise PortError(f"Input {name} must be a channel signal")
      else:
        raise PortError(f"Port {name} must be a signal")
    if clk is None:
      raise PortError("Missing 'clk' signal")
    if rst is None:
      raise PortError("Missing 'rst' signal")
    if len(remaining_inputs) > 0:
      raise PortError(
          f"Missing input signals for ports: {', '.join(remaining_inputs)}")

    circt_mod = self.circt_mod
    assert circt_mod is not None
    result_types = [Channel(port.type)._type for port in self.outputs]
    inst = raw_handshake.ESIInstanceOp(
        result_types,
        ir.StringAttr(circt_mod.attributes["sym_name"]).value,
        instance_name,
        clk=clk,
        rst=rst,
        opOperands=circt_inputs,
        loc=get_user_loc())
    inst.operation.verify()
    return inst


class Func(Module):
  """A pure ESI module has no ports and contains only instances of modules with
  only ESI ports and connections between said instances. Use ESI services for
  external communication."""

  BuilderType: type[ModuleLikeBuilderBase] = FuncBuilder
  _builder: FuncBuilder
