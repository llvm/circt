#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import sys
from typing import List, Optional, Dict, Tuple

from .module import Module, ModuleLikeBuilderBase, PortError
from .signals import (BitsSignal, ChannelSignal, ClockSignal, Signal,
                      _FromCirctValue)
from .system import System
from .support import clog2, get_user_loc
from .types import Bits, Channel

from .circt.dialects import handshake as raw_handshake
from .circt import ir


class FuncBuilder(ModuleLikeBuilderBase):
  """Defines how a handshake function gets built."""

  def create_op(self, sys: System, symbol):
    """Callback for creating a handshake.func op."""

    self.create_op_common(sys, symbol)

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

    print(
        sys.stderr, "WARNING: the Func handshake flow is currently broken "
        "and thus disabled.")
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
              f"Wrong type on input signal '{name}'. Got '{signal.type.inner_type}',"
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
  """A Handshake function is intended to implicitly model dataflow. If can
  contain any combinational operation and offers a software-like (HLS) approach
  to hardware design.

  The PyCDE interface to it (this class) is considered experimental. Use at your
  own risk and test the resulting RTL thoroughly.

  Warning: the DC flow is currently broken and thus disabled. Do not use this.
  https://github.com/llvm/circt/issues/7949
  """

  BuilderType: type[ModuleLikeBuilderBase] = FuncBuilder
  _builder: FuncBuilder


def demux(cond: BitsSignal, data: Signal) -> Tuple[Signal, Signal]:
  """Demux a signal based on a condition."""
  condbr = raw_handshake.ConditionalBranchOp(cond.value, data.value)
  return (_FromCirctValue(condbr.trueResult),
          _FromCirctValue(condbr.falseResult))


def cmerge(*args: Signal) -> Tuple[Signal, BitsSignal]:
  """Merge multiple signals into one and the index of the signal."""
  if len(args) == 0:
    raise ValueError("cmerge must have at least one argument")
  first = args[0]
  for a in args[1:]:
    if a.type != first.type:
      raise ValueError("All arguments to cmerge must have the same type")
  idx_type = Bits(clog2(len(args)))
  cm = raw_handshake.ControlMergeOp(a.type._type, idx_type._type,
                                    [a.value for a in args])
  return (_FromCirctValue(cm.result), BitsSignal(cm.index, idx_type))
