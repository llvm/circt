# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, types, Module
from pycde.common import Clock, Input
from pycde.constructs import NamedWire, Reg, Wire
from pycde.testing import unittestmodule
from pycde.types import Channel, UInt


@unittestmodule()
class WireTypeTest(Module):
  In = Input(types.i8)

  @generator
  def create(ports):
    w = Wire(types.i4)
    # CHECK: TypeError: Cannot assign i8 to i4
    w.assign(ports.In)


# -----


@unittestmodule()
class WireDoubleAssignTest(Module):
  In = Input(types.i8)

  @generator
  def create(ports):
    w = Wire(types.i8)
    w.assign(ports.In)
    # CHECK: ValueError: Cannot assign value to Wire twice.
    w.assign(ports.In)


# -----


@unittestmodule()
class RegTypeTest(Module):
  clk = Clock()
  In = Input(types.i8)

  @generator
  def create(ports):
    r = Reg(types.i4)
    # CHECK: TypeError: Cannot assign i8 to i4
    r.assign(ports.In)


# -----


@unittestmodule()
class RegDoubleAssignTest(Module):
  Clk = Clock()
  In = Input(types.i8)

  @generator
  def create(ports):
    r = Reg(types.i8)
    r.assign(ports.In)
    # CHECK: ValueError: Cannot assign value to Reg twice.
    r.assign(ports.In)


# -----


@unittestmodule()
class NamedWireHWError(Module):

  @generator
  def create(ports):
    # CHECK: TypeError: NamedWire must have a hardware type, not Channel<UInt<32>, ValidReady>
    NamedWire(Channel(UInt(32)), "asdf")
