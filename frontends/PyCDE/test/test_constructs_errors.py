# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, Module
from pycde.common import Clock, Input
from pycde.constructs import NamedWire, Reg, Wire
from pycde.testing import unittestmodule
from pycde.types import Bit, Bits, Channel, UInt

I1 = Bit
I2 = Bits(2)
I4 = Bits(4)
I8 = Bits(8)


@unittestmodule()
class WireTypeTest(Module):
  In = Input(I8)

  @generator
  def create(ports):
    w = Wire(I4)
    # CHECK: TypeError: Cannot assign i8 to i4
    w.assign(ports.In)


# -----


@unittestmodule()
class WireDoubleAssignTest(Module):
  In = Input(I8)

  @generator
  def create(ports):
    w = Wire(I8)
    w.assign(ports.In)
    # CHECK: ValueError: Cannot assign value to Wire twice.
    w.assign(ports.In)


# -----


@unittestmodule()
class RegTypeTest(Module):
  clk = Clock()
  In = Input(I8)

  @generator
  def create(ports):
    r = Reg(I4)
    # CHECK: TypeError: Cannot assign i8 to i4
    r.assign(ports.In)


# -----


@unittestmodule()
class RegDoubleAssignTest(Module):
  Clk = Clock()
  In = Input(I8)

  @generator
  def create(ports):
    r = Reg(I8)
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
