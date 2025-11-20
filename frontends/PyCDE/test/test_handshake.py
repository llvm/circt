# RUN: %PYTHON% %s | FileCheck %s

from pycde import (Clock, Output, Input, generator, Module)
from pycde.handshake import Func, cmerge, demux
from pycde.testing import unittestmodule
from pycde.types import Bits, Channel, StructType, TypeAlias

# CHECK:  hw.module @Top(in %clk : !seq.clock, in %rst : i1, in %a : !esi.channel<i8>, in %b : !esi.channel<i8>, out x : !esi.channel<i8>)
# CHECK:    %0:2 = handshake.esi_instance @TestFunc "TestFunc" clk %clk rst %rst(%a, %b) : (!esi.channel<i8>, !esi.channel<i8>) -> (!esi.channel<i8>, !esi.channel<i8>)
# CHECK:    hw.output %0#0 : !esi.channel<i8>

# CHECK:  handshake.func @TestFunc(%arg0: i8, %arg1: i8, ...) -> (i8, i8)
# CHECK:    %result, %index = control_merge %arg0, %arg1 : i8, i1
# CHECK:    %c15_i8 = hw.constant 15 : i8
# CHECK:    [[R0:%.+]] = comb.and bin %result, %c15_i8 : i8
# CHECK:    %trueResult, %falseResult = cond_br %index, [[R0]] : i8
# CHECK:    return %trueResult, %falseResult : i8, i8


class TestFunc(Func):
  a = Input(Bits(8))
  b = Input(Bits(8))
  x = Output(Bits(8))
  y = Output(Bits(8))

  @generator
  def build(ports):
    c, sel = cmerge(ports.a, ports.b)
    z = c & Bits(8)(0xF)
    x, y = demux(sel, z)
    ports.x = x
    ports.y = y


BarType = TypeAlias(StructType({"foo": Bits(12)}), "bar")


@unittestmodule(print=True)
class Top(Module):
  clk = Clock()
  rst = Input(Bits(1))

  a = Input(Channel(Bits(8)))
  b = Input(Channel(Bits(8)))
  x = Output(Channel(Bits(8)))

  @generator
  def build(ports):
    test = TestFunc(clk=ports.clk, rst=ports.rst, a=ports.a, b=ports.b)
    ports.x = test.x
