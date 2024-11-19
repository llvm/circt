# RUN: %PYTHON% %s | FileCheck %s

from pycde import (Clock, Output, Input, generator, types, Module)
from pycde.handshake import Func
from pycde.testing import unittestmodule
from pycde.types import Bits, Channel

# CHECK:  hw.module @Top(in %clk : !seq.clock, in %rst : i1, in %a : !esi.channel<i8>, out x : !esi.channel<i8>) attributes {output_file = #hw.output_file<"Top.sv", includeReplicatedOps>} {
# CHECK:    %0 = handshake.esi_instance @TestFunc "TestFunc" clk %clk rst %rst(%a) : (!esi.channel<i8>) -> !esi.channel<i8>
# CHECK:    hw.output %0 : !esi.channel<i8>
# CHECK:  }
# CHECK:  handshake.func @TestFunc(%arg0: i8, ...) -> i8 attributes {argNames = ["a"], output_file = #hw.output_file<"TestFunc.sv", includeReplicatedOps>, resNames = ["x"]} {
# CHECK:    %c15_i8 = hw.constant 15 : i8
# CHECK:    %0 = comb.and bin %arg0, %c15_i8 : i8
# CHECK:    return %0 : i8
# CHECK:  }


class TestFunc(Func):
  a = Input(Bits(8))
  x = Output(Bits(8))

  @generator
  def build(ports):
    ports.x = ports.a & Bits(8)(0xF)


BarType = types.struct({"foo": types.i12}, "bar")


@unittestmodule()
class Top(Module):
  clk = Clock()
  rst = Input(Bits(1))

  a = Input(Channel(Bits(8)))
  x = Output(Channel(Bits(8)))

  @generator
  def build(ports):
    test = TestFunc(clk=ports.clk, rst=ports.rst, a=ports.a)
    ports.x = test.x
