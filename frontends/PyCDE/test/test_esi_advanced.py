# RUN: %PYTHON% %s | FileCheck %s

from pycde import generator, Clock, Module, Reset
from pycde.common import InputChannel, OutputChannel, Output
from pycde.testing import unittestmodule
from pycde.types import Bits

# CHECK-LABEL:  hw.module @Merge(in %clk : !seq.clock, in %rst : i1, in %a : !esi.channel<i8>, in %b : !esi.channel<i8>, out x : !esi.channel<i8>, out sel : i1)
# CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %a, [[R1:%.+]] : i8
# CHECK-NEXT:     %rawOutput_0, %valid_1 = esi.unwrap.vr %b, [[R2:%.+]] : i8
# CHECK-NEXT:     %true = hw.constant true
# CHECK-NEXT:     [[R0:%.+]] = comb.xor bin %valid, %true : i1
# CHECK-NEXT:     [[R1]] = comb.and bin %valid, %ready : i1
# CHECK-NEXT:     [[R2]] = comb.and bin [[R0]], %ready : i1
# CHECK-NEXT:     [[R3:%.+]] = comb.and bin %valid, %valid : i1
# CHECK-NEXT:     [[R4:%.+]] = comb.and bin [[R0]], %valid_1 : i1
# CHECK-NEXT:     [[R5:%.+]] = comb.or bin [[R3]], [[R4]] : i1
# CHECK-NEXT:     [[R6:%.+]] = comb.mux bin %valid, %rawOutput, %rawOutput_0
# CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr [[R6]], [[R5]] : i8
# CHECK-NEXT:     %true_2 = hw.constant true
# CHECK-NEXT:     [[R7:%.+]] = comb.xor bin %valid, %true_2 : i1
# CHECK-NEXT:     hw.output %chanOutput, [[R7]] : !esi.channel<i8>, i1


@unittestmodule()
class Merge(Module):
  clk = Clock()
  rst = Reset()
  a = InputChannel(Bits(8))
  b = InputChannel(Bits(8))

  x = OutputChannel(Bits(8))
  sel = Output(Bits(1))

  @generator
  def build(ports):
    chan, sel = ports.a.type.merge(ports.a, ports.b)
    ports.x = chan
    ports.sel = sel
