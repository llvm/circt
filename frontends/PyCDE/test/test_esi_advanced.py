# RUN: %PYTHON% %s | FileCheck %s

from pycde import generator, Clock, Module, Reset
from pycde.common import InputChannel, OutputChannel
from pycde.testing import unittestmodule
from pycde.types import Bits, UInt

# CHECK-LABEL:    hw.module @Merge(in %clk : !seq.clock, in %rst : i1, in %a : !esi.channel<i8>, in %b : !esi.channel<i8>, out x : !esi.channel<i8>)
# CHECK-NEXT:       %rawOutput, %valid = esi.unwrap.vr %a, [[R1:%.+]] : i8
# CHECK-NEXT:       %rawOutput_0, %valid_1 = esi.unwrap.vr %b, [[R2:%.+]] : i8
# CHECK-NEXT:       %true = hw.constant true
# CHECK-NEXT:       [[R0:%.+]] = comb.xor bin %valid, %true : i1
# CHECK-NEXT:       [[R1]] = comb.and bin %valid, %ready : i1
# CHECK-NEXT:       [[R2]] = comb.and bin [[R0]], %ready : i1
# CHECK-NEXT:       [[R3:%.+]] = comb.and bin %valid, %valid : i1
# CHECK-NEXT:       [[R4:%.+]] = comb.and bin [[R0]], %valid_1 : i1
# CHECK-NEXT:       [[R5:%.+]] = comb.or bin [[R3]], [[R4]] : i1
# CHECK-NEXT:       [[R6:%.+]] = comb.mux bin %valid, %rawOutput, %rawOutput_0
# CHECK-NEXT:       %chanOutput, %ready = esi.wrap.vr [[R6]], [[R5]] : i8
# CHECK-NEXT:       hw.output %chanOutput : !esi.channel<i8>


@unittestmodule()
class Merge(Module):
  clk = Clock()
  rst = Reset()
  a = InputChannel(Bits(8))
  b = InputChannel(Bits(8))

  x = OutputChannel(Bits(8))

  @generator
  def build(ports):
    chan = ports.a.type.merge(ports.a, ports.b)
    ports.x = chan


# CHECK-LABEL:    hw.module @Join(in %clk : !seq.clock, in %rst : i1, in %a : !esi.channel<ui8>, in %b : !esi.channel<ui8>, out x : !esi.channel<ui9>)
# CHECK-NEXT:       %rawOutput, %valid = esi.unwrap.vr %a, [[R2:%.+]] : ui8
# CHECK-NEXT:       %rawOutput_0, %valid_1 = esi.unwrap.vr %b, [[R2]] : ui8
# CHECK-NEXT:       [[R0:%.+]] = comb.and bin %valid, %valid_1 : i1
# CHECK-NEXT:       [[R1:%.+]] = hw.struct_create (%rawOutput, %rawOutput_0) : !hw.struct<a: ui8, b: ui8>
# CHECK-NEXT:       %chanOutput, %ready = esi.wrap.vr [[R1]], [[R0]] : !hw.struct<a: ui8, b: ui8>
# CHECK-NEXT:       [[R2]] = comb.and bin %ready, [[R0]] : i1
# CHECK-NEXT:       %rawOutput_2, %valid_3 = esi.unwrap.vr %chanOutput, %ready_7 : !hw.struct<a: ui8, b: ui8>
# CHECK-NEXT:       %a_4 = hw.struct_extract %rawOutput_2["a"] : !hw.struct<a: ui8, b: ui8>
# CHECK-NEXT:       %b_5 = hw.struct_extract %rawOutput_2["b"] : !hw.struct<a: ui8, b: ui8>
# CHECK-NEXT:       [[R3:%.+]] = hwarith.add %a_4, %b_5 : (ui8, ui8) -> ui9
# CHECK-NEXT:       %chanOutput_6, %ready_7 = esi.wrap.vr [[R3]], %valid_3 : ui9
# CHECK-NEXT:       hw.output %chanOutput_6 : !esi.channel<ui9>
@unittestmodule(run_passes=True, emit_outputs=True)
class Join(Module):
  clk = Clock()
  rst = Reset()
  a = InputChannel(UInt(8))
  b = InputChannel(UInt(8))

  x = OutputChannel(UInt(9))

  @generator
  def build(ports):
    joined = ports.a.type.join(ports.a, ports.b)
    ports.x = joined.transform(lambda x: x.a + x.b)


# CHECK-LABEL:    hw.module @Fork(in %clk : !seq.clock, in %rst : i1, in %a : !esi.channel<ui8>, out x : !esi.channel<ui8>, out y : !esi.channel<ui8>)
# CHECK-NEXT:       %rawOutput, %valid = esi.unwrap.vr %a, [[R3:%.+]] : ui8
# CHECK-NEXT:       [[R0:%.+]] = comb.and bin [[R3]], %valid : i1
# CHECK-NEXT:       %chanOutput, %ready = esi.wrap.vr %rawOutput, [[R0]] : ui8
# CHECK-NEXT:       %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, [[R0]] : ui8
# CHECK-NEXT:       [[R1:%.+]] = esi.buffer %clk, %rst, %chanOutput {stages = 1 : i64} : !esi.channel<ui8> -> !esi.channel<ui8>
# CHECK-NEXT:       [[R2:%.+]] = esi.buffer %clk, %rst, %chanOutput_0 {stages = 1 : i64} : !esi.channel<ui8> -> !esi.channel<ui8>
# CHECK-NEXT:       [[R3]] = comb.and bin %ready, %ready_1 : i1
# CHECK-NEXT:       hw.output [[R1]], [[R2]] : !esi.channel<ui8>, !esi.channel<ui8>
@unittestmodule(run_passes=True, emit_outputs=True)
class Fork(Module):
  clk = Clock()
  rst = Reset()
  a = InputChannel(UInt(8))

  x = OutputChannel(UInt(8))
  y = OutputChannel(UInt(8))

  @generator
  def build(ports):
    x, y = ports.a.fork(ports.clk, ports.rst)
    ports.x = x
    ports.y = y
