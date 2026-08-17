# RUN: %PYTHON% %s | FileCheck %s

from pycde import Module, Clock, Reset, Input, Output
from pycde.seq import Counter, DownCounter, FIFO
from pycde.testing import unittestmodule
from pycde.types import Bits, UInt

from pycde.module import generator

# CHECK-LABEL: hw.module @SimpleFIFOTest(in %clk : !seq.clock, in %rst : i1)
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      [[R0:%.+]] = hwarith.constant 0 : ui32
# CHECK-NEXT:      %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 16 almost_full 16 almost_empty 0 in [[R0]] rdEn %false wrEn %false clk %clk rst %rst : ui32


@unittestmodule(run_passes=False)
class SimpleFIFOTest(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    c0 = Bits(1)(0)
    ui32 = UInt(32)(0)

    fifo = FIFO(type=UInt(32), depth=16, clk=ports.clk, rst=ports.rst)
    fifo.push(ui32, c0)
    fifo.pop(c0)


# CHECK-LABEL: hw.module @SimpleFIFOTestRd1(in %clk : !seq.clock, in %rst : i1)
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      [[R0:%.+]] = hwarith.constant 0 : ui32
# CHECK-NEXT:      %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 16 rd_latency 1 almost_full 16 almost_empty 0 in [[R0]] rdEn %false wrEn %false clk %clk rst %rst : ui32


@unittestmodule(run_passes=True)
class SimpleFIFOTestRd1(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    c0 = Bits(1)(0)
    ui32 = UInt(32)(0)

    fifo = FIFO(type=UInt(32),
                depth=16,
                clk=ports.clk,
                rst=ports.rst,
                rd_latency=1)
    fifo.push(ui32, c0)
    fifo.pop(c0)


# Both counters only update when something happens (clock enable), and both
# reset to zero.
# CHECK-LABEL:  hw.module @CounterTest
# CHECK:          hw.instance "Counter" sym @Counter @Counter_increment_on_clearFalse_width8
# CHECK:          hw.instance "Counter_1" sym @Counter_1 @Counter_increment_on_clearTrue_width8

# Default: `clear` wins over a coincident `increment`, i.e. the register is fed
# `clear ? 0 : count + 1`.
# CHECK-LABEL:  hw.module @Counter_increment_on_clearFalse_width8(in %clk : !seq.clock, in %rst : i1, in %clear : i1, in %increment : i1, out out : ui8)
# CHECK:          [[CE:%.+]] = comb.or bin %increment, %clear
# CHECK:          %count__reg1 = seq.compreg.ce sym @count__reg1 [[NEXT:%.+]], %clk, [[CE]] reset %rst
# CHECK:          [[ZERO:%.+]] = hwarith.constant 0 : ui8
# CHECK:          [[ONE:%.+]] = hwarith.constant 1 : ui1
# CHECK:          [[SUM:%.+]] = hwarith.add %count__reg1, [[ONE]]
# CHECK:          [[SUM8:%.+]] = hwarith.cast [[SUM]] : (ui9) -> ui8
# CHECK:          [[NEXT]] = comb.mux bin %clear, [[ZERO]], [[SUM8]]

# increment_on_clear: the increment is applied to the cleared value, i.e. the
# register is fed `increment ? base + 1 : base` where `base = clear ? 0 : count`.
# CHECK-LABEL:  hw.module @Counter_increment_on_clearTrue_width8(in %clk : !seq.clock, in %rst : i1, in %clear : i1, in %increment : i1, out out : ui8)
# CHECK:          [[CE:%.+]] = comb.or bin %increment, %clear
# CHECK:          %count__reg1 = seq.compreg.ce sym @count__reg1 [[NEXT:%.+]], %clk, [[CE]] reset %rst
# CHECK:          [[ZERO:%.+]] = hwarith.constant 0 : ui8
# CHECK:          [[BASE:%.+]] = comb.mux bin %clear, [[ZERO]], %count__reg1 {{.*}}"base"
# CHECK:          [[ONE:%.+]] = hwarith.constant 1 : ui1
# CHECK:          [[SUM:%.+]] = hwarith.add [[BASE]], [[ONE]]
# CHECK:          [[SUM8:%.+]] = hwarith.cast [[SUM]] : (ui9) -> ui8
# CHECK:          [[NEXT]] = comb.mux bin %increment, [[SUM8]], [[BASE]]


@unittestmodule(run_passes=False)
class CounterTest(Module):
  clk = Clock()
  rst = Reset()
  clear = Input(Bits(1))
  increment = Input(Bits(1))
  out = Output(UInt(8))
  out_ioc = Output(UInt(8))

  @generator
  def construct(ports):
    counter = Counter(8)(clk=ports.clk,
                         rst=ports.rst,
                         clear=ports.clear,
                         increment=ports.increment)
    ioc_counter = Counter(8, increment_on_clear=True)(clk=ports.clk,
                                                      rst=ports.rst,
                                                      clear=ports.clear,
                                                      increment=ports.increment)
    ports.out = counter.out
    ports.out_ioc = ioc_counter.out


# CHECK-LABEL:  hw.module @DownCounterTest
# CHECK:          hw.instance "DownCounter" sym @DownCounter @DownCounter_decrement_on_loadFalse_width8
# CHECK:          hw.instance "DownCounter_1" sym @DownCounter_1 @DownCounter_decrement_on_loadTrue_width8

# Default: `load` wins over a coincident `decrement`, and the decrement
# saturates at zero rather than wrapping. `is_zero` is the (registered) count
# compared against zero, so it costs no extra latency.
# CHECK-LABEL:  hw.module @DownCounter_decrement_on_loadFalse_width8(in %clk : !seq.clock, in %rst : i1, in %load : i1, in %load_value : ui8, in %decrement : i1, out out : ui8, out is_zero : i1)
# CHECK:          [[CE:%.+]] = comb.or bin %load, %decrement
# CHECK:          %count__reg1 = seq.compreg.ce sym @count__reg1 [[NEXT:%.+]], %clk, [[CE]] reset %rst
# CHECK:          [[ZERO:%.+]] = hwarith.constant 0 : ui8
# CHECK:          [[ONE:%.+]] = hwarith.constant 1 : ui8
# CHECK:          [[ISZERO:%.+]] = hwarith.icmp eq %count__reg1, [[ZERO]] {{.*}}"count_is_zero"
# CHECK:          [[DIFF:%.+]] = hwarith.sub %count__reg1, [[ONE]]
# CHECK:          [[DIFF8:%.+]] = hwarith.cast [[DIFF]] : (si9) -> ui8
# CHECK:          [[SAT:%.+]] = comb.mux bin [[ISZERO]], %count__reg1, [[DIFF8]]
# CHECK:          [[DEC:%.+]] = comb.mux bin %decrement, [[SAT]], %count__reg1
# CHECK:          [[NEXT]] = comb.mux bin %load, %load_value, [[DEC]]
# CHECK:          hw.output %count__reg1, [[ISZERO]]

# decrement_on_load: the decrement is applied to the newly loaded value, i.e.
# the saturating decrement operates on `base = load ? load_value : count`.
# CHECK-LABEL:  hw.module @DownCounter_decrement_on_loadTrue_width8(in %clk : !seq.clock, in %rst : i1, in %load : i1, in %load_value : ui8, in %decrement : i1, out out : ui8, out is_zero : i1)
# CHECK:          [[CE:%.+]] = comb.or bin %load, %decrement
# CHECK:          %count__reg1 = seq.compreg.ce sym @count__reg1 [[NEXT:%.+]], %clk, [[CE]] reset %rst
# CHECK:          [[ZERO:%.+]] = hwarith.constant 0 : ui8
# CHECK:          [[ONE:%.+]] = hwarith.constant 1 : ui8
# CHECK:          [[ISZERO:%.+]] = hwarith.icmp eq %count__reg1, [[ZERO]] {{.*}}"count_is_zero"
# CHECK:          [[BASE:%.+]] = comb.mux bin %load, %load_value, %count__reg1 {{.*}}"base"
# CHECK:          [[LDZERO:%.+]] = hwarith.icmp eq %load_value, [[ZERO]]
# CHECK:          [[BASEZERO:%.+]] = comb.mux bin %load, [[LDZERO]], [[ISZERO]] {{.*}}"base_is_zero"
# CHECK:          [[DIFF:%.+]] = hwarith.sub [[BASE]], [[ONE]]
# CHECK:          [[DIFF8:%.+]] = hwarith.cast [[DIFF]] : (si9) -> ui8
# CHECK:          [[SAT:%.+]] = comb.mux bin [[BASEZERO]], [[BASE]], [[DIFF8]]
# CHECK:          [[NEXT]] = comb.mux bin %decrement, [[SAT]], [[BASE]]
# CHECK:          hw.output %count__reg1, [[ISZERO]]


@unittestmodule(run_passes=False)
class DownCounterTest(Module):
  clk = Clock()
  rst = Reset()
  load = Input(Bits(1))
  load_value = Input(UInt(8))
  decrement = Input(Bits(1))
  out = Output(UInt(8))
  is_zero = Output(Bits(1))
  out_dol = Output(UInt(8))

  @generator
  def construct(ports):
    counter = DownCounter(8)(clk=ports.clk,
                             rst=ports.rst,
                             load=ports.load,
                             load_value=ports.load_value,
                             decrement=ports.decrement)
    dol_counter = DownCounter(8, decrement_on_load=True)(
        clk=ports.clk,
        rst=ports.rst,
        load=ports.load,
        load_value=ports.load_value,
        decrement=ports.decrement)
    ports.out = counter.out
    ports.is_zero = counter.is_zero
    ports.out_dol = dol_counter.out
