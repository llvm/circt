# RUN: %PYTHON% %s | FileCheck %s

from pycde import Module, Clock, Reset, Input, Output
from pycde.seq import FIFO
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
