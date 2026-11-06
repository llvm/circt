# REQUIRES: iverilog-12,cocotb
# RUN: %PYTHON% %s 2>&1 | FileCheck %s

# `esi.TaggedDemux` must consume its input exactly when the output selected by
# the tag accepts it, regardless of the other outputs' readiness.

from pycde import Clock, Input, Module, Output, Reset, generator
from pycde.esi import TaggedDemux
from pycde.testing import cocotest, cocotestbench
from pycde.types import Bits, Channel, UInt

DemuxChannel = Channel(UInt(8))


class TaggedDemuxTop(Module):
  clk = Clock()
  rst = Reset()

  in_data = Input(UInt(8))
  in_valid = Input(Bits(1))
  in_ready = Output(Bits(1))

  out0_data = Output(UInt(8))
  out0_valid = Output(Bits(1))
  out0_ready = Input(Bits(1))

  out1_data = Output(UInt(8))
  out1_valid = Output(Bits(1))
  out1_ready = Input(Bits(1))

  @generator
  def build(ports):
    in_ch, in_ready = DemuxChannel.wrap(ports.in_data, ports.in_valid)
    ports.in_ready = in_ready
    demux = TaggedDemux(2, DemuxChannel)(clk=ports.clk,
                                         rst=ports.rst,
                                         in_=in_ch)
    ports.out0_data, ports.out0_valid = demux.out0.unwrap(ports.out0_ready)
    ports.out1_data, ports.out1_valid = demux.out1.unwrap(ports.out1_ready)


# CHECK:      ** TEST
# CHECK:      ** test_TaggedDemuxTop.ready_follows_selected_output
# CHECK:      ** TESTS=1 PASS=1 FAIL=0 SKIP=0


@cocotestbench(TaggedDemuxTop, simulator="icarus")
class TaggedDemuxTester:

  @cocotest
  async def ready_follows_selected_output(ports):
    from cocotb.triggers import Timer

    async def check(tag, out0_ready, out1_ready, in_ready, out0_valid,
                    out1_valid):
      ports.in_data.value = tag
      ports.in_valid.value = 1
      ports.out0_ready.value = out0_ready
      ports.out1_ready.value = out1_ready
      await Timer(1, "ns")
      state = (f"tag={tag} out0_ready={out0_ready} out1_ready={out1_ready}")
      assert ports.in_ready.value == in_ready, state
      assert ports.out0_valid.value == out0_valid, state
      assert ports.out1_valid.value == out1_valid, state

    # The selected output being ready is enough, even if the other isn't.
    await check(0, 1, 0, in_ready=1, out0_valid=1, out1_valid=0)
    await check(1, 0, 1, in_ready=1, out0_valid=0, out1_valid=1)
    # The selected output not being ready backpressures the input, even if
    # the other is ready.
    await check(0, 0, 1, in_ready=0, out0_valid=1, out1_valid=0)
    await check(1, 1, 0, in_ready=0, out0_valid=0, out1_valid=1)
    # Messages whose tag selects no output are dropped.
    await check(5, 0, 0, in_ready=1, out0_valid=0, out1_valid=0)
