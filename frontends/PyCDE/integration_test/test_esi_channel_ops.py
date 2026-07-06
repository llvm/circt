# REQUIRES: iverilog,cocotb
# RUN: %PYTHON% %s 2>&1 | FileCheck %s

# Integration tests for ChannelSignal methods: buffer, snoop, snoop_xact,
# transform, fork, and wait_for_ready.
#
# Each PyCDE module exercises one (or a combination) of the highlighted
# ChannelSignal methods.  The cocotb testbench pushes data through the
# generated Verilog and checks functional correctness under various
# validity/backpressure cadences.

import os
import sys

from pycde import (Clock, Input, InputChannel, Output, OutputChannel, Module,
                   Reset, generator)
from pycde.common import AppID
from pycde.constructs import Wire
from pycde.types import Bit, Bits, Channel, ChannelSignaling, UInt
from pycde.testing import cocotestbench, cocotest, cocoextra

# ---------------------------------------------------------------------------
# DUT modules
# ---------------------------------------------------------------------------

MASK8 = (1 << 8) - 1
MASK16 = (1 << 16) - 1

ESI_PRIMS = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                         "lib", "Dialect", "ESI", "ESIPrimitives.sv")


class BufferPassthrough(Module):
  """Channel in -> 2-stage buffer -> Channel out."""
  clk = Clock()
  rst = Input(Bits(1))
  inp = InputChannel(Bits(16))
  out = OutputChannel(Bits(16))

  @generator
  def build(ports):
    ports.out = ports.inp.buffer(ports.clk, ports.rst, 2)


class SnoopPassthrough(Module):
  """Channel in -> unwrap -> rewrap -> channel out.

  Also exposes snoop outputs (valid, ready, data) as module outputs so the
  testbench can observe them.
  """
  clk = Clock()
  rst = Input(Bits(1))
  inp = InputChannel(Bits(16))
  out = OutputChannel(Bits(16))
  snp_valid = Output(Bits(1))
  snp_ready = Output(Bits(1))
  snp_data = Output(Bits(16))

  @generator
  def build(ports):
    valid, ready, data = ports.inp.snoop()
    ports.snp_valid = valid
    ports.snp_ready = ready
    ports.snp_data = data
    # Pass the channel through a buffer so there is an actual consumer (and
    # therefore a ready signal for the snoop to observe).
    ports.out = ports.inp.buffer(ports.clk, ports.rst, 1)


class SnoopXactPassthrough(Module):
  """Like SnoopPassthrough but uses snoop_xact (transaction indicator)."""
  clk = Clock()
  rst = Input(Bits(1))
  inp = InputChannel(Bits(16))
  out = OutputChannel(Bits(16))
  snp_xact = Output(Bits(1))
  snp_data = Output(Bits(16))

  @generator
  def build(ports):
    xact, data = ports.inp.snoop_xact()
    ports.snp_xact = xact
    ports.snp_data = data
    ports.out = ports.inp.buffer(ports.clk, ports.rst, 1)


class TransformLowByte(Module):
  """Use ``transform`` to extract the low byte from a 16-bit channel."""
  clk = Clock()
  rst = Input(Bits(1))
  inp = InputChannel(Bits(16))
  out = OutputChannel(Bits(8))

  @generator
  def build(ports):
    ports.out = ports.inp.transform(lambda data: data[0:8])


class ForkModule(Module):
  """Use ``fork`` to duplicate a channel into two outputs."""
  clk = Clock()
  rst = Input(Bits(1))
  inp = InputChannel(Bits(16))
  out_a = OutputChannel(Bits(16))
  out_b = OutputChannel(Bits(16))

  @generator
  def build(ports):
    a, b = ports.inp.fork(ports.clk, ports.rst)
    ports.out_a = a
    ports.out_b = b


class WaitForReadyModule(Module):
  """Use ``wait_for_ready`` to gate a channel on another channel's readiness.

  ``inp_data`` only produces valid when ``inp_gate`` is ready.
  """
  clk = Clock()
  rst = Input(Bits(1))
  inp_data = InputChannel(Bits(16))
  inp_gate = InputChannel(Bits(16))
  out_data = OutputChannel(Bits(16))
  out_gate = OutputChannel(Bits(16))

  @generator
  def build(ports):
    ports.out_data = ports.inp_data.wait_for_ready(ports.inp_gate)
    # The gate channel still needs a consumer — pass it through a buffer.
    ports.out_gate = ports.inp_gate.buffer(ports.clk, ports.rst, 1)


# ===================================================================
# Tests
# ===================================================================

# CHECK:      ** TEST
# CHECK:      ** TESTS={{[0-9]+}} PASS={{[0-9]+}} FAIL=0 SKIP=0


@cocotestbench(BufferPassthrough, simulator="icarus")
class TestBuffer:

  @cocoextra
  def extra():
    return [ESI_PRIMS]

  @cocotest
  async def test_buffer_always(dut):
    """Buffer with continuous valid + ready."""
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset,
                                   run_passthrough_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut, "inp", ChannelSignaling.ValidReady)
    mon = ESIOutputMonitor.create(dut, "out", ChannelSignaling.ValidReady)

    values = list(range(32))
    await run_passthrough_test(dut, drv, mon, values)

  @cocotest
  async def test_buffer_slow_producer(dut):
    """Buffer with a producer that asserts valid every 3rd cycle."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset,
                                   run_passthrough_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut,
                                "inp",
                                ChannelSignaling.ValidReady,
                                valid_cadence=Cadence.fixed(3))
    mon = ESIOutputMonitor.create(dut, "out", ChannelSignaling.ValidReady)

    values = list(range(16))
    await run_passthrough_test(dut, drv, mon, values)

  @cocotest
  async def test_buffer_slow_consumer(dut):
    """Buffer with a consumer that asserts ready every 4th cycle."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset,
                                   run_passthrough_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut, "inp", ChannelSignaling.ValidReady)
    mon = ESIOutputMonitor.create(dut,
                                  "out",
                                  ChannelSignaling.ValidReady,
                                  ready_cadence=Cadence.fixed(4))

    values = list(range(16))
    await run_passthrough_test(dut, drv, mon, values)

  @cocotest
  async def test_buffer_random(dut):
    """Buffer with random valid (p=0.6) and random ready (p=0.5)."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset,
                                   run_passthrough_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut,
                                "inp",
                                ChannelSignaling.ValidReady,
                                valid_cadence=Cadence.random(0.6))
    mon = ESIOutputMonitor.create(dut,
                                  "out",
                                  ChannelSignaling.ValidReady,
                                  ready_cadence=Cadence.random(0.5))

    values = list(range(32))
    await run_passthrough_test(dut, drv, mon, values)


# CHECK:      ** TEST
# CHECK:      ** TESTS={{[0-9]+}} PASS={{[0-9]+}} FAIL=0 SKIP=0


@cocotestbench(SnoopPassthrough, simulator="icarus")
class TestSnoop:

  @cocoextra
  def extra():
    return [ESI_PRIMS]

  @cocotest
  async def test_snoop_observes_transactions(dut):
    """Verify that snoop outputs reflect valid, ready, and data."""
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset, timeout_watchdog)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)
    cocotb.start_soon(timeout_watchdog(dut, 5000))

    drv = ESIInputDriver.create(dut, "inp", ChannelSignaling.ValidReady)
    mon = ESIOutputMonitor.create(dut, "out", ChannelSignaling.ValidReady)
    mon.start()

    values = list(range(8))
    snooped_data = []

    async def snoop_watcher():
      """Capture data when snoop shows a transaction (valid & ready)."""
      while len(snooped_data) < len(values):
        await RisingEdge(dut.clk)
        try:
          v = int(dut.snp_valid.value)
          r = int(dut.snp_ready.value)
          d = int(dut.snp_data.value)
        except ValueError:
          continue
        if v == 1 and r == 1:
          snooped_data.append(d)

    cocotb.start_soon(snoop_watcher())
    await drv.send_all(values)
    results = await mon.recv_n(len(values))

    # Allow a few extra cycles for the snoop watcher to catch up.
    for _ in range(10):
      await RisingEdge(dut.clk)

    assert results == values, f"Output mismatch: {results} != {values}"
    assert snooped_data == values, \
        f"Snoop mismatch: {snooped_data} != {values}"
    mon.stop()


# CHECK:      ** TEST
# CHECK:      ** TESTS={{[0-9]+}} PASS={{[0-9]+}} FAIL=0 SKIP=0


@cocotestbench(SnoopXactPassthrough, simulator="icarus")
class TestSnoopXact:

  @cocoextra
  def extra():
    return [ESI_PRIMS]

  @cocotest
  async def test_snoop_xact_counts_transactions(dut):
    """Verify snoop_xact fires on each completed transaction."""
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset, timeout_watchdog)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)
    cocotb.start_soon(timeout_watchdog(dut, 5000))

    drv = ESIInputDriver.create(dut, "inp", ChannelSignaling.ValidReady)
    mon = ESIOutputMonitor.create(dut, "out", ChannelSignaling.ValidReady)
    mon.start()

    values = list(range(10))
    xact_data = []

    async def xact_watcher():
      while len(xact_data) < len(values):
        await RisingEdge(dut.clk)
        try:
          x = int(dut.snp_xact.value)
          d = int(dut.snp_data.value)
        except ValueError:
          continue
        if x == 1:
          xact_data.append(d)

    cocotb.start_soon(xact_watcher())
    await drv.send_all(values)
    results = await mon.recv_n(len(values))

    for _ in range(10):
      await RisingEdge(dut.clk)

    assert results == values
    assert xact_data == values, \
        f"Xact snoop mismatch: {xact_data} != {values}"
    mon.stop()


# CHECK:      ** TEST
# CHECK:      ** TESTS={{[0-9]+}} PASS={{[0-9]+}} FAIL=0 SKIP=0


@cocotestbench(TransformLowByte, simulator="icarus")
class TestTransform:

  @cocotest
  async def test_transform_always(dut):
    """Transform extracts low byte, continuous flow."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset,
                                   run_passthrough_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut, "inp", ChannelSignaling.ValidReady)
    mon = ESIOutputMonitor.create(dut, "out", ChannelSignaling.ValidReady)

    values = [0x1234, 0xABCD, 0xFF00, 0x00FF, 0xDEAD]
    await run_passthrough_test(dut,
                               drv,
                               mon,
                               values,
                               transform=lambda v: v & 0xFF)

  @cocotest
  async def test_transform_random_backpressure(dut):
    """Transform with random backpressure."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset,
                                   run_passthrough_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut,
                                "inp",
                                ChannelSignaling.ValidReady,
                                valid_cadence=Cadence.random(0.7))
    mon = ESIOutputMonitor.create(dut,
                                  "out",
                                  ChannelSignaling.ValidReady,
                                  ready_cadence=Cadence.random(0.4))

    values = [i * 0x0101 for i in range(20)]
    await run_passthrough_test(dut,
                               drv,
                               mon,
                               values,
                               transform=lambda v: v & 0xFF)


# CHECK:      ** TEST
# CHECK:      ** TESTS={{[0-9]+}} PASS={{[0-9]+}} FAIL=0 SKIP=0


@cocotestbench(ForkModule, simulator="icarus")
class TestFork:

  @cocoextra
  def extra():
    return [ESI_PRIMS]

  @cocotest
  async def test_fork_always(dut):
    """Fork: both outputs receive the same data, continuous flow."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset, run_fork_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut, "inp", ChannelSignaling.ValidReady)
    mon_a = ESIOutputMonitor.create(dut, "out_a", ChannelSignaling.ValidReady)
    mon_b = ESIOutputMonitor.create(dut, "out_b", ChannelSignaling.ValidReady)

    values = list(range(16))
    await run_fork_test(dut, drv, mon_a, mon_b, values)

  @cocotest
  async def test_fork_asymmetric_backpressure(dut):
    """Fork: one consumer is slower than the other."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset, run_fork_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut, "inp", ChannelSignaling.ValidReady)
    mon_a = ESIOutputMonitor.create(dut,
                                    "out_a",
                                    ChannelSignaling.ValidReady,
                                    ready_cadence=Cadence.always())
    mon_b = ESIOutputMonitor.create(dut,
                                    "out_b",
                                    ChannelSignaling.ValidReady,
                                    ready_cadence=Cadence.fixed(5))

    values = list(range(12))
    await run_fork_test(dut, drv, mon_a, mon_b, values)

  @cocotest
  async def test_fork_random(dut):
    """Fork with random validity and backpressure."""
    import cocotb
    from cocotb.clock import Clock
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset, run_fork_test)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    drv = ESIInputDriver.create(dut,
                                "inp",
                                ChannelSignaling.ValidReady,
                                valid_cadence=Cadence.random(0.7))
    mon_a = ESIOutputMonitor.create(dut,
                                    "out_a",
                                    ChannelSignaling.ValidReady,
                                    ready_cadence=Cadence.random(0.6))
    mon_b = ESIOutputMonitor.create(dut,
                                    "out_b",
                                    ChannelSignaling.ValidReady,
                                    ready_cadence=Cadence.random(0.4))

    values = list(range(16))
    await run_fork_test(dut, drv, mon_a, mon_b, values)


# CHECK:      ** TEST
# CHECK:      ** TESTS={{[0-9]+}} PASS={{[0-9]+}} FAIL=0 SKIP=0


@cocotestbench(WaitForReadyModule, simulator="icarus")
class TestWaitForReady:

  @cocoextra
  def extra():
    return [ESI_PRIMS]

  @cocotest
  async def test_wait_for_ready_basic(dut):
    """Data flows only when gate channel is ready."""
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset, timeout_watchdog)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)
    cocotb.start_soon(timeout_watchdog(dut, 10000))

    drv_data = ESIInputDriver.create(dut, "inp_data",
                                     ChannelSignaling.ValidReady)
    drv_gate = ESIInputDriver.create(dut, "inp_gate",
                                     ChannelSignaling.ValidReady)
    mon_data = ESIOutputMonitor.create(dut, "out_data",
                                       ChannelSignaling.ValidReady)
    mon_gate = ESIOutputMonitor.create(dut, "out_gate",
                                       ChannelSignaling.ValidReady)

    mon_data.start()
    mon_gate.start()

    num_values = 8
    data_values = list(range(num_values))
    gate_values = list(range(100, 100 + num_values))

    async def send_data():
      await drv_data.send_all(data_values)

    async def send_gate():
      await drv_gate.send_all(gate_values)

    cocotb.start_soon(send_data())
    cocotb.start_soon(send_gate())

    data_results = await mon_data.recv_n(num_values)
    gate_results = await mon_gate.recv_n(num_values)

    assert data_results == data_values, \
        f"Data mismatch: {data_results} != {data_values}"
    assert gate_results == gate_values, \
        f"Gate mismatch: {gate_results} != {gate_values}"
    mon_data.stop()
    mon_gate.stop()

  @cocotest
  async def test_wait_for_ready_stalled_gate(dut):
    """Data channel stalls when gate's consumer is slow."""
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import RisingEdge
    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor, Cadence,
                                   ChannelSignaling, reset, timeout_watchdog)

    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)
    cocotb.start_soon(timeout_watchdog(dut, 10000))

    drv_data = ESIInputDriver.create(dut, "inp_data",
                                     ChannelSignaling.ValidReady)
    drv_gate = ESIInputDriver.create(dut, "inp_gate",
                                     ChannelSignaling.ValidReady)
    mon_data = ESIOutputMonitor.create(dut, "out_data",
                                       ChannelSignaling.ValidReady)
    # Slow gate consumer — only accepts every 3rd cycle.
    mon_gate = ESIOutputMonitor.create(dut,
                                       "out_gate",
                                       ChannelSignaling.ValidReady,
                                       ready_cadence=Cadence.fixed(3))

    mon_data.start()
    mon_gate.start()

    num_values = 6
    data_values = list(range(num_values))
    gate_values = list(range(200, 200 + num_values))

    async def send_data():
      await drv_data.send_all(data_values)

    async def send_gate():
      await drv_gate.send_all(gate_values)

    cocotb.start_soon(send_data())
    cocotb.start_soon(send_gate())

    data_results = await mon_data.recv_n(num_values)
    gate_results = await mon_gate.recv_n(num_values)

    assert data_results == data_values
    assert gate_results == gate_values
    mon_data.stop()
    mon_gate.stop()
