#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Cocotb-based framework for testing ESI channels.

Provides drivers (producers) and monitors (consumers) for ValidReady and FIFO
signaled ESI channels.  Both validity and backpressure can be driven at fixed
cadences (every Nth cycle), burst patterns, or randomly.

Typical usage inside a ``@cocotest`` function::

    from pycde.esi_testing import (ESIInputDriver, ESIOutputMonitor,
                                   Cadence, CadenceMode)

    @cocotest
    async def my_test(dut):
        import cocotb
        from cocotb.clock import Clock
        from cocotb.triggers import RisingEdge

        clock = Clock(dut.clk, 1, units="ns")
        cocotb.start_soon(clock.start())
        await reset(dut)

        from pycde.types import ChannelSignaling
        drv = ESIInputDriver.create(dut, "inp",
                                    signaling=ChannelSignaling.ValidReady)
        mon = ESIOutputMonitor.create(dut, "out",
                                     signaling=ChannelSignaling.ValidReady)
        mon.start()

        for i in range(16):
            await drv.send(i)
        results = []
        for _ in range(16):
            results.append(await mon.recv())
"""

import random
from enum import Enum, auto
from typing import List, Optional

from .types import ChannelSignaling

# These imports are available only when running under cocotb. The module can be
# imported at generation time (to collect ``@cocoextra`` helpers, etc.) so guard
# them.
try:
  import cocotb
  from cocotb.triggers import RisingEdge
except ImportError:
  pass

# ---------------------------------------------------------------------------
# Cadence helpers
# ---------------------------------------------------------------------------


class CadenceMode(Enum):
  """How validity / backpressure is driven."""
  ALWAYS = auto()
  FIXED = auto()
  RANDOM = auto()


class Cadence:
  """Describes a cadence pattern for asserting a 1-bit signal.

  ``ALWAYS``  – the signal is asserted every cycle.
  ``FIXED(n)`` – asserted every *n*-th cycle (duty = 1/n).
  ``RANDOM(p)`` – asserted with probability *p* each cycle.
  """

  def __init__(self, mode: CadenceMode = CadenceMode.ALWAYS, param=None):
    self.mode = mode
    self.param = param
    self._counter = 0

  @classmethod
  def always(cls) -> "Cadence":
    return cls(CadenceMode.ALWAYS)

  @classmethod
  def fixed(cls, period: int) -> "Cadence":
    """Assert every *period*-th cycle (1 = every cycle)."""
    assert period >= 1
    return cls(CadenceMode.FIXED, period)

  @classmethod
  def random(cls, probability: float = 0.5) -> "Cadence":
    """Assert with *probability* each cycle."""
    assert 0.0 < probability <= 1.0
    return cls(CadenceMode.RANDOM, probability)

  def should_assert(self) -> bool:
    if self.mode == CadenceMode.ALWAYS:
      return True
    elif self.mode == CadenceMode.FIXED:
      self._counter += 1
      if self._counter >= self.param:
        self._counter = 0
        return True
      return False
    else:
      return random.random() < self.param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def reset(dut, cycles: int = 10):
  """Drive ``rst`` high for *cycles* clock edges, then de-assert."""
  dut.rst.value = 1
  for _ in range(cycles):
    await RisingEdge(dut.clk)
  dut.rst.value = 0
  await RisingEdge(dut.clk)


async def timeout_watchdog(dut, cycles: int):
  """Fail the test if *cycles* clock edges elapse."""
  for _ in range(cycles):
    await RisingEdge(dut.clk)
  raise AssertionError(f"Timeout after {cycles} cycles")


# ---------------------------------------------------------------------------
# Input drivers  (push messages into the DUT)
# ---------------------------------------------------------------------------


class ESIInputDriver:
  """Base class for channel input drivers."""

  @staticmethod
  def create(dut,
             name: str,
             signaling: ChannelSignaling = ChannelSignaling.ValidReady,
             valid_cadence: Optional[Cadence] = None):
    """Factory: build the right driver for the signaling protocol.

    Args:
      dut: cocotb DUT handle.
      name: base name of the channel port (e.g. ``"inp"``).
      signaling: ``ChannelSignaling.ValidReady`` or ``ChannelSignaling.FIFO``.
      valid_cadence: controls how often valid is asserted.
    """
    cad = valid_cadence or Cadence.always()
    if signaling == ChannelSignaling.ValidReady:
      return ValidReadyInputDriver(dut, name, cad)
    elif signaling == ChannelSignaling.FIFO:
      return FIFOInputDriver(dut, name, cad)
    else:
      raise ValueError(f"Unknown signaling: {signaling}")


class ValidReadyInputDriver(ESIInputDriver):
  """Push data into a ValidReady input channel.

  Port convention (after ``--lower-esi-to-hw --hw-flatten-io``):
    * ``<name>``        – data (input to DUT)
    * ``<name>_valid``  – valid (input to DUT)
    * ``<name>_ready``  – ready (output from DUT)
  """

  def __init__(self, dut, name: str, valid_cadence: Cadence):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.valid = getattr(dut, f"{name}_valid")
    self.ready = getattr(dut, f"{name}_ready")
    self.cadence = valid_cadence
    # Initial state
    self.valid.value = 0

  async def send(self, data: int):
    """Send a single message.  Blocks until the handshake completes."""
    self.data.value = data
    # Wait until cadence says we may assert valid.
    while not self.cadence.should_assert():
      self.valid.value = 0
      await RisingEdge(self.dut.clk)
    self.valid.value = 1
    await RisingEdge(self.dut.clk)
    # Hold valid until ready.
    while self.ready.value == 0:
      await RisingEdge(self.dut.clk)
    self.valid.value = 0

  async def send_all(self, items):
    """Send every item in *items* sequentially."""
    for item in items:
      await self.send(item)


class FIFOInputDriver(ESIInputDriver):
  """Push data into a FIFO input channel.

  Port convention:
    * ``<name>``       – data (input to DUT)
    * ``<name>_empty`` – empty flag (input to DUT, active-high)
    * ``<name>_rden``  – read-enable (output from DUT)
  """

  def __init__(self, dut, name: str, valid_cadence: Cadence):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.empty = getattr(dut, f"{name}_empty")
    self.rden = getattr(dut, f"{name}_rden")
    self.cadence = valid_cadence
    # Initial state: empty
    self.empty.value = 1

  async def send(self, data: int):
    """Present data on the FIFO interface, wait for ``rden``."""
    self.data.value = data
    # Wait until cadence allows us to present data.
    while not self.cadence.should_assert():
      self.empty.value = 1
      await RisingEdge(self.dut.clk)
    self.empty.value = 0
    await RisingEdge(self.dut.clk)
    while self.rden.value == 0:
      await RisingEdge(self.dut.clk)
    self.empty.value = 1

  async def send_all(self, items):
    for item in items:
      await self.send(item)


# ---------------------------------------------------------------------------
# Output monitors  (consume messages from the DUT)
# ---------------------------------------------------------------------------


class ESIOutputMonitor:
  """Base class for channel output monitors."""

  @staticmethod
  def create(dut,
             name: str,
             signaling: ChannelSignaling = ChannelSignaling.ValidReady,
             ready_cadence: Optional[Cadence] = None):
    """Factory: build the right monitor for the signaling protocol.

    Args:
      dut: cocotb DUT handle.
      name: base name of the channel port.
      signaling: ``ChannelSignaling.ValidReady`` or ``ChannelSignaling.FIFO``.
      ready_cadence: controls backpressure (how often ready/rden is asserted).
    """
    cad = ready_cadence or Cadence.always()
    if signaling == ChannelSignaling.ValidReady:
      return ValidReadyOutputMonitor(dut, name, cad)
    elif signaling == ChannelSignaling.FIFO:
      return FIFOOutputMonitor(dut, name, cad)
    else:
      raise ValueError(f"Unknown signaling: {signaling}")


class ValidReadyOutputMonitor(ESIOutputMonitor):
  """Consume data from a ValidReady output channel.

  Port convention:
    * ``<name>``        – data (output from DUT)
    * ``<name>_valid``  – valid (output from DUT)
    * ``<name>_ready``  – ready (input to DUT)
  """

  def __init__(self, dut, name: str, ready_cadence: Cadence):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.valid = getattr(dut, f"{name}_valid")
    self.ready = getattr(dut, f"{name}_ready")
    self.cadence = ready_cadence
    self._q: List[int] = []
    self._running = False
    self.ready.value = 0

  def start(self):
    """Start a background coroutine that continuously reads the channel."""
    if not self._running:
      self._running = True
      cocotb.start_soon(self._run())

  async def _run(self):
    while self._running:
      if self.cadence.should_assert():
        self.ready.value = 1
      else:
        self.ready.value = 0
      await RisingEdge(self.dut.clk)
      if self.valid.value == 1 and self.ready.value == 1:
        self._q.append(int(self.data.value))

  async def recv(self) -> int:
    """Block until a message is available and return it."""
    while len(self._q) == 0:
      await RisingEdge(self.dut.clk)
    return self._q.pop(0)

  async def recv_n(self, n: int) -> List[int]:
    """Receive exactly *n* messages."""
    results = []
    for _ in range(n):
      results.append(await self.recv())
    return results

  def stop(self):
    self._running = False
    self.ready.value = 0


class FIFOOutputMonitor(ESIOutputMonitor):
  """Consume data from a FIFO output channel.

  Port convention:
    * ``<name>``       – data (output from DUT)
    * ``<name>_empty`` – empty flag (output from DUT)
    * ``<name>_rden``  – read-enable (input to DUT)
  """

  def __init__(self, dut, name: str, ready_cadence: Cadence):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.empty = getattr(dut, f"{name}_empty")
    self.rden = getattr(dut, f"{name}_rden")
    self.cadence = ready_cadence
    self._q: List[int] = []
    self._running = False
    self.rden.value = 0

  def start(self):
    if not self._running:
      self._running = True
      cocotb.start_soon(self._run())

  async def _run(self):
    while self._running:
      if self.empty.value == 0 and self.cadence.should_assert():
        self.rden.value = 1
      else:
        self.rden.value = 0
      await RisingEdge(self.dut.clk)
      if self.rden.value == 1 and self.empty.value == 0:
        self._q.append(int(self.data.value))

  async def recv(self) -> int:
    while len(self._q) == 0:
      await RisingEdge(self.dut.clk)
    return self._q.pop(0)

  async def recv_n(self, n: int) -> List[int]:
    results = []
    for _ in range(n):
      results.append(await self.recv())
    return results

  def stop(self):
    self._running = False
    self.rden.value = 0


# ---------------------------------------------------------------------------
# Snoop monitors  (observe without consuming)
# ---------------------------------------------------------------------------


class SnoopMonitor:
  """Observe snoop outputs (valid, ready, data) without consuming the channel.

  Port convention (for snoop outputs exposed as module ports):
    * ``<name>_valid`` – i1 output
    * ``<name>_ready`` – i1 output
    * ``<name>_data``  – data output
  """

  def __init__(self, dut, name: str):
    self.dut = dut
    self.valid = getattr(dut, f"{name}_valid")
    self.ready = getattr(dut, f"{name}_ready")
    self.data = getattr(dut, f"{name}_data")

  async def wait_transaction(self) -> int:
    """Wait until both valid and ready are high and return the data."""
    while True:
      await RisingEdge(self.dut.clk)
      if int(self.valid.value) == 1 and int(self.ready.value) == 1:
        return int(self.data.value)


class XactSnoopMonitor:
  """Observe snoop_xact outputs.

  Port convention:
    * ``<name>_xact`` – transaction indicator (i1)
    * ``<name>_data`` – data
  """

  def __init__(self, dut, name: str):
    self.dut = dut
    self.xact = getattr(dut, f"{name}_xact")
    self.data = getattr(dut, f"{name}_data")

  async def wait_transaction(self) -> int:
    """Wait for a transaction and return the data."""
    while True:
      await RisingEdge(self.dut.clk)
      if int(self.xact.value) == 1:
        return int(self.data.value)


# ---------------------------------------------------------------------------
# High-level test patterns
# ---------------------------------------------------------------------------


async def run_passthrough_test(dut,
                               in_driver: ESIInputDriver,
                               out_monitor: ESIOutputMonitor,
                               values: List[int],
                               timeout_cycles: int = 5000,
                               transform=None):
  """Send *values* through a DUT and verify the output.

  Args:
    transform: optional function mapping input value to expected output.
               Defaults to identity.
  """
  if transform is None:
    transform = lambda x: x

  wd = cocotb.start_soon(timeout_watchdog(dut, timeout_cycles))
  out_monitor.start()

  async def writer():
    await in_driver.send_all(values)

  cocotb.start_soon(writer())

  results = await out_monitor.recv_n(len(values))
  expected = [transform(v) for v in values]
  assert results == expected, f"Mismatch:\n  got:      {results}\n  expected: {expected}"
  out_monitor.stop()


async def run_fork_test(dut,
                        in_driver: ESIInputDriver,
                        out_a_monitor: ESIOutputMonitor,
                        out_b_monitor: ESIOutputMonitor,
                        values: List[int],
                        timeout_cycles: int = 5000):
  """Send *values* and verify both fork outputs receive them."""
  wd = cocotb.start_soon(timeout_watchdog(dut, timeout_cycles))
  out_a_monitor.start()
  out_b_monitor.start()

  async def writer():
    await in_driver.send_all(values)

  cocotb.start_soon(writer())

  results_a = await out_a_monitor.recv_n(len(values))
  results_b = await out_b_monitor.recv_n(len(values))

  assert results_a == values, \
      f"Fork A mismatch:\n  got: {results_a}\n  exp: {values}"
  assert results_b == values, \
      f"Fork B mismatch:\n  got: {results_b}\n  exp: {values}"
  out_a_monitor.stop()
  out_b_monitor.stop()
