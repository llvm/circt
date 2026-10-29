"""Test `esi.buffer` with channels which have a non-zero data delay.

The input and output of each buffer have the same data delay: the data for a
transfer is on the data signal that many cycles after the handshake.

All inputs are driven, and all outputs sampled, on the falling clock edge. That
is unambiguous since every control output of the DUT which the testbench reads
(`full`, `*_ready`, `*_valid`, `*_empty`) comes straight from registers.
"""

import random
from typing import Dict, List

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

NUM_ITEMS = 200
MAX_CYCLES = 20 * NUM_ITEMS

# Port name prefix -> data delay.
FIFO_PORTS = {"fifo11": 1, "fifo12": 1, "fifo21": 2, "fifo22": 2}
VR_PORTS = {"vr1": 1, "vr2": 2, "vr3": 3}
VO_PORTS = {"vo1": 1}
# Outputs which use FIFO signaling. All others are ValidReady.
FIFO_OUTPUTS = {"fifo21"}


def junk() -> int:
  return random.randrange(1 << 16)


async def init(dut):
  for p in FIFO_PORTS:
    getattr(dut, f"{p}_din").value = 0
    getattr(dut, f"{p}_push").value = 0
  for p in list(VR_PORTS) + list(VO_PORTS):
    getattr(dut, f"{p}_in").value = 0
    getattr(dut, f"{p}_in_valid").value = 0
  for p in list(FIFO_PORTS) + list(VR_PORTS) + list(VO_PORTS):
    if p in FIFO_OUTPUTS:
      getattr(dut, f"{p}_out_rden").value = 0
    else:
      getattr(dut, f"{p}_out_ready").value = 0

  cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
  dut.rst.value = 1
  for _ in range(5):
    await RisingEdge(dut.clk)
  await FallingEdge(dut.clk)
  dut.rst.value = 0


async def fifo_source(dut, prefix: str, values: List[int], push_prob: float):
  """Push into the `seq.fifo` which sources the FIFO(L) channel."""
  din = getattr(dut, f"{prefix}_din")
  push = getattr(dut, f"{prefix}_push")
  full = getattr(dut, f"{prefix}_full")
  i = 0
  while i < len(values):
    await FallingEdge(dut.clk)
    if random.random() < push_prob and full.value == 0:
      din.value = values[i]
      push.value = 1
      i += 1
    else:
      din.value = junk()
      push.value = 0
  await FallingEdge(dut.clk)
  push.value = 0


async def delayed_source(dut, prefix: str, delay: int, values: List[int],
                         valid_prob: float, has_ready: bool):
  """Drive a ValidReady(delay) or ValidOnly(delay) channel. The data for a
  transfer is driven `delay` cycles after the handshake. Junk otherwise."""
  data = getattr(dut, f"{prefix}_in")
  valid = getattr(dut, f"{prefix}_in_valid")
  ready = getattr(dut, f"{prefix}_in_ready") if has_ready else None
  scheduled: Dict[int, int] = {}
  cycle = 0
  i = 0
  while i < len(values) or scheduled:
    await FallingEdge(dut.clk)
    data.value = scheduled.pop(cycle, junk())
    if i < len(values) and random.random() < valid_prob:
      valid.value = 1
      if ready is None or ready.value == 1:
        scheduled[cycle + delay] = values[i]
        i += 1
    else:
      valid.value = 0
    cycle += 1


class Sink:
  """Consume a ValidReady or FIFO output, recording the transfer cycles and
  reading each transfer's data `delay` cycles later."""

  def __init__(self, dut, prefix: str, delay: int, ready_prob: float):
    self.dut = dut
    name = f"{prefix}_out"
    self.data = getattr(dut, name)
    self.fifo = prefix in FIFO_OUTPUTS
    if self.fifo:
      self.empty = getattr(dut, f"{name}_empty")
      self.rden = getattr(dut, f"{name}_rden")
    else:
      self.valid = getattr(dut, f"{name}_valid")
      self.ready = getattr(dut, f"{name}_ready")
    self.delay = delay
    self.ready_prob = ready_prob
    self.received: List[int] = []
    self.cycles: List[int] = []

  async def run(self):
    cycle = 0
    pending: List[int] = []
    while True:
      await FallingEdge(self.dut.clk)
      if pending and pending[0] == cycle:
        pending.pop(0)
        self.received.append(int(self.data.value))
      want = random.random() < self.ready_prob
      if self.fifo:
        # FIFO signaling: only read when not empty.
        xfer = want and self.empty.value == 0
        self.rden.value = int(xfer)
      else:
        self.ready.value = int(want)
        xfer = want and self.valid.value == 1
      if xfer:
        self.cycles.append(cycle)
        pending.append(cycle + self.delay)
      cycle += 1


async def run_all(dut, src_prob: float, sink_prob: float):
  await init(dut)
  sent = {}
  sinks = {}
  for p, delay in FIFO_PORTS.items():
    sent[p] = [junk() for _ in range(NUM_ITEMS)]
    cocotb.start_soon(fifo_source(dut, p, sent[p], src_prob))
    sinks[p] = Sink(dut, p, delay, sink_prob)
  for p, delay in VR_PORTS.items():
    sent[p] = [junk() for _ in range(NUM_ITEMS)]
    cocotb.start_soon(
        delayed_source(dut, p, delay, sent[p], src_prob, has_ready=True))
    sinks[p] = Sink(dut, p, delay, sink_prob)
  for p, delay in VO_PORTS.items():
    sent[p] = [junk() for _ in range(NUM_ITEMS)]
    cocotb.start_soon(
        delayed_source(dut, p, delay, sent[p], src_prob, has_ready=False))
    # ValidOnly has no backpressure, so the sink must always be ready.
    sinks[p] = Sink(dut, p, delay, 1.0)
  for s in sinks.values():
    cocotb.start_soon(s.run())

  for _ in range(MAX_CYCLES):
    await FallingEdge(dut.clk)
    if all(len(s.received) >= NUM_ITEMS for s in sinks.values()):
      break
  # Let anything extra (duplicates) show up.
  for _ in range(20):
    await FallingEdge(dut.clk)

  for p, s in sinks.items():
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(s.received, sent[p])) if a != b),
        min(len(s.received), len(sent[p])))
    assert s.received == sent[p], (
        f"{p}: mismatch. First difference at index {first_diff}; "
        f"received {len(s.received)} of {len(sent[p])}")
  return sinks


@cocotb.test()
async def back_to_back(dut):
  """Source and sink always ready: everything in order, one per cycle."""
  random.seed(0)
  sinks = await run_all(dut, src_prob=1.0, sink_prob=1.0)
  for p, s in sinks.items():
    gaps = [b - a for a, b in zip(s.cycles, s.cycles[1:]) if b - a != 1]
    assert not gaps, f"{p}: bubbles in the output stream: {gaps}"


@cocotb.test()
async def random_backpressure(dut):
  """Random source gaps and sink backpressure: everything in order."""
  random.seed(1)
  await run_all(dut, src_prob=0.7, sink_prob=0.5)


@cocotb.test()
async def heavy_backpressure(dut):
  """Bursty source, mostly stalled sink: exercises the buffer being full."""
  random.seed(2)
  await run_all(dut, src_prob=0.95, sink_prob=0.2)
