import random
from collections import deque

import cocotb
from cocotb.triggers import Timer

FIFO_DEPTH = 4
RD_LATENCY = 3


async def clock(dut):
  dut.clk.value = 0
  await Timer(1, units='ns')
  dut.clk.value = 1
  await Timer(1, units='ns')


class FifoTester:
  """Cycle-level driver which only pushes when not full and only pops when not
  empty. Checks the data of each pop RD_LATENCY cycles later."""

  def __init__(self, dut):
    self.dut = dut
    self.cycle = 0
    self.model = deque()
    # (cycle on which the data is valid, expected data) for in-flight reads.
    self.pending = deque()
    self.reads = 0

  async def reset(self):
    dut = self.dut
    dut.inp.value = 0
    dut.rdEn.value = 0
    dut.wrEn.value = 0
    dut.rst.value = 1
    await clock(dut)
    dut.rst.value = 0
    await clock(dut)

  async def step(self, want_push, value, want_pop):
    dut = self.dut
    empty = int(dut.empty.value) == 1
    full = int(dut.full.value) == 1
    assert empty == (len(self.model) == 0), f"cycle {self.cycle}: empty"
    # Popped slots stay reserved until their delayed read has happened.
    expected_full = len(self.model) + len(self.pending) == FIFO_DEPTH
    assert full == expected_full, f"cycle {self.cycle}: full"

    push = want_push and not full
    pop = want_pop and not empty
    dut.inp.value = value
    dut.wrEn.value = int(push)
    dut.rdEn.value = int(pop)
    if pop:
      self.pending.append((self.cycle + RD_LATENCY, self.model.popleft()))
    if push:
      self.model.append(value)

    await clock(dut)
    self.cycle += 1
    dut.wrEn.value = 0
    dut.rdEn.value = 0

    if self.pending and self.pending[0][0] == self.cycle:
      _, expected = self.pending.popleft()
      actual = int(dut.out.value)
      assert actual == expected, \
          f"cycle {self.cycle}: read {actual}, expected {expected}"
      self.reads += 1
    return push, pop

  async def drain(self):
    while self.model or self.pending:
      await self.step(False, 0, True)


@cocotb.test()
async def test_overwrite_pattern(dut):
  """Keep the FIFO nearly full while pushing and popping together. This used to
  return entries which had already been overwritten by later pushes."""
  tb = FifoTester(dut)
  await tb.reset()
  for v in range(3):
    push, _ = await tb.step(True, v, False)
    assert push
  for v in range(3, 12):
    push = False
    while not push:
      push, _ = await tb.step(True, v, True)
  await tb.drain()
  assert tb.reads == 12


@cocotb.test()
async def test_fill(dut):
  """Fill the FIFO, then drain it."""
  tb = FifoTester(dut)
  await tb.reset()
  v = 0
  while int(dut.full.value) == 0:
    await tb.step(True, v, False)
    v += 1
  assert v == FIFO_DEPTH
  await tb.drain()
  assert tb.reads == FIFO_DEPTH


@cocotb.test()
async def test_throughput(dut):
  """With depth > rd_latency, a push and a pop can happen every cycle."""
  tb = FifoTester(dut)
  await tb.reset()
  await tb.step(True, 0, False)
  for v in range(1, 33):
    push, pop = await tb.step(True, v, True)
    assert push and pop, f"cycle {tb.cycle}: push={push} pop={pop}"
  await tb.drain()
  assert tb.reads == 33


@cocotb.test()
async def test_random(dut):
  """Random pushes and pops."""
  rng = random.Random(11332)
  tb = FifoTester(dut)
  await tb.reset()
  for _ in range(1000):
    await tb.step(rng.random() < 0.6, rng.randrange(1 << 16),
                  rng.random() < 0.6)
  await tb.drain()
