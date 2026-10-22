import cocotb
from cocotb.triggers import Timer
import cocotb.clock
import random


async def clock(dut):
  dut.clk.value = 0
  await Timer(1, units='ns')
  dut.clk.value = 1
  await Timer(1, units='ns')


async def initDut(dut):
  # Reset
  dut.rst.value = 1
  await clock(dut)
  dut.rst.value = 0
  await clock(dut)


async def write(dut, value):
  dut.inp.value = value
  assert dut.full.value == 0
  dut.wrEn.value = 1
  await clock(dut)
  dut.wrEn.value = 0
  await clock(dut)


async def combRead(dut, out):
  dut.rdEn.value = 1
  # Combinational reads, so let the model settle before reading
  await Timer(1, units='ns')
  return dut.out.value


async def read(dut):
  data = await combRead(dut, dut.out.value)
  await clock(dut)
  dut.rdEn.value = 0
  await clock(dut)
  return data


async def readWrite(dut, value):
  dut.inp.value = value
  dut.wrEn.value = 1
  data = await combRead(dut, dut.out.value)
  await clock(dut)
  dut.rdEn.value = 0
  dut.wrEn.value = 0
  await clock(dut)
  return data


FIFO_DEPTH = 6
FIFO_ALMOST_FULL = 2
FIFO_ALMOST_EMPTY = 1


@cocotb.test()
async def test_separate_read_write(dut):
  """Run a test where we incrementally write and read values from 1 to
  FIFO_DEPTH values."""

  dut.inp.value = 0
  dut.rdEn.value = 0
  dut.wrEn.value = 0
  await initDut(dut)
  for i in range(1, FIFO_DEPTH + 1):
    for j in range(i):
      await write(dut, 42 + j)

    if i >= FIFO_ALMOST_FULL:
      assert dut.almost_full.value == 1

    if i <= FIFO_ALMOST_EMPTY:
      assert dut.almost_empty.value == 1

    if i == FIFO_DEPTH:
      assert dut.full.value == 1

    for j in range(i):
      assert await read(dut) == 42 + j

    assert dut.empty.value == 1


@cocotb.test()
async def test_concurrent_read_write(dut):
  """Fill up the FIFO halfway and concurrently read and write. Should be able
  to do this continuously."""

  dut.inp.value = 0
  dut.rdEn.value = 0
  dut.wrEn.value = 0
  await initDut(dut)

  counter = 0
  for i in range(FIFO_DEPTH // 2):
    await write(dut, counter)
    counter += 1

  for i in range(FIFO_DEPTH * 2):
    expected_value = counter - FIFO_DEPTH // 2
    print(f"expected_value: {expected_value}")
    assert await readWrite(dut, counter) == expected_value
    assert dut.full.value == 0
    assert dut.empty.value == 0
    counter += 1


@cocotb.test()
async def test_random(dut):
  """Random reads and writes, which never read while empty or write while full,
  at rates which change every so often so that the FIFO fills and drains. Every
  output is checked against a model on every cycle."""

  rng = random.Random(1)
  dut.inp.value = 0
  dut.rdEn.value = 0
  dut.wrEn.value = 0
  await initDut(dut)

  model = []
  for cycle in range(4000):
    if cycle % 50 == 0:
      wrRate, rdRate = rng.random(), rng.random()
    assert int(dut.full.value) == int(len(model) == FIFO_DEPTH)
    assert int(dut.empty.value) == int(len(model) == 0)
    assert int(dut.almost_full.value) == int(len(model) >= FIFO_ALMOST_FULL)
    assert int(dut.almost_empty.value) == int(len(model) <= FIFO_ALMOST_EMPTY)

    wr = len(model) < FIFO_DEPTH and rng.random() < wrRate
    rd = len(model) > 0 and rng.random() < rdRate
    value = rng.getrandbits(32)
    dut.inp.value = value
    dut.wrEn.value = int(wr)
    dut.rdEn.value = int(rd)
    if rd:
      assert int(await combRead(dut, dut.out.value)) == model.pop(0)
    if wr:
      model.append(value)
    await clock(dut)
