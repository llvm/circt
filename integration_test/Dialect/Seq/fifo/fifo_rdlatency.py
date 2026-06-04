import cocotb
from cocotb.triggers import Timer

FIFO_DEPTH = 6
FIFO_ALMOST_FULL = 2
FIFO_ALMOST_EMPTY = 1


async def clock(dut):
  dut.clk.value = 0
  await Timer(1, unit='ns')
  dut.clk.value = 1
  await Timer(1, unit='ns')


async def initDut(dut):
  dut.inp.value = 0
  dut.rdEn.value = 0
  dut.wrEn.value = 0
  dut.rst.value = 1
  await clock(dut)
  dut.rst.value = 0
  await clock(dut)


async def write(dut, value):
  assert dut.full.value == 0
  dut.inp.value = value
  dut.wrEn.value = 1
  await clock(dut)
  dut.wrEn.value = 0
  dut.inp.value = 0


async def read(dut):
  """Registered read (rd_latency 1): assert rdEn for one cycle, then the data
  is available on the output on the following cycle."""
  assert dut.empty.value == 0
  dut.rdEn.value = 1
  await clock(dut)
  dut.rdEn.value = 0
  # Data is valid one cycle after rdEn was asserted. Let the model settle.
  await Timer(1, unit='ns')
  return int(dut.out.value)


@cocotb.test()
async def test_write_then_read(dut):
  """Incrementally write and then read values, one element at a time."""
  await initDut(dut)
  for i in range(1, FIFO_DEPTH + 1):
    for j in range(i):
      await write(dut, 42 + j)

    if i >= FIFO_ALMOST_FULL:
      assert dut.almost_full.value == 1
    if i == FIFO_DEPTH:
      assert dut.full.value == 1

    for j in range(i):
      data = await read(dut)
      assert data == 42 + j, f"expected {42 + j}, got {data}"

    assert dut.empty.value == 1


@cocotb.test()
async def test_fill_drain(dut):
  """Fill the FIFO completely, then drain it, checking FIFO ordering."""
  await initDut(dut)
  for i in range(FIFO_DEPTH):
    await write(dut, 100 + i)
  assert dut.full.value == 1

  for i in range(FIFO_DEPTH):
    data = await read(dut)
    assert data == 100 + i, f"expected {100 + i}, got {data}"
  assert dut.empty.value == 1
