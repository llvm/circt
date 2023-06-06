import cocotb
from cocotb.triggers import Timer
import cocotb.clock


async def clock(dut):
  dut.clock.value = 0
  await Timer(1, units='ns')
  dut.clock.value = 1
  await Timer(1, units='ns')


async def initDut(dut):
  """
  Initializes a dut by adding a clock, setting initial valid and ready flags,
  and performing a reset.
  """
  # Reset
  dut.reset.value = 1
  await clock(dut)
  dut.reset.value = 0
  await clock(dut)


@cocotb.test()
async def test1(dut):
  await initDut(dut)

  dut.arg0.value = 42
  dut.arg1.value = 24
  dut.go.value = 1

  for i in range(3):
    await clock(dut)

  assert dut.out == 174, f"Expected 174, got {dut.out}"
