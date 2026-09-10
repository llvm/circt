import cocotb
from nonstallable_helper import nonstallable_test


@cocotb.test()
async def test1(dut):
  await nonstallable_test(dut, [0, 1, 1, 0, 0])
