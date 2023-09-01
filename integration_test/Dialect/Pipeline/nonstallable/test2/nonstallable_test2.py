import cocotb
from cocotb.triggers import Timer
import cocotb.clock

from nonstallable_helper import nonstallable_test


@cocotb.test()
async def test2(dut):
  await nonstallable_test(dut, 5, [1, 3])
