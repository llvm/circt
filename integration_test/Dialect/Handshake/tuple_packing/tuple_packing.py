import cocotb
import cocotb.clock
from cocotb.triggers import FallingEdge, RisingEdge

# Hack to allow imports from parent directory
import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from helper import HandshakePort, getPorts


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = getPorts(dut, ["in0", "inCtrl"],
                                            ["out0", "outCtrl"])

  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  in0.setValid(0)
  inCtrl.setValid(0)

  out0.setReady(1)
  outCtrl.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)

  resCheck = cocotb.start_soon(out0.checkOutputs([42]))

  in0Send = cocotb.start_soon(in0.send((24, 18)))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await resCheck
