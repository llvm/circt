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


async def initDut(dut):
  ins, outs = getPorts(dut, ["in0", "in1", "inCtrl"], ["out0", "outCtrl"])
  [in0, in1, inCtrl] = ins
  [out0, outCtrl] = outs
  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  in0.setValid(0)
  in1.setValid(0)
  inCtrl.setValid(0)

  out0.setReady(1)
  outCtrl.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)
  return ins, outs


@cocotb.test()
async def oneInput(dut):
  [in0, in1, inCtrl], [out0, outCtrl] = await initDut(dut)

  resCheck = cocotb.start_soon(out0.checkOutputs([42]))

  in0Send = cocotb.start_soon(in0.send(42))
  in1Send = cocotb.start_soon(in1.send(1))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await in1Send
  await inCtrlSend

  await resCheck


@cocotb.test()
async def multipleInputs(dut):
  [in0, in1, inCtrl], [out0, outCtrl] = await initDut(dut)

  resCheck = cocotb.start_soon(out0.checkOutputs([42, 42, 10, 10, 10]))

  inputs = [(42, 1), (0, 0), (10, 1), (42, 0), (99, 0)]
  for (data, w) in inputs:
    in0Send = cocotb.start_soon(in0.send(data))
    in1Send = cocotb.start_soon(in1.send(w))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await in1Send
    await inCtrlSend

  await resCheck
