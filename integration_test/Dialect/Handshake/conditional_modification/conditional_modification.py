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
  ins, outs = getPorts(dut, ["in0", "in1", "in2", "inCtrl"],
                       ["out0", "out1", "outCtrl"])

  [in0, in1, in2, inCtrl] = ins
  [out0, out1, outCtrl] = outs
  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  in0.setValid(0)
  in1.setValid(0)
  in2.setValid(0)
  inCtrl.setValid(0)

  out0.setReady(1)
  out1.setReady(1)
  outCtrl.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)

  return ins, outs


@cocotb.test()
async def oneInput(dut):
  [in0, in1, in2, inCtrl], [out0, out1, outCtrl] = await initDut(dut)

  out0Check = cocotb.start_soon(out0.checkOutputs([18]))
  out1Check = cocotb.start_soon(out1.checkOutputs([24]))

  in0Send = cocotb.start_soon(in0.send(18))
  in1Send = cocotb.start_soon(in1.send(24))
  in2Send = cocotb.start_soon(in2.send(0))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await in1Send
  await in2Send
  await inCtrlSend

  await out0Check
  await out1Check


@cocotb.test()
async def multiple(dut):
  [in0, in1, in2, inCtrl], [out0, out1, outCtrl] = await initDut(dut)

  out0Check = cocotb.start_soon(out0.checkOutputs([18, 42, 42]))
  # COCOTB treats all integers as unsigned, thus we have to compare with the
  # two's complement representation
  out1Check = cocotb.start_soon(out1.checkOutputs([24, 2**32 - 6, 42]))

  inputs = [(18, 24, 0), (18, 24, 1), (42, 0, 1)]
  for (d0, d1, cond) in inputs:
    in0Send = cocotb.start_soon(in0.send(d0))
    in1Send = cocotb.start_soon(in1.send(d1))
    in2Send = cocotb.start_soon(in2.send(cond))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await in1Send
    await in2Send
    await inCtrlSend

  await out0Check
  await out1Check
