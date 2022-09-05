import cocotb
import cocotb.clock
from cocotb.triggers import FallingEdge, RisingEdge
from helper import HandshakePort, getPorts
import math


async def initDut(dut):
  ins, outs = getPorts(dut, ["in0", "inCtrl"], ["out0", "out1", "outCtrl"])

  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  for i in ins:
    i.setValid(0)

  for o in outs:
    o.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)
  return ins, outs


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, out1, outCtrl] = await initDut(dut)
  out0Check = cocotb.start_soon(out0.checkOutputs([15]))
  out1Check = cocotb.start_soon(out1.checkOutputs([120]))

  in0Send = cocotb.start_soon(in0.send(5))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await out0Check
  await out1Check


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, out1, outCtrl] = await initDut(dut)

  N = 10
  res0 = [i * (i + 1) / 2 for i in range(N)]
  res1 = [math.factorial(i) for i in range(N)]

  out0Check = cocotb.start_soon(out0.checkOutputs(res0))
  out1Check = cocotb.start_soon(out1.checkOutputs(res1))

  for i in range(N):
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await out0Check
  await out1Check
