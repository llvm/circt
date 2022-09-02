import cocotb
import cocotb.clock
from cocotb.triggers import FallingEdge, RisingEdge
from helper import HandshakePort, getPorts


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
  #out1Check = cocotb.start_soon(out1.checkOutputs([120]))

  in0Send = cocotb.start_soon(in0.send(5))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await out0Check
  #await out1Check


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, out1, outCtrl] = await initDut(dut)

  out0Check = cocotb.start_soon(out0.checkOutputs([1,3,6,10,15,21,28]))
  #out1Check = cocotb.start_soon(out1.checkOutputs([1,2,6,24,120,720,5040]))

  for i in [1,2,3,4,5,6,7]:
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await out0Check
  #await out1Check
