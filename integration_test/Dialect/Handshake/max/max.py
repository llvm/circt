import cocotb
import cocotb.clock
from cocotb.triggers import FallingEdge, RisingEdge
from helper import HandshakePort, getPorts
import random


async def initDut(dut):
  ins, outs = getPorts(dut, [f"in{i}" for i in range(8)] + ["inCtrl"],
                       ["out0", "outCtrl"])
  inCtrl = ins[-1]
  inPorts = ins[:-1]
  [out0, outCtrl] = outs
  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  for i in inPorts:
    i.setValid(0)

  inCtrl.setValid(0)

  out0.setReady(1)
  outCtrl.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)
  return ins, outs


def genInOutPair():
  inputs = [random.randint(0, 1000) for _ in range(8)]
  output = max(inputs)
  return inputs, output


@cocotb.test()
async def oneInput(dut):
  ins, [out0, outCtrl] = await initDut(dut)
  inCtrl = ins[-1]
  inPorts = ins[:-1]

  inputs, output = genInOutPair()
  resCheck = cocotb.start_soon(out0.checkOutputs([output]))

  sends = [
      cocotb.start_soon(inPort.send(data))
      for [inPort, data] in zip(inPorts, inputs)
  ]
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  for s in sends:
    await s
  await inCtrlSend

  await resCheck


@cocotb.test()
async def multipleInputs(dut):
  ins, [out0, outCtrl] = await initDut(dut)
  inCtrl = ins[-1]
  inPorts = ins[:-1]

  inOutPairs = [genInOutPair() for _ in range(8)]
  resCheck = cocotb.start_soon(
      out0.checkOutputs([out for (_, out) in inOutPairs]))

  for (inputs, _) in inOutPairs:
    sends = [
        cocotb.start_soon(inPort.send(data))
        for [inPort, data] in zip(inPorts, inputs)
    ]
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    for s in sends:
      await s
    await inCtrlSend

  await resCheck
