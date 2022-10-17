import cocotb
from helper import initDut
import random

random.seed(0)


def kernel(e):
  if (e >> 1) & 0x1:
    if (e >> 2) & 0x1:
      return e + 4
    return e * 10
  return e


def getOutput(t):
  return tuple(map(kernel, t))


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  inputs = [(8, 8, 4, 8, 5, 3, 1, 0)]
  outputs = [getOutput(i) for i in inputs]
  resCheck = cocotb.start_soon(out0.checkOutputs(outputs))

  in0Send = cocotb.start_soon(in0.send(inputs[0]))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await resCheck


def randomTuple():
  return tuple([random.randint(0, 100) for _ in range(8)])


@cocotb.test()
async def multipleInputs(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  N = 10
  inputs = [randomTuple() for _ in range(N)]

  outputs = [getOutput(i) for i in inputs]
  resCheck = cocotb.start_soon(out0.checkOutputs(outputs))

  for i in inputs:
    print("in: ", i)
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await resCheck
