import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  resCheck = cocotb.start_soon(out0.checkOutputs([42]))

  in0Send = cocotb.start_soon(in0.send((24, 18)))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await resCheck
