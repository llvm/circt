import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [inCtrl,
   startTokens], [outCtrl] = await initDut(dut, ["inCtrl", "startTokens"],
                                           ["outCtrl"])

  await startTokens.send()
  outCtrlCheck = cocotb.start_soon(outCtrl.awaitNOutputs(1))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await inCtrlSend
  await outCtrlCheck
