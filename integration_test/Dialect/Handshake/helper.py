from cocotb.triggers import RisingEdge


class HandshakePort:
  """
  Helper class that encapsulates a handshake port from the DUT.
  """

  def __init__(self, dut, rdy, val, data=None):
    self.dut = dut
    self.ready = rdy
    self.valid = val
    self.data = data

  def isReady(self):
    return self.ready.value.is_resolvable and self.ready.value == 1

  def setReady(self, v):
    self.ready.value = v

  def isValid(self):
    return self.valid.value.is_resolvable and self.valid.value == 1

  def setValid(self, v):
    self.valid.value = v

  async def waitUntilReady(self):
    while (not self.isReady()):
      await RisingEdge(self.dut.clock)

  async def waitUntilValid(self):
    while (not self.isValid()):
      await RisingEdge(self.dut.clock)

  async def awaitHandshake(self):
    directSend = self.isReady()
    await self.waitUntilReady()

    if (directSend):
      # If it was initially ready, the handshake happens in the current cycle.
      # Thus the invalidation has to wait until the next cycle
      await RisingEdge(self.dut.clock)

    self.setValid(0)

    if (not directSend):
      # The handshake happend already, so we only have to ensure that valid 0
      # gets communicated correctly.
      await RisingEdge(self.dut.clock)

  async def send(self, val=None):
    if (val is not None):
      self.data.value = val
    self.setValid(1)
    await self.awaitHandshake()

  async def checkOutputs(self, results):
    assert (self.isReady())
    for res in results:
      await self.waitUntilValid()
      assert (self.data.value == res)
      await RisingEdge(self.dut.clock)


def getPorts(dut, inNames, outNames):
  """
  Helper function to produce in and out ports for the provided dut.
  """
  ins = []
  outs = []

  for inName in inNames:
    readyName = f"{inName}_ready"
    validName = f"{inName}_valid"
    dataName = f"{inName}_data"
    if (not hasattr(dut, readyName) or not hasattr(dut, validName)):
      raise Exception(f"dut does not have the input {i}")

    ready = getattr(dut, readyName)
    valid = getattr(dut, validName)

    ins.append(HandshakePort(dut, ready, valid, getattr(dut, dataName, None)))

  for outName in outNames:
    readyName = f"{outName}_ready"
    validName = f"{outName}_valid"
    dataName = f"{outName}_data"
    if (not hasattr(dut, readyName) or not hasattr(dut, validName)):
      raise Exception(f"dut does not have the output {i}")

    ready = getattr(dut, readyName)
    valid = getattr(dut, validName)

    outs.append(HandshakePort(dut, ready, valid, getattr(dut, dataName, None)))
  return ins, outs
