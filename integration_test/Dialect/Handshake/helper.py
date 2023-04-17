import cocotb.clock
from cocotb.triggers import RisingEdge, ReadOnly


class HandshakePort:
  """
  Helper class that encapsulates a handshake port from the DUT.
  """

  def __init__(self, dut, rdy, val):
    self.dut = dut
    self.ready = rdy
    self.valid = val

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
    # Make sure that changes to ready are propagated before it is checked.
    await ReadOnly()
    directSend = self.isReady()
    await self.waitUntilReady()

    if (directSend):
      # If it was initially ready, the handshake happens in the current cycle.
      # Thus the invalidation has to wait until the next cycle
      await RisingEdge(self.dut.clock)

    self.setValid(0)

  async def send(self, val=None):
    self.setValid(1)
    await self.awaitHandshake()

  async def awaitNOutputs(self, n):
    assert (self.isReady())
    for _ in range(n):
      await self.waitUntilValid()
      await RisingEdge(self.dut.clock)


class HandshakeDataPort(HandshakePort):
  """
  A handshaked port with a data field.
  """

  def __init__(self, dut, rdy, val, data):
    super().__init__(dut, rdy, val)
    self.data = data

  async def send(self, val):
    self.data.value = val
    await super().send()

  async def checkOutputs(self, results):
    assert (self.isReady())
    for res in results:
      await self.waitUntilValid()
      assert self.data.value == res, f"Expected {res}, got {self.data.value}"
      await RisingEdge(self.dut.clock)


class HandshakeTuplePort(HandshakePort):
  """
  A handshaked port that sends a tuple.
  """

  def __init__(self, dut, rdy, val, fields):
    super().__init__(dut, rdy, val)
    self.fields = fields

  async def send(self, val):
    assert (len(list(val)) == len(self.fields))
    for (f, v) in zip(self.fields, list(val)):
      f.value = v

    await super().send()

  async def checkOutputs(self, results):
    assert (self.isReady())
    for res in results:
      await self.waitUntilValid()

      assert (len(list(res)) == len(self.fields))
      for (f, r) in zip(self.fields, list(res)):
        assert (f.value == r)
      await RisingEdge(self.dut.clock)


def _findPort(dut, name):
  """
  Checks if dut has a port of the provided name. Either throws an exception or
  returns a HandshakePort that encapsulates the dut's interface.
  """
  readyName = f"{name}_ready"
  validName = f"{name}_valid"
  dataName = f"{name}_data"
  if (not hasattr(dut, readyName) or not hasattr(dut, validName)):
    raise Exception(f"dut does not have a port named {name}")

  ready = getattr(dut, readyName)
  valid = getattr(dut, validName)
  data = getattr(dut, dataName, None)

  if data is None:
    # Try with just the base name - this is the case for HandshakeToHW (ESI standard)
    data = getattr(dut, name, None)

  # Needed, as it otherwise would try to resolve the value
  hasData = not isinstance(data, type(None))
  if hasData:
    return HandshakeDataPort(dut, ready, valid, data)

  isCtrl = not hasattr(dut, f"{name}_data_field0")

  if (isCtrl):
    return HandshakePort(dut, ready, valid)

  fields = []
  i = 0
  while hasattr(dut, f"{name}_data_field{i}"):
    fields.append(getattr(dut, f"{name}_data_field{i}"))
    i += 1

  return HandshakeTuplePort(dut, ready, valid, fields)


def getPorts(dut, inNames, outNames):
  """
  Helper function to produce in and out ports for the provided dut.
  """
  ins = [_findPort(dut, name) for name in inNames]
  outs = [_findPort(dut, name) for name in outNames]
  return ins, outs


async def initDut(dut, inNames, outNames):
  """
  Initializes a dut by adding a clock, setting initial valid and ready flags,
  and performing a reset.
  """
  ins, outs = getPorts(dut, inNames, outNames)

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
