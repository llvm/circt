from typing import List, Optional
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def init(dut, timeout: Optional[int] = None):
  # Create a 10ns period (100MHz) clock on port clock
  clk = Clock(dut.clk, 10, units="ns")
  cocotb.start_soon(clk.start())  # Start the clock

  # Reset
  dut.rst.value = 1
  for i in range(10):
    await RisingEdge(dut.clk)
  dut.rst.value = 0
  await RisingEdge(dut.clk)

  if timeout is None:
    return

  async def timeout_fn():
    for i in range(timeout):
      await RisingEdge(dut.clk)
    assert False, "Timeout"

  cocotb.start_soon(timeout_fn())


class ESIInputPort:

  async def write(self, data):
    raise RuntimeError("Must be implemented by subclass")


class ESIFifoInputPort(ESIInputPort):

  def __init__(self, dut, name):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.rden = getattr(dut, f"{name}_rden")
    self.empty = getattr(dut, f"{name}_empty")
    # Configure initial state
    self.empty.value = 1

  async def write(self, data: int):
    self.data.value = data
    self.empty.value = 0
    await RisingEdge(self.dut.clk)
    while self.rden.value == 0:
      await RisingEdge(self.dut.clk)
    self.empty.value = 1


class ESIOutputPort:

  async def read(self) -> Optional[int]:
    raise RuntimeError("Must be implemented by subclass")

  async def cmd_read(self):
    raise RuntimeError("Must be implemented by subclass")


class ESIFifoOutputPort(ESIOutputPort):

  def __init__(self, dut, name: str, latency: int = 0):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.rden = getattr(dut, f"{name}_rden")
    self.empty = getattr(dut, f"{name}_empty")
    self.latency = latency
    # Configure initial state
    self.rden.value = 0
    self.running = 0
    self.q: List[int] = []

  async def init_read(self):

    async def read_after_latency():
      for i in range(self.latency):
        await RisingEdge(self.dut.clk)
      self.q.append(self.data.value)

    self.running = 1
    self.empty.value = 0
    while await RisingEdge(self.dut.clk):
      if self.rden.value == 1:
        cocotb.start_soon(read_after_latency())

  async def read(self) -> Optional[int]:
    if len(self.q) == 0:
      await RisingEdge(self.dut.clk)
    return self.q.pop(0)

  async def cmd_read(self):
    if self.running == 0:
      cocotb.start_soon(self.init_read())

    while self.empty.value == 1:
      await RisingEdge(self.dut.clk)
    self.rden.value = 1
    await RisingEdge(self.dut.clk)
    self.rden.value = 0


@cocotb.test()
async def fillAndDrain(dut):
  in_port = ESIFifoInputPort(dut, "in")
  out_port = ESIFifoOutputPort(dut, "out", 2)
  await init(dut, timeout=10000)

  for i in range(10):
    for i in range(12):
      await in_port.write(i)
    for i in range(12):
      await out_port.cmd_read()
    for i in range(12):
      data = await out_port.read()
      # print(f"{i:2}: 0x{int(data):016x}")
      assert data == i

  for i in range(4):
    await RisingEdge(dut.clk)


@cocotb.test()
async def backToBack(dut):
  in_port = ESIFifoInputPort(dut, "in")
  out_port = ESIFifoOutputPort(dut, "out", 2)
  await init(dut)

  NUM_ITERS = 500

  async def write():
    for i in range(NUM_ITERS):
      await in_port.write(i)

  cocotb.start_soon(write())

  for i in range(NUM_ITERS):
    await out_port.cmd_read()

  for i in range(NUM_ITERS):
    data = await out_port.read()
    # print(f"{i:2}: 0x{int(data):016x}")
    assert data == i

  for i in range(4):
    await RisingEdge(dut.clk)
