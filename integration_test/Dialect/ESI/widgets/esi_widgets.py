from typing import List, Optional
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def init(dut, timeout: Optional[int] = None):
  # Create a 10ns period (100MHz) clock on port clock
  clk = Clock(dut.clk, 1, units="ns")
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


class ESIValidReadyInputPort(ESIInputPort):

  def __init__(self, dut, name):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.valid = getattr(dut, f"{name}_valid")
    self.ready = getattr(dut, f"{name}_ready")
    # Configure initial state
    self.valid.value = 0

  async def write(self, data: int):
    self.data.value = data
    self.valid.value = 1
    await RisingEdge(self.dut.clk)
    while self.ready.value == 0:
      await RisingEdge(self.dut.clk)
    self.valid.value = 0


class ESIOutputPort:

  def __init__(self, dut, name: str, latency: int = 0):
    self.dut = dut
    self.name = name
    self.data = getattr(dut, name)
    self.latency = latency
    self.q: List[int] = []

  async def read(self) -> Optional[int]:
    raise RuntimeError("Must be implemented by subclass")

  async def cmd_read(self):
    raise RuntimeError("Must be implemented by subclass")

  async def read_after_latency(self):
    for i in range(self.latency):
      await RisingEdge(self.dut.clk)
    self.q.append(self.data.value)


class ESIFifoOutputPort(ESIOutputPort):

  def __init__(self, dut, name: str, latency: int = 0):
    super().__init__(dut, name, latency)

    self.rden = getattr(dut, f"{name}_rden")
    self.empty = getattr(dut, f"{name}_empty")
    # Configure initial state
    self.rden.value = 0

  async def init_read(self):
    self.running = 1
    self.empty.value = 0
    while await RisingEdge(self.dut.clk):
      if self.rden.value == 1:
        cocotb.start_soon(self.read_after_latency())

  async def read(self) -> Optional[int]:
    while len(self.q) == 0:
      await RisingEdge(self.dut.clk)
    return self.q.pop(0)

  async def cmd_read(self):
    while self.empty.value == 1:
      await RisingEdge(self.dut.clk)
    self.rden.value = 1
    await RisingEdge(self.dut.clk)
    self.rden.value = 0
    cocotb.start_soon(self.read_after_latency())


class ESIValidReadyOutputPort(ESIOutputPort):

  def __init__(self, dut, name: str, latency: int = 0):
    super().__init__(dut, name, latency)

    self.valid = getattr(dut, f"{name}_valid")
    self.ready = getattr(dut, f"{name}_ready")
    # Configure initial state
    self.ready.value = 0

  async def read(self) -> Optional[int]:
    while len(self.q) == 0:
      await RisingEdge(self.dut.clk)
    return self.q.pop(0)

  async def cmd_read(self):
    self.ready.value = 1
    await RisingEdge(self.dut.clk)
    while self.valid.value == 0:
      await RisingEdge(self.dut.clk)
    self.ready.value = 0
    cocotb.start_soon(self.read_after_latency())


async def runFillAndDrain(dut, in_port, out_port):
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


async def runBackToBack(dut, in_port, out_port):
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


@cocotb.test()
async def fillAndDrainFIFO(dut):
  in_port = ESIFifoInputPort(dut, "fifo1_in")
  out_port = ESIFifoOutputPort(dut, "fifo1_out", 2)
  await runFillAndDrain(dut, in_port, out_port)


@cocotb.test()
async def backToBack(dut):
  in_port = ESIFifoInputPort(dut, "fifo1_in")
  out_port = ESIFifoOutputPort(dut, "fifo1_out", 2)
  await runBackToBack(dut, in_port, out_port)


@cocotb.test()
async def fillAndDrainValidReadyInputFIFO(dut):
  in_port = ESIValidReadyInputPort(dut, "fifoValidReadyInput_in")
  out_port = ESIFifoOutputPort(dut, "fifoValidReadyInput_out", 2)
  await runFillAndDrain(dut, in_port, out_port)


@cocotb.test()
async def backToBackValidReadyInputFIFO(dut):
  in_port = ESIValidReadyInputPort(dut, "fifoValidReadyInput_in")
  out_port = ESIFifoOutputPort(dut, "fifoValidReadyInput_out", 2)
  await runBackToBack(dut, in_port, out_port)


@cocotb.test()
async def fillAndDrainValidReadyOutputFIFO(dut):
  in_port = ESIFifoInputPort(dut, "fifoValidReadyOutput_in")
  out_port = ESIValidReadyOutputPort(dut, "fifoValidReadyOutput_out", 0)
  await runFillAndDrain(dut, in_port, out_port)


@cocotb.test()
async def backToBackValidReadyOutputFIFO(dut):
  in_port = ESIFifoInputPort(dut, "fifoValidReadyOutput_in")
  out_port = ESIValidReadyOutputPort(dut, "fifoValidReadyOutput_out", 0)
  await runBackToBack(dut, in_port, out_port)
