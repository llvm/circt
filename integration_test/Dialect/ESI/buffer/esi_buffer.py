import random
from typing import List

import cocotb
from cocotb.triggers import RisingEdge

from esi_widgets import ESIValidReadyInputPort, ESIValidReadyOutputPort, init


async def run_stream(dut, in_name: str, out_name: str, n: int,
                     backpressure: float):
  in_port = ESIValidReadyInputPort(dut, in_name)
  out_port = ESIValidReadyOutputPort(dut, out_name, backpressure=backpressure)
  await init(dut, timeout=100000)

  values = [random.randint(0, 0xFFFFFFFF) for _ in range(n)]
  received: List[int] = []

  async def producer():
    for v in values:
      await in_port.write(v)

  cocotb.start_soon(producer())

  for _ in range(n):
    await out_port.cmd_read()
  for _ in range(n):
    received.append(int(await out_port.read()))

  assert received == values, \
      f"data mismatch on {out_name}: got {received[:8]}... expected {values[:8]}..."

  for _ in range(4):
    await RisingEdge(dut.clk)


# Single-stage buffer, no backpressure (streaming throughput).
@cocotb.test()
async def s1_stream(dut):
  await run_stream(dut, "s1_in", "s1_out", 200, backpressure=0.0)


# Single-stage buffer with heavy consumer backpressure.
@cocotb.test()
async def s1_backpressure(dut):
  await run_stream(dut, "s1_in", "s1_out", 100, backpressure=0.6)


# Multi-stage buffer, no backpressure.
@cocotb.test()
async def s4_stream(dut):
  await run_stream(dut, "s4_in", "s4_out", 200, backpressure=0.0)


# Multi-stage buffer with heavy consumer backpressure, exercising the deeper
# run-out FIFO and feed-forward register chain stalling.
@cocotb.test()
async def s4_backpressure(dut):
  await run_stream(dut, "s4_in", "s4_out", 100, backpressure=0.6)
