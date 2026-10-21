#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Cosim integration tests for channel backpressure (cosim protocol v4 flow
control), using the DUTs in `hw/backpressure.py`.

  * Host -> simulation: with the loopback's gate closed the hardware never
    consumes `in`, so `try_write` must start failing after roughly the
    default in-flight limit (16) plus the endpoint's internal buffering, and
    recover once the gate opens.

  * Simulation -> host: with `stream` connected with a small buffer and not
    read, the producer must stall (its `Cosim_Endpoint_ToHost` deasserts
    ready) instead of pushing everything into host memory.
"""

from __future__ import annotations

import time

import esiaccel
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR

# Must match RpcClient::DefaultToServerMaxInFlight.
DEFAULT_TO_SERVER_MAX_IN_FLIGHT = 16
# Slack for messages absorbed by hardware buffering (cosim endpoints, BSP
# pipeline stages) beyond the cosim flow-control in-flight limits.
HW_SLACK = 8


def _fill(try_write, start: int) -> int:
  """Call `try_write(value)` with increasing values from `start` until it keeps
  failing for a while, so credits for anything the hardware takes in the
  meantime are observed. Returns the next unwritten value."""
  value = start
  quiet_since = time.monotonic()
  while time.monotonic() - quiet_since < 1.0:
    if try_write(value):
      value += 1
      quiet_since = time.monotonic()
      assert value - start < 1000, "try_write never applied backpressure"
    else:
      time.sleep(0.01)
  return value


def _wait_stable(read_value, settle: float = 1.0, timeout: float = 60.0):
  """Poll `read_value()` until it stops changing for `settle` seconds."""
  deadline = time.monotonic() + timeout
  last = read_value()
  last_change = time.monotonic()
  while time.monotonic() - last_change < settle:
    assert time.monotonic() < deadline, "value never settled"
    time.sleep(0.1)
    cur = read_value()
    if cur != last:
      last, last_change = cur, time.monotonic()
  return last


@cosim_test(HW_DIR / "backpressure.py")
class TestBackpressure:

  def test_host_to_sim(self, conn: AcceleratorConnection) -> None:
    acc = conn.build_accelerator()
    dut = acc.children[esiaccel.AppID("loopback")]
    gate = dut.ports[esiaccel.AppID("gate")]
    gate_connected = gate.connect()
    inp = dut.ports[esiaccel.AppID("in")].write_port("data").connect()
    # Unbounded, so the loopback output never stalls the input.
    out = dut.ports[esiaccel.AppID("out")].read_port("data").connect(
        buffer_size=0)
    # connect() doesn't block, but the futures complete once the simulator
    # side (connected at startup) acknowledges.
    gate_connected.result(timeout=30)
    inp.connected_future.result(timeout=30)
    out.connected_future.result(timeout=30)
    assert gate_connected.done()

    # Gate closed (reset state): writes must back up.
    accepted = _fill(inp.try_write, 0)
    assert DEFAULT_TO_SERVER_MAX_IN_FLIGHT <= accepted <= (
        DEFAULT_TO_SERVER_MAX_IN_FLIGHT + HW_SLACK), accepted
    assert not inp.try_write(accepted)

    # Open the gate: everything accepted so far comes out, in order.
    gate.write(1)
    for i in range(accepted):
      assert out.read() == i

    # With the consumer running, blocking writes flow despite the limit.
    total = accepted + 200
    for i in range(accepted, total):
      inp.write(i)
    for i in range(accepted, total):
      assert out.read() == i

    # Close the gate again and check try_write recovers once it reopens.
    gate.write(0)
    stalled_at = _fill(inp.try_write, total)
    assert stalled_at - total <= DEFAULT_TO_SERVER_MAX_IN_FLIGHT + HW_SLACK
    gate.write(1)
    for i in range(total, stalled_at):
      assert out.read() == i
    deadline = time.monotonic() + 30
    while not inp.try_write(stalled_at):
      assert time.monotonic() < deadline, "try_write never recovered"
      time.sleep(0.01)
    assert out.read() == stalled_at

  def test_sim_to_host(self, conn: AcceleratorConnection) -> None:
    acc = conn.build_accelerator()
    dut = acc.children[esiaccel.AppID("producer")]
    produce = dut.ports[esiaccel.AppID("produce")]
    query = dut.ports[esiaccel.AppID("query")]
    sent = dut.ports[esiaccel.AppID("sent")]
    for p in (produce, query, sent):
      p.connect()
    # Small buffer: both the host polling queue and the in-flight limit.
    buffer = 4
    stream = dut.ports[esiaccel.AppID("stream")].read_port("data").connect(
        buffer_size=buffer)

    def sent_count() -> int:
      query.write(0)
      return sent.read().result()

    n = 1000
    produce.write(n)
    stalled = _wait_stable(sent_count)
    # Host polling queue + in-flight limit + hardware buffering.
    assert 2 * buffer <= stalled <= 2 * buffer + HW_SLACK, stalled

    # Draining the stream lets the producer finish, with nothing lost.
    for i in range(n):
      assert stream.read() == i
    assert sent_count() == n
