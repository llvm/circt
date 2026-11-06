#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Cosim tests for configurable widths and multiple outstanding HostMem reads.

`hw/hostmem_read.py` builds one host-driven `read_list` client per element
width under `CosimBSP(max_outstanding_reads=N)`. Each test closes the client's
response gate, queues several reads of varied lengths, checks that exactly
min(#reads, N) are accepted while nothing drains (so that many logical requests
are genuinely in flight inside `HostMemReadReqSplitter`), then opens the gate
and checks every returned element, the per-request element counts and that
`last` lands exactly on each request's final element.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import List, Tuple

import esiaccel
import pytest
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR

# Must match hw/hostmem_read.py.
ELEMENT_WIDTHS = [24, 32, 96]

# (element offset, element count). The first read is long enough that the
# read path's buffering cannot absorb its whole burst, so nothing behind it can
# complete while the gate is closed. Lengths cover single elements, counts
# that leave a partial final 64-bit word, and reads spanning several 256-byte
# upstream chunks.
GATED_READS: List[Tuple[int, int]] = [
    (0, 120),
    (5, 1),
    (17, 3),
    (40, 7),
    (3, 90),
    (300, 2),
    (11, 33),
    (64, 5),
]
# Short reads streamed with the gate open, so requests are accepted while
# earlier bursts are still draining.
STREAMED_READS: List[Tuple[int,
                           int]] = [(i * 7 % 50, 1 + i % 3) for i in range(24)]

BUFFER_ELEMENTS = 4096


@pytest.fixture(autouse=True)
def read_frame_log(monkeypatch):
  monkeypatch.setenv("COSIM_DEBUG_FILE", "hostmem_read.log")


def _pattern(num_bytes: int) -> bytes:
  return bytes((i * 37 + 11) & 0xFF for i in range(num_bytes))


def _as_int(v) -> int:
  if isinstance(v, (bytes, bytearray)):
    return int.from_bytes(v, "little")
  return int(v)


class _Dut:

  def __init__(self, conn: AcceleratorConnection, acc, width: int):
    self.width = width
    self.stride = (width + 7) // 8
    dut = acc.children[esiaccel.AppID("rd", width)]
    self.req = dut.ports[esiaccel.AppID("req")]
    self.gate = dut.ports[esiaccel.AppID("gate")]
    self.resp = dut.ports[esiaccel.AppID("resp")]
    self.accepted = dut.ports[esiaccel.AppID("accepted")]
    for p in (self.req, self.gate, self.resp, self.accepted):
      p.connect()

    hostmem = conn.get_service_hostmem()
    hostmem.start()
    nbytes = BUFFER_ELEMENTS * self.stride + 64
    self.mem = hostmem.allocate(nbytes)
    self.data = _pattern(nbytes)
    ctypes.memmove(self.mem.ptr, self.data, nbytes)

  def expected(self, offset: int) -> int:
    start = offset * self.stride
    raw = int.from_bytes(self.data[start:start + self.stride], "little")
    return raw & ((1 << self.width) - 1)

  def accepted_count(self) -> int:
    return self.accepted.cpp_port.readInt()

  def issue(self, reads: List[Tuple[int, int]]) -> None:
    for offset, count in reads:
      self.req.write({
          "address": self.mem.ptr + offset * self.stride,
          "length": count
      })

  def check_responses(self, reads: List[Tuple[int, int]]) -> None:
    for ridx, (offset, count) in enumerate(reads):
      for e in range(count):
        msg = self.resp.read().result()
        got = _as_int(msg["data"])
        last = _as_int(msg["last"])
        exp = self.expected(offset + e)
        assert got == exp, (
            f"width {self.width} read {ridx} element {e}: got {got:#x}, "
            f"expected {exp:#x}")
        assert last == (e == count - 1), (
            f"width {self.width} read {ridx} (count {count}) element {e}: "
            f"last={last}")


def _wait_for_accepted(dut: _Dut, target: int, timeout_s: float = 30) -> int:
  deadline = time.time() + timeout_s
  count = dut.accepted_count()
  while count < target and time.time() < deadline:
    time.sleep(0.05)
    count = dut.accepted_count()
  return count


def _check_frames(expected_reads, bus_width):
  """Check actual transport requests and responses, including zero padding."""
  lines = Path("hostmem_read.log").read_text().splitlines()
  requests = [
      bytes.fromhex(line.split("]", 1)[1])
      for line in lines
      if "__cosim_hostmem_read_req.data" in line
  ]
  responses = [
      bytes.fromhex(line.split("]", 1)[1])
      for line in lines
      if "__cosim_hostmem_read_resp.data" in line
  ]
  word_bytes = bus_width // 8
  cap = max(256, word_bytes)
  chunks = [(address + offset, data[offset:offset + cap])
            for address, data in expected_reads
            for offset in range(0, len(data), cap)]
  assert len(requests) == len(chunks)
  assert len(responses) == sum(
      (len(data) + word_bytes - 1) // word_bytes for _, data in expected_reads)
  response_idx = 0
  for request, (address, data) in zip(requests, chunks):
    assert len(request) == 13
    assert int.from_bytes(request[1:5], "little") == len(data)
    assert int.from_bytes(request[5:], "little") == address
    for offset in range(0, len(data), word_bytes):
      response = responses[response_idx]
      response_idx += 1
      assert len(response) == word_bytes + 2
      packed = int.from_bytes(response, "little")
      assert packed & 1 == int(offset + word_bytes >= len(data))
      assert packed >> (bus_width + 1) == request[0]
      assert (packed >> 1) & ((1 << bus_width) - 1) == int.from_bytes(
          data[offset:offset + word_bytes], "little")


def _run(conn: AcceleratorConnection,
         max_outstanding: int,
         bus_width: int = 64) -> None:
  hostmem = conn.get_service_hostmem()
  hostmem.start()
  acc = conn.build_accelerator()
  expected_reads = []
  for width in ELEMENT_WIDTHS:
    dut = _Dut(conn, acc, width)
    # Keep the first burst larger than the response buffering even on wide
    # buses, so it cannot finish while the client response gate is closed.
    gated_reads = [(0, max(120, 4 * (bus_width // 8) // dut.stride + 1))
                  ] + GATED_READS[1:]

    # Gate closed (its reset state): queue every read, then confirm exactly
    # min(#reads, max_outstanding) of them are accepted and held in flight.
    dut.gate.write(0)
    dut.issue(gated_reads)
    in_flight = min(len(gated_reads), max_outstanding)
    assert _wait_for_accepted(dut, in_flight) >= in_flight
    time.sleep(0.5)
    assert dut.accepted_count() == in_flight, (
        f"width {width}: expected {in_flight} reads in flight with the "
        f"response side stalled")

    dut.gate.write(1)
    dut.check_responses(gated_reads)
    assert dut.accepted_count() == len(gated_reads)

    # Gate open: stream short reads back to back.
    dut.issue(STREAMED_READS)
    dut.check_responses(STREAMED_READS)
    assert dut.accepted_count() == len(gated_reads) + len(STREAMED_READS)
    for offset, count in gated_reads + STREAMED_READS:
      start = offset * dut.stride
      expected_reads.append(
          (dut.mem.ptr + start, dut.data[start:start + count * dut.stride]))

  for width in [13, bus_width, bus_width + 24, 16384]:
    dut = acc.children[esiaccel.AppID("single", width)]
    req = dut.ports[esiaccel.AppID("req")]
    resp = dut.ports[esiaccel.AppID("resp")]
    req.connect()
    resp.connect()
    size = (width + 7) // 8
    mem = hostmem.allocate(size + bus_width // 8 + 6)
    ctypes.memset(mem.ptr, 0xA5, mem.size)
    data = _pattern(size)
    ctypes.memmove(mem.ptr + 3, data, size)
    req.write(mem.ptr + 3)
    assert resp.read().result(timeout=30) == (int.from_bytes(data, "little") &
                                              ((1 << width) - 1))
    expected_reads.append((mem.ptr + 3, data))
  _check_frames(expected_reads, bus_width)


@cosim_test(HW_DIR / "hostmem_read.py", args=("{tmp_dir}", "1"))
class TestHostMemReadOneOutstanding:

  def test_reads(self, conn: AcceleratorConnection) -> None:
    _run(conn, 1)


@cosim_test(HW_DIR / "hostmem_read.py", args=("{tmp_dir}", "4"))
class TestHostMemReadFourOutstanding:

  def test_reads(self, conn: AcceleratorConnection) -> None:
    _run(conn, 4)


@cosim_test(HW_DIR / "hostmem_read.py", args=("{tmp_dir}", "4", "8"))
class TestHostMemRead8:

  def test_reads(self, conn: AcceleratorConnection) -> None:
    _run(conn, 4, 8)


@cosim_test(HW_DIR / "hostmem_read.py", args=("{tmp_dir}", "4", "512"))
class TestHostMemRead512:

  def test_reads(self, conn: AcceleratorConnection) -> None:
    _run(conn, 4, 512)


@cosim_test(HW_DIR / "hostmem_read.py", args=("{tmp_dir}", "4", "2048"))
class TestHostMemRead2048:

  def test_reads(self, conn: AcceleratorConnection) -> None:
    _run(conn, 4, 2048)


@cosim_test(HW_DIR / "hostmem_read.py", args=("{tmp_dir}", "4", "4096"))
class TestHostMemRead4096:

  def test_reads(self, conn: AcceleratorConnection) -> None:
    _run(conn, 4, 4096)
