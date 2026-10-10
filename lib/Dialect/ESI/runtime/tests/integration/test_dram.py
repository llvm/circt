#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Cosim integration tests for the ESI DRAM service (`esi.Dram`).

The hardware (`hw/dram.py`) exposes host-callable functions per DRAM channel
which issue DRAM reads/writes through the cosim BSP's `ChannelDram` service
implementation, backed by the behavioral `EsiDramModel`. The host mirrors every
write into a byte-level reference model and checks all reads against it.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import esiaccel
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR

NUM_CHANNELS = 2  # Must match hw/dram.py.
ELEM24_MASK = (1 << 24) - 1


class RefMem:
  """Sparse byte-addressed reference memory. Unwritten bytes read as zero."""

  def __init__(self) -> None:
    self.bytes: Dict[int, int] = {}

  def write(self, addr: int, data: bytes, mask: Optional[int] = None) -> None:
    for i, b in enumerate(data):
      if mask is None or (mask >> i) & 1:
        self.bytes[addr + i] = b

  def read(self, addr: int, n: int) -> bytes:
    return bytes(self.bytes.get(addr + i, 0) for i in range(n))


class DramChannel:
  """Host-side driver for one DRAM channel's test functions."""

  def __init__(self, acc, ch: int) -> None:
    self.ch = ch
    self.mem = RefMem()
    ports = acc.children[esiaccel.AppID(f"dram{ch}")].ports
    self.funcs = {}
    for name in ("write64", "read64", "write128", "read128", "write24",
                 "read24", "read_list24", "write_list24"):
      f = ports[esiaccel.AppID(name)]
      f.connect()
      self.funcs[name] = f

  def write(self,
            width: int,
            addr: int,
            data: bytes,
            mask: Optional[int] = None):
    nbytes = width // 8
    assert len(data) == nbytes
    args = {"address": addr, "data": bytearray(data)}
    if width in (64, 128):
      # These functions take a byteenable mask; default to all bytes.
      be = (1 << nbytes) - 1 if mask is None else mask
      args["byteenable"] = bytearray(be.to_bytes((nbytes + 7) // 8, "little"))
    else:
      assert mask is None
    self.funcs[f"write{width}"].call(**args).result()
    self.mem.write(addr, data, mask)

  def read(self, width: int, addr: int) -> bytes:
    return bytes(self.funcs[f"read{width}"].call(addr).result())

  def check_read(self, width: int, addr: int) -> None:
    got = self.read(width, addr)
    exp = self.mem.read(addr, width // 8)
    assert got == exp, (f"ch{self.ch} read{width} @0x{addr:x}: got {got.hex()}"
                        f" expected {exp.hex()}")

  def write_list24(self, addr: int, count: int, seed: int) -> None:
    result = self.funcs["write_list24"].call(address=addr,
                                             count=count,
                                             seed=seed).result()
    assert result == count
    for i in range(count):
      value = (seed + i) & ELEM24_MASK
      self.mem.write(addr + 3 * i, value.to_bytes(3, "little"))

  def check_read_list24(self, addr: int, length: int) -> None:
    got = self.funcs["read_list24"].call(address=addr, length=length).result()
    xor = 0
    total = 0
    for i in range(length):
      v = int.from_bytes(self.mem.read(addr + 3 * i, 3), "little")
      xor ^= v
      total += v
    assert got["count"] == length, got
    assert int.from_bytes(bytes(got["xor"]), "little") == xor, got
    assert got["sum"] == total & 0xFFFFFFFF, got


def rand_bytes(rng: random.Random, n: int) -> bytes:
  return bytes(rng.getrandbits(8) for _ in range(n))


@cosim_test(HW_DIR / "dram.py", args=("{tmp_dir}", "cosim"))
class TestCosimDram:

  def test_single_rw(self, conn: AcceleratorConnection) -> None:
    """Aligned and unaligned 8- and 16-byte writes/reads (full masks)."""
    rng = random.Random(1)
    dram = DramChannel(conn.build_accelerator(), 0)
    # Unwritten memory reads as zero.
    dram.check_read(64, 0x40)
    addrs = [0x0, 0x8, 0x3, 0x15, 0x1007, 0x2000, 0x7_FFFF_FFF0]
    for addr in addrs:
      dram.write(64, addr, rand_bytes(rng, 8))
      dram.check_read(64, addr)
    for addr in [0x101, 0x200, 0x30F]:
      dram.write(128, addr, rand_bytes(rng, 16))
      dram.check_read(128, addr)
    # Re-read everything, including overlapping, unaligned windows.
    for addr in addrs + [0x1, 0x6, 0x1003, 0x100, 0x305]:
      dram.check_read(64, addr)
      dram.check_read(128, addr)
    # Near the top of the (32-bit word) address space.
    dram.check_read(64, 0x7_FFFF_FFF4)

  def test_byteenable(self, conn: AcceleratorConnection) -> None:
    """Sparse, byte-enabled writes leave masked-off bytes untouched."""
    rng = random.Random(2)
    dram = DramChannel(conn.build_accelerator(), 0)
    for addr in [0x400, 0x505]:
      dram.write(128, addr, rand_bytes(rng, 16))
      for mask in [0x0F, 0xA5, 0x00, 0x80, 0x01]:
        dram.write(64, addr + 3, rand_bytes(rng, 8), mask)
        dram.check_read(128, addr)
        dram.check_read(64, addr + 3)
      for mask in [0x0FF0, 0x8001, 0x5555]:
        dram.write(128, addr + 1, rand_bytes(rng, 16), mask)
        dram.check_read(128, addr)
        dram.check_read(128, addr + 1)

  def test_odd_size(self, conn: AcceleratorConnection) -> None:
    """3-byte elements at consecutive (mostly unaligned) addresses."""
    rng = random.Random(3)
    dram = DramChannel(conn.build_accelerator(), 0)
    base = 0x801
    for i in range(12):
      dram.write(24, base + 3 * i, rand_bytes(rng, 3))
    for i in range(12):
      dram.check_read(24, base + 3 * i)
    for off in range(0, 36, 5):
      dram.check_read(64, base + off)

  def test_lists(self, conn: AcceleratorConnection) -> None:
    """Burst (list) writes and reads of 3-byte elements, including bursts
    long enough to be split into multiple DRAM requests."""
    dram = DramChannel(conn.build_accelerator(), 0)
    dram.write_list24(0x3001, 37, 0x123456)
    dram.check_read_list24(0x3001, 37)
    dram.check_read_list24(0x3004, 10)
    for i in [0, 1, 17, 36]:
      dram.check_read(24, 0x3001 + 3 * i)
    dram.write_list24(0x8003, 300, 0xFFFF00)
    dram.check_read_list24(0x8003, 300)
    dram.check_read_list24(0x8000, 301)
    dram.check_read(64, 0x8003 + 3 * 299 - 5)

  def test_channels_independent(self, conn: AcceleratorConnection) -> None:
    """Each channel is a separate memory."""
    rng = random.Random(4)
    acc = conn.build_accelerator()
    drams = [DramChannel(acc, ch) for ch in range(NUM_CHANNELS)]
    for addr in [0x0, 0x13, 0x1000]:
      for dram in drams:
        dram.write(64, addr, rand_bytes(rng, 8))
    for addr in [0x0, 0x13, 0x1000, 0x10]:
      for dram in drams:
        dram.check_read(64, addr)
    drams[1].write_list24(0x51, 20, 0x10)
    for dram in drams:
      dram.check_read_list24(0x51, 20)

  def test_random(self, conn: AcceleratorConnection) -> None:
    """Randomized mix of reads, (masked) writes and list operations on both
    channels within a small address window, so operations overlap heavily."""
    rng = random.Random(5)
    acc = conn.build_accelerator()
    drams = [DramChannel(acc, ch) for ch in range(NUM_CHANNELS)]
    for _ in range(300):
      dram = rng.choice(drams)
      addr = rng.randrange(0, 256)
      op = rng.randrange(8)
      if op == 0:
        dram.write(64, addr, rand_bytes(rng, 8), rng.getrandbits(8))
      elif op == 1:
        dram.write(128, addr, rand_bytes(rng, 16), rng.getrandbits(16))
      elif op == 2:
        dram.write(24, addr, rand_bytes(rng, 3))
      elif op == 3:
        dram.write_list24(addr, rng.randrange(1, 12), rng.getrandbits(24))
      elif op == 4:
        dram.check_read_list24(addr, rng.randrange(1, 12))
      else:
        dram.check_read(rng.choice([24, 64, 128]), addr)
    for dram in drams:
      for addr in range(0, 300, 8):
        dram.check_read(64, addr)
