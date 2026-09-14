#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from esiaccel.bsp.common import ChannelMMIO


class _Bundle:

  def __init__(self, port, options=None):
    self.port = port
    self.options = options or {}
    self.records = []

  def add_record(self, details):
    self.records.append(details)


class _Bundles:

  def __init__(self, *bundles):
    self.to_client_reqs = bundles


def test_channel_mmio_allocations():
  default = _Bundle("read")
  custom = _Bundle("read_write", {"size": 0x1203})

  table, manifest_loc = ChannelMMIO.build_table(_Bundles(default, custom))

  assert list(table) == [0x100, 0x2000]
  assert default.records == [{"offset": 0x100, "size": 0x100, "type": "ro"}]
  assert custom.records == [{"offset": 0x2000, "size": 0x2000, "type": "rw"}]
  # The manifest lands directly above the last client, rounded to a power of
  # two so a single prefix test separates it from the client space.
  assert manifest_loc == 0x4000


def test_channel_mmio_manifest_tracks_client_space():
  """The manifest window must not be a fixed reservation: it follows whatever
  the clients actually use so the BAR stays as small as possible."""

  small, = _Bundles(_Bundle("read", {"size": 8})).to_client_reqs
  _, small_loc = ChannelMMIO.build_table(_Bundles(small))
  assert small_loc == 0x200

  big = _Bundle("read", {"size": 1 << 20})
  _, big_loc = ChannelMMIO.build_table(_Bundles(big))
  assert big_loc == 1 << 21


def test_channel_mmio_places_manifest_after_clients():
  table, manifest_loc = ChannelMMIO.build_table(
      _Bundles(_Bundle("read"), _Bundle("read", {"size": 0x1203})))
  last_base = list(table)[-1]
  last_size = table[last_base][0]

  assert manifest_loc >= last_base + last_size
  # A power-of-two base means 'address has a bit set above the clients' is the
  # whole decode: no comparator, and the region can't overflow.
  assert manifest_loc & (manifest_loc - 1) == 0


@pytest.mark.parametrize("size", [True, 1.5, "256", 0, -8])
def test_channel_mmio_rejects_invalid_size(size):
  with pytest.raises(ValueError, match="option 'size'"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read", {"size": size})))


def test_channel_mmio_rejects_address_overflow(monkeypatch):
  monkeypatch.setattr(ChannelMMIO, "initial_offset", 0xFFFF_FFF8)
  with pytest.raises(ValueError, match="exceeds the 32-bit space"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read", {"size": 8})))
