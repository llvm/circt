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
  assert manifest_loc == ChannelMMIO.ManifestSpace
  assert default.records == [{"offset": 0x100, "size": 0x100, "type": "ro"}]
  assert custom.records == [{"offset": 0x2000, "size": 0x2000, "type": "rw"}]


def test_channel_mmio_allocation_granularity(monkeypatch):
  monkeypatch.setattr(ChannelMMIO, "AllocationGranularity", 0x400)
  default = _Bundle("read")
  custom = _Bundle("read", {"size": 0x401})

  table, manifest_loc = ChannelMMIO.build_table(_Bundles(default, custom))

  assert list(table) == [0x400, 0x800]
  assert default.records[0]["size"] == 0x400
  assert custom.records[0]["size"] == 0x800
  assert manifest_loc == ChannelMMIO.ManifestSpace


def test_channel_mmio_places_manifest_after_clients(monkeypatch):
  monkeypatch.setattr(ChannelMMIO, "ManifestSpace", 0x1000)
  table, manifest_loc = ChannelMMIO.build_table(
      _Bundles(_Bundle("read"), _Bundle("read", {"size": 0x1203})))
  last_base = list(table)[-1]
  last_size = table[last_base][0]

  assert manifest_loc == 0x4000
  assert manifest_loc >= last_base + last_size
  assert manifest_loc % ChannelMMIO.ManifestSpace == 0


@pytest.mark.parametrize("granularity", [True, 0, 4, 12, 24])
def test_channel_mmio_rejects_invalid_granularity(monkeypatch, granularity):
  monkeypatch.setattr(ChannelMMIO, "AllocationGranularity", granularity)
  with pytest.raises(ValueError, match="allocation granularity"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read")))


@pytest.mark.parametrize("register_space", [True, 0, 4, 12, 24, 0x180])
def test_channel_mmio_rejects_invalid_register_space(monkeypatch,
                                                     register_space):
  monkeypatch.setattr(ChannelMMIO, "RegisterSpace", register_space)
  with pytest.raises(ValueError, match="register space"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read")))


@pytest.mark.parametrize("size", [True, 1.5, "256", 0, -8])
def test_channel_mmio_rejects_invalid_size(size):
  with pytest.raises(ValueError, match="option 'size'"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read", {"size": size})))


def test_channel_mmio_rejects_address_overflow(monkeypatch):
  monkeypatch.setattr(ChannelMMIO, "initial_offset", 0xFFFF_FFF8)
  with pytest.raises(ValueError, match="exceeds the 32-bit space"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read", {"size": 8})))
