#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import json

import esiaccel
import pytest
from esiaccel import esiCppAccel as cpp
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR


def _int(value):
  return int.from_bytes(value, "little") if isinstance(
      value, (bytes, bytearray)) else int(value)


def _run(conn: AcceleratorConnection, bus_width: int):
  # Exercise discovery without a caller-populated manifest type table.
  hostmem = conn.get_service_hostmem()
  hostmem.start()
  acc = conn.build_accelerator()
  allocations = []
  writers = []
  for width in [13, bus_width, bus_width + 24, bus_width * 2 + 24, 16384]:
    dut = acc.children[esiaccel.AppID("wr", width)]
    req, ack, gate = [
        dut.ports[esiaccel.AppID(name)] for name in ("req", "ack", "gate")
    ]
    for port in (req, ack, gate):
      port.connect()
    size = (width + 7) // 8
    expected = []
    for i in range(3):
      mem = hostmem.allocate(size + 16)
      allocations.append(mem)
      ctypes.memset(mem.ptr, 0xA5, size + 16)
      data = bytes((j * 37 + 11 + i) & 0xFF for j in range(size))
      value = int.from_bytes(data, "little") & ((1 << width) - 1)
      data = value.to_bytes(size, "little")
      req.write({"address": mem.ptr + 3, "tag": 0, "data": data})
      expected.append((mem, data))
    writers.append((ack, gate, expected))

  # Release completions after issuing all clients' writes. Open every gate
  # before waiting on any ack: the clients share one in-order response stream,
  # so a response for a still-gated client would block those behind it.
  for _, gate, _ in writers:
    gate.write(1)
  for ack, _, expected in writers:
    for mem, data in expected:
      ack.read().result(timeout=30)
      assert ctypes.string_at(mem.ptr,
                              len(data) + 16) == (b"\xA5" * 3 + data +
                                                  b"\xA5" * 13)

  req = acc.ports[esiaccel.AppID("list_req")]
  ack = acc.ports[esiaccel.AppID("list_ack")]
  req.connect()
  ack.connect()
  for count in [1, 7, 100]:
    mem = hostmem.allocate(count * 3 + 16)
    allocations.append(mem)
    ctypes.memset(mem.ptr, 0xA5, count * 3 + 16)
    data = bytes((i * 29 + 7) & 0xFF for i in range(count * 3))
    for i in range(count):
      req.write({
          "address": mem.ptr + 3,
          "data": data[i * 3:i * 3 + 3],
          "last": int(i == count - 1),
      })
    for _ in range(count):
      ack.read().result(timeout=30)
    assert ctypes.string_at(mem.ptr, count * 3 + 16) == (b"\xA5" * 3 + data +
                                                         b"\xA5" * 13)

  req = acc.ports[esiaccel.AppID("probe_req")]
  resp = acc.ports[esiaccel.AppID("probe_resp")]
  req.connect()
  resp.connect()
  data = bytes((i * 13 + 3) & 0xFF for i in range(2048))
  req.write({"address": 19, "tag": 17, "data": data})
  beat_bytes = bus_width // 8
  cap = max(256, beat_bytes)
  for offset in range(0, len(data), beat_bytes):
    beat = resp.read().result(timeout=30)
    assert _int(beat["address"]) == 19 + offset
    assert _int(beat["tag"]) == 17
    assert _int(beat["valid_bytes"]) == beat_bytes
    assert _int(beat["data"]) == int.from_bytes(
        data[offset:offset + beat_bytes], "little")
    assert _int(beat["last"]) == int((offset + beat_bytes) % cap == 0)


@cosim_test(HW_DIR / "hostmem_write.py", args=("{tmp_dir}",))
class TestHostMemWriteDefault:

  def test_writes(self, conn: AcceleratorConnection):
    _run(conn, 64)

  @pytest.mark.parametrize("invalid_layout", [
      "width",
      "field_name",
      "missing_data",
      "missing_field",
      "extra_field",
      "field_order",
      "data_type",
  ])
  @pytest.mark.parametrize("direction", ["read", "write"])
  def test_rejects_invalid_layout(self, host, port, invalid_layout, direction):
    # Do not reuse cached valid types from earlier tests in a Windows worker.
    conn = esiaccel.Context().connect("cosim", f"{host}:{port}")
    manifest = json.loads(conn.sysinfo().json_manifest())
    names = (["tag", "data", "last"] if direction == "read" else
             ["address", "tag", "data", "data_size", "last"])
    layout = next(
        channel["type"]["inner"]
        for t in manifest["types"]
        for channel in t.get("channels", [])
        if
        [f["name"]
         for f in channel["type"].get("inner", {}).get("fields", [])] == names)
    fields = layout["fields"]
    data = next(field for field in fields if field["name"] == "data")
    message = f"Invalid HostMem {direction} field"
    if invalid_layout == "width":
      data["type"].update(id="i24", hwBitwidth=24)
      message = f"{direction} width must be a power of two"
    elif invalid_layout == "field_name":
      fields[0]["name"] = "invalid_address"
    elif invalid_layout == "missing_data":
      fields.remove(data)
      message += ": data"
    elif invalid_layout == "missing_field":
      fields.pop()
      message = f"Invalid HostMem {direction} channel layout"
    elif invalid_layout == "extra_field":
      fields.append({"name": "extra", "type": fields[0]["type"]})
      message = f"Invalid HostMem {direction} channel layout"
    elif invalid_layout == "field_order":
      fields.reverse()
    elif invalid_layout == "data_type":
      data["type"] = {
          "id": "invalid_data_struct",
          "mnemonic": "struct",
          "fields": [{
              "name": "bits",
              "type": data["type"]
          }],
      }
    cpp.Manifest(conn.ctxt.cpp_ctxt, json.dumps(manifest))
    with pytest.raises(RuntimeError, match=message):
      conn.get_service_hostmem().start()


@cosim_test(HW_DIR / "hostmem_write.py", args=("{tmp_dir}", "8"))
class TestHostMemWrite8:

  def test_writes(self, conn: AcceleratorConnection):
    _run(conn, 8)


@cosim_test(HW_DIR / "hostmem_write.py", args=("{tmp_dir}", "512"))
class TestHostMemWrite512:

  def test_writes(self, conn: AcceleratorConnection):
    _run(conn, 512)


@cosim_test(HW_DIR / "hostmem_write.py", args=("{tmp_dir}", "2048"))
class TestHostMemWrite2048:

  def test_writes(self, conn: AcceleratorConnection):
    _run(conn, 2048)


@cosim_test(HW_DIR / "hostmem_write.py", args=("{tmp_dir}", "4096"))
class TestHostMemWrite4096:

  def test_writes(self, conn: AcceleratorConnection):
    _run(conn, 4096)
