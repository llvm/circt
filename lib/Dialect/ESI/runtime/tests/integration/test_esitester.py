#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import esiaccel
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test
import esiaccel.types as types

from .conftest import HW_DIR, check_lines, require_tool, run_cmd


@cosim_test(HW_DIR / "esitester.py", args=("{tmp_dir}", "cosim"))
class TestCosimEsitester:

  def setup_method(self) -> None:
    require_tool("esitester")
    require_tool("esiquery")

  def test_callback(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esitester", "-v", "cosim", conn, "callback", "-i", "5"])
    check_lines(stdout, [
        "[CONNECT] connecting to backend",
    ])
    # The callback loop should print values 0 through 4.
    for i in range(5):
      assert f"callback: {i}" in stdout, \
          f"Expected 'callback: {i}' in stdout"

  def test_streaming_add(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esitester", "cosim", conn, "streaming_add"])
    check_lines(stdout, [
        "Streaming add test results:",
        "input[0]=222709 + 5 = 222714 (expected 222714)",
        "input[1]=894611 + 5 = 894616 (expected 894616)",
        "input[2]=772894 + 5 = 772899 (expected 772899)",
        "input[3]=429150 + 5 = 429155 (expected 429155)",
        "input[4]=629806 + 5 = 629811 (expected 629811)",
        "Streaming add test passed",
    ])

  def test_streaming_add_quiet(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esitester", "cosim", conn, "streaming_add", "-t"])
    check_lines(stdout, [
        "Streaming add test results:",
        "Streaming add test passed",
    ])

  def test_translate_coords(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esitester", "cosim", conn, "translate_coords"])
    check_lines(stdout, [
        "Coord translate test results:",
        "coord[0]=(222709,894611) + (10,20) = (222719,894631)",
        "coord[1]=(772894,429150) + (10,20) = (772904,429170)",
        "coord[2]=(629806,138727) + (10,20) = (629816,138747)",
        "coord[3]=(218516,390276) + (10,20) = (218526,390296)",
        "coord[4]=(750021,423525) + (10,20) = (750031,423545)",
        "Coord translate test passed",
    ])

  def test_serial_coords(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(
        ["esitester", "cosim", conn, "serial_coords", "-n", "40", "-b", "33"])
    check_lines(stdout, [
        "Serial coord translate test results:",
        "coord[0]=",
        "Serial coord translate test passed",
    ])

  def test_channel(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esitester", "cosim", conn, "channel", "-i", "3"])
    check_lines(stdout, [
        "[channel] producer i=0 got=0",
        "[channel] producer i=1 got=1",
        "[channel] producer i=2 got=2",
        "[channel] loopback i=0",
        "[channel] loopback i=1",
        "[channel] loopback i=2",
        "Channel test passed",
    ])

  def test_telemetry(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esiquery", "cosim", conn, "telemetry"])
    check_lines(stdout, [
        "* Telemetry",
        "fromhostdma[32].fromHostCycles: 0",
        "readmem[32].addrCmdCycles: 0",
        "readmem[32].addrCmdIssued: 0",
        "readmem[32].addrCmdResponses: 0",
        "readmem[32].lastReadLSB: 0",
        "tohostdma[32].toHostCycles: 0",
        "tohostdma[32].totalWrites: 0",
        "writemem[32].addrCmdCycles: 0",
        "writemem[32].addrCmdIssued: 0",
        "writemem[32].addrCmdResponses: 0",
    ])


@cosim_test(HW_DIR / "esitester.py", args=("{tmp_dir}", "cosim_dma"))
class TestCosimEsitesterDma:

  def setup_method(self) -> None:
    require_tool("esitester")
    require_tool("esiquery")

  def test_hostmem(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    run_cmd(["esitester", "cosim", conn, "hostmem"])

  def test_dma(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    run_cmd(["esitester", "cosim", conn, "dma", "-w", "-r"])

  def test_telemetry(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esiquery", "cosim", conn, "telemetry"])
    check_lines(stdout, [
        "* Telemetry",
        "fromhostdma[32].fromHostCycles: 0",
        "tohostdma[32].toHostCycles: 0",
    ])


@cosim_test(HW_DIR / "esitester.py", args=("{tmp_dir}", "cosim"))
def test_channel_python(conn: AcceleratorConnection) -> None:
  """Test ChannelService ToHost and FromHost ports from Python."""
  acc = conn.build_accelerator()
  channel_test = acc.children[esiaccel.AppID("channel_test")]
  ports = channel_test.ports

  # Get the MMIO port and trigger the producer to send 5 values.
  mmio = ports[esiaccel.AppID("cmd")]
  assert isinstance(mmio, types.MMIORegion), \
      f"Expected MMIORegion, got {type(mmio)}"

  producer = ports[esiaccel.AppID("producer")]
  assert isinstance(producer, types.ToHostPort), \
      f"Expected ToHostPort, got {type(producer)}"
  producer.connect()

  num_values = 5
  mmio.write(0x0, num_values)
  for i in range(num_values):
    result = producer.read().result()
    assert result == i, f"Producer: expected {i}, got {result}"

  # Test from_host -> to_host loopback.
  loopback_in = ports[esiaccel.AppID("loopback_in")]
  assert isinstance(loopback_in, types.FromHostPort), \
      f"Expected FromHostPort, got {type(loopback_in)}"
  loopback_in.connect()

  loopback_out = ports[esiaccel.AppID("loopback_out")]
  assert isinstance(loopback_out, types.ToHostPort), \
      f"Expected ToHostPort, got {type(loopback_out)}"
  loopback_out.connect()

  for i in range(5):
    loopback_in.write(42 + i)
    result = loopback_out.read().result()
    assert result == 42 + i, \
        f"Loopback: expected {42 + i}, got {result}"
