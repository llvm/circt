#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess

import esiaccel
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test
import esiaccel.types as types
import pytest

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

  def test_auto_serial_coords(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(
        ["esitester", "cosim", conn, "auto_serial_coords", "-n", "5"])
    check_lines(stdout, [
        "Auto serial coord translate test results:",
        "coord[0]=",
        "Auto serial coord translate test passed",
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
    result = subprocess.run(
        ["esiquery", "cosim", conn, "telemetry"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    stdout = result.stdout
    check_lines(stdout, [
        "* Telemetry",
        "fromhostdma[32].fromHostChecksum: 0",
        "fromhostdma[32].fromHostCycles: 0",
        "readmem[32].addrCmdIssued: 0",
        "readmem[32].addrCmdResponses: 0",
        "readmem[32].lastReadLSB: 0",
        "readmem[32].readChecksum: 0",
        "tohostdma[32].toHostCycles: 0",
        "tohostdma[32].totalWrites: 0",
        "writemem[32].addrCmdIssued: 0",
        "writemem[32].addrCmdResponses: 0",
        "readmem[32].addrCmdResp.cycles: 0",
        "writemem[32].addrCmdResp.cycles: 0",
    ])

  def test_reset(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(["esitester", "cosim", conn, "reset"])
    check_lines(stdout, [
        "[reset] reset requested",
        "[reset] telemetry addrCmdResponses after reset = 0",
        "Reset test passed",
    ])

  def test_channel_python(self, conn: AcceleratorConnection) -> None:
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
    result = subprocess.run(
        ["esitester", "cosim", conn, "dma", "-w", "-r"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

  def test_bandwidth_data_integrity(self, host: str, port: int) -> None:
    """Verify complete engine payloads in both transfer directions."""
    conn = f"{host}:{port}"
    result = subprocess.run(
        [
            "esitester",
            "cosim",
            conn,
            "bandwidth",
            "-w",
            "-r",
            "--check-data",
            "--widths",
            "24",
            "64",
            "534",
            "--count",
            "24",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

  def test_hostmembw(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    # Widths chosen to cover every element-vs-engine-word relationship the read
    # gearbox must unpack from a contiguous, byte-packed layout: 24 (sub-word,
    # does not divide the 64-bit engine word), 32 (sub-word divisor), 72
    # (wider than the word, not a multiple -> straddles words), and 128 (a whole
    # number of words). A 1000-element 24-bit read burst is 3000 bytes, which
    # exceeds the PCIe maximum read request size and so exercises the
    # read-request splitter. The -w/-r data-integrity checks verify each element
    # landed at / was fetched from the right bytes.
    result = subprocess.run(
        [
            "esitester",
            "cosim",
            conn,
            "hostmembw",
            "-w",
            "-r",
            "-c",
            "1000",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "exceeds the PCIe maximum read request size" not in combined, \
        combined

  def test_aggbandwidth(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    # Aggregate bandwidth across the width-512 readmem*/writemem* units
    # (4 read + 4 write), exercising the multi-unit nested-AppID resolution
    # (mmio[width]/cmd and addrCmdResp/cycles) of the burst modules.
    result = subprocess.run(
        [
            "esitester", "cosim", conn, "aggbandwidth", "--width", "512", "-r",
            "-w", "-c", "64"
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

  def test_telemetry(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    result = subprocess.run(
        ["esiquery", "cosim", conn, "telemetry"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    stdout = result.stdout
    check_lines(stdout, [
        "* Telemetry",
        "fromhostdma[32].fromHostChecksum: 0",
        "fromhostdma[32].fromHostCycles: 0",
        "tohostdma[32].toHostCycles: 0",
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

  def test_serial_coords(self, host: str, port: int) -> None:
    conn = f"{host}:{port}"
    stdout = run_cmd(
        ["esitester", "cosim", conn, "serial_coords", "-n", "40", "-b", "33"])
    check_lines(stdout, [
        "Serial coord translate test results:",
        "coord[0]=",
        "Serial coord translate test passed",
    ])
