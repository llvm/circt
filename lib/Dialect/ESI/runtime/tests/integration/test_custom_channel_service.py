#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
import json
from pathlib import Path

import esiaccel
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR


@cosim_test(HW_DIR / "custom_channel_service.py")
def test_custom_channel_service(conn: AcceleratorConnection,
                                sources_dir: Path) -> None:
  """Route a custom service's multiple MMIO and HostMem clients via the BSP."""
  manifest = json.loads((sources_dir / "esi_system_manifest.json").read_text())
  engine_names = {"OneItemBuffersFromHost", "OneItemBuffersToHost"}
  engines = [
      engine for engine in manifest["design"]["engines"]
      if engine["serviceImplName"] in engine_names
  ]
  assert Counter(engine["serviceImplName"] for engine in engines) == {
      "OneItemBuffersFromHost": 2,
      "OneItemBuffersToHost": 2,
  }
  expected_clients = {
      f"loopback_{direction}_{i}" for direction in ("in", "out")
      for i in range(2)
  }
  assert {
      tuple(appid["name"]
            for appid in client["relAppIDPath"])
      for engine in engines
      for client in engine["clientDetails"]
  } == {(name,) for name in expected_clients}
  assert {engine["mmio"]["name"] for engine in engines
         } == {f"custom_{name}.mmio" for name in expected_clients}

  acc = conn.build_accelerator()
  inputs = [acc.ports[esiaccel.AppID(f"loopback_in_{i}")] for i in range(2)]
  outputs = [acc.ports[esiaccel.AppID(f"loopback_out_{i}")] for i in range(2)]
  for port in inputs + outputs:
    port.connect()

  # Distinct data in every 64-bit word detects routing and gearbox errors.
  for iteration in range(8):
    values = [
        sum((1 + iteration + i * 16 + word * 256) << (64 * word)
            for word in range(4))
        for i in range(2)
    ]
    reads = [port.read() for port in outputs]
    for port, value in zip(inputs, values):
      port.write(value)
    assert [read.result(timeout=10) for read in reads] == values
