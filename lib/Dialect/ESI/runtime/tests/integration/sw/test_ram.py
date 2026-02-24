from __future__ import annotations

from pathlib import Path
import random
import sys
import time
from typing import cast

import esiaccel as esi
from esiaccel.accelerator import AcceleratorConnection
from esiaccel.cosim.pytest import cosim_test

HW_DIR = Path(__file__).resolve().parent.parent / "hw"


def run(conn: AcceleratorConnection) -> None:
  d = conn.build_accelerator()

  mem_write = d.ports[esi.AppID("write")].write_port("req")
  mem_write.connect()
  mem_read_addr = d.ports[esi.AppID("read")].write_port("address")
  mem_read_addr.connect()
  mem_read_data = d.ports[esi.AppID("read")].read_port("data")
  mem_read_data.connect()

  # Baseline
  m = conn.manifest()
  # TODO: I broke this. Need to fix it.
  # if (platform == "cosim"):
  # MMIO method
  # conn.cpp_accel.set_manifest_method(esi.esiCppAccel.ManifestMMIO)
  # m_alt = conn.manifest()
  # assert len(m.type_table) == len(m_alt.type_table)

  info = m.module_infos
  dummy_info = None
  for i in info:
    if i.name == "Dummy":
      dummy_info = i
      break
  assert dummy_info is not None

  def read(addr: int) -> bytearray:
    mem_read_addr.write(addr)
    resp = cast(bytearray, mem_read_data.read())
    print(f"resp: {resp}")
    return resp

  # The contents of address 3 are continuously updated to the contents of
  # address 2 by the accelerator.
  data = bytearray([random.randint(0, 2**8 - 1) for _ in range(8)])
  mem_write.write({"address": 2, "data": data})
  resp = read(2)
  try_count = 0

  # Spin until the accelerator has updated the data. Only try a certain number
  # of times. In practice, this should not be used (write should be a function
  # which blocks until the write is complete). Since we are testing
  # functionality, this is appropriate.
  while resp != data and try_count < 10:
    time.sleep(0.01)
    try_count += 1
    resp = read(2)
  assert resp == data
  resp = read(3)
  assert resp == data

  # Check this by writing to address 3 and reading from it. Shouldn't have
  # changed.
  resp = None
  zeros = bytearray([0] * 8)
  mem_write.write({"address": 3, "data": zeros})
  try_count = 0
  while resp != data and try_count < 10:
    time.sleep(0.01)
    try_count += 1
    resp = read(3)
  assert resp == data


@cosim_test(HW_DIR / "esi_ram.py")
def test_cosim_ram(conn: AcceleratorConnection) -> None:
  run(conn)


if __name__ == "__main__":
  platform = sys.argv[1]
  conn_str = sys.argv[2]
  conn = esi.connect(platform, conn_str)
  run(conn)
