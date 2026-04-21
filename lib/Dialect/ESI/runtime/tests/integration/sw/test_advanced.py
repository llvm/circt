from __future__ import annotations

from pathlib import Path
import sys

import esiaccel as esi
from esiaccel.accelerator import Accelerator
from esiaccel.cosim.pytest import cosim_test

HW_DIR = Path(__file__).resolve().parent.parent / "hw"


def run(acc: Accelerator) -> None:

  merge_a = acc.ports[esi.AppID("merge_a")].write_port("data")
  merge_a.connect()
  merge_b = acc.ports[esi.AppID("merge_b")].write_port("data")
  merge_b.connect()
  merge_x = acc.ports[esi.AppID("merge_x")].read_port("data")
  merge_x.connect()

  for i in range(10, 15):
    merge_a.write(i)
    merge_b.write(i + 10)
    x1 = merge_x.read()
    x2 = merge_x.read()
    print(f"merge_a: {i}, merge_b: {i + 10}, "
          f"merge_x 1: {x1}, merge_x 2: {x2}")
    assert x1 == i + 10 or x1 == i
    assert x2 == i + 10 or x2 == i
    assert x1 != x2

  join_a = acc.ports[esi.AppID("join_a")].write_port("data")
  join_a.connect()
  join_b = acc.ports[esi.AppID("join_b")].write_port("data")
  join_b.connect()
  join_x = acc.ports[esi.AppID("join_x")].read_port("data")
  join_x.connect()

  for i in range(15, 27):
    join_a.write(i)
    join_b.write(i + 10)
    x = join_x.read()
    print(f"join_a: {i}, join_b: {i + 10}, join_x: {x}")
    assert x == (i + i + 10) & 0xFFFF

  fork_a = acc.ports[esi.AppID("fork_a")].write_port("data")
  fork_a.connect()
  fork_x = acc.ports[esi.AppID("fork_x")].read_port("data")
  fork_x.connect()
  fork_y = acc.ports[esi.AppID("fork_y")].read_port("data")
  fork_y.connect()

  for i in range(27, 33):
    fork_a.write(i)
    x = fork_x.read()
    y = fork_y.read()
    print(f"fork_a: {i}, fork_x: {x}, fork_y: {y}")
    assert x == y


@cosim_test(HW_DIR / "esi_advanced.py")
def test_cosim_advanced(accelerator: Accelerator) -> None:
  run(accelerator)


if __name__ == "__main__":
  platform = sys.argv[1]
  conn_str = sys.argv[2]
  conn = esi.connect(platform, conn_str)
  run(conn)
