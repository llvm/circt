import esiaccel as esi
import sys

platform = sys.argv[1]
conn_str = sys.argv[2]
acc = esi.connect(platform, conn_str)

d = acc.build_accelerator()

merge_a = d.ports[esi.AppID("merge_a")].write_port("data")
merge_a.connect()
merge_b = d.ports[esi.AppID("merge_b")].write_port("data")
merge_b.connect()
merge_x = d.ports[esi.AppID("merge_x")].read_port("data")
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

join_a = d.ports[esi.AppID("join_a")].write_port("data")
join_a.connect()
join_b = d.ports[esi.AppID("join_b")].write_port("data")
join_b.connect()
join_x = d.ports[esi.AppID("join_x")].read_port("data")
join_x.connect()

for i in range(15, 27):
  join_a.write(i)
  join_b.write(i + 10)
  x = join_x.read()
  print(f"join_a: {i}, join_b: {i + 10}, join_x: {x}")
  assert x == (i + i + 10) & 0xFFFF

fork_a = d.ports[esi.AppID("fork_a")].write_port("data")
fork_a.connect()
fork_x = d.ports[esi.AppID("fork_x")].read_port("data")
fork_x.connect()
fork_y = d.ports[esi.AppID("fork_y")].read_port("data")
fork_y.connect()

for i in range(27, 33):
  fork_a.write(i)
  x = fork_x.read()
  y = fork_y.read()
  print(f"fork_a: {i}, fork_x: {x}, fork_y: {y}")
  assert x == y
