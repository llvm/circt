import esiaccel as esi
import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

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
  print(f"merge_a: {i}, merge_b: {i + 10}, "
        f"merge_x 1: {merge_x.read()}, merge_x 2: {merge_x.read()}")

sys.stdin.read()
