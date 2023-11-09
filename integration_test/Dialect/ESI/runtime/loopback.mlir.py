import esi
import sys

platform = sys.argv[1]
acc = esi.Accelerator(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1
print(m.type_table)

d = m.build_design(acc)

appid = d.children[0].id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0

# Services are only hooked up for the trace backend currently.
if platform == "trace":
  d.children[0].ports[0].getWrite("recv").write([45] * 128)
  d.children[0].ports[0].getWrite("recv").write([24])
  d.children[0].ports[0].getWrite("recv").write([24, 45, 138])

  d.children[0].ports[1].getRead("send").read(8)
