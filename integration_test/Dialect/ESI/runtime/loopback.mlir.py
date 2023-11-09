import esi
import sys

acc = esi.Accelerator(sys.argv[1], sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1
print(m.type_table)

d = m.build_design(acc)

appid = d.children[0].id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0
