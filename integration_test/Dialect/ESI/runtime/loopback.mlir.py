import esi
import sys

acc = esi.Accelerator(sys.argv[1], sys.argv[2])

assert acc.sysinfo().esi_version() == 1
assert acc.manifest.api_version == 1

d = acc.manifest.build_design(acc)

appid = d.children[0].id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0
