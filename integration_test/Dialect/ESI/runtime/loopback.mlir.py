import esi
import IPython
import sys

conn = f"{sys.argv[1]}:{sys.argv[2]}"

acc = esi.Accelerator("cosim", conn)

assert acc.sysinfo().esi_version() == 1
assert acc.manifest.api_version == 1

#print(acc.sysinfo().json_manifest())
d = acc.manifest.build_design(acc)

IPython.embed()
print(d.children)
appid = d.children[0].id
print(appid.name)
print(appid.idx)
