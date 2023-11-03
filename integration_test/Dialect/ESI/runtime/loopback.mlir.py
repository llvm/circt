import esi
import os
import IPython
import sys

conn = f"{sys.argv[1]}:{sys.argv[2]}"

acc = esi.Accelerator("cosim", conn)

assert acc.sysinfo().esi_version() == 1
assert acc.manifest.api_version == 1

appid = acc.manifest.design.id
print(appid.name)
print(appid.idx)

IPython.embed()
