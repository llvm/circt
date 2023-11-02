import esi
import os
import sys

conn = f"{sys.argv[1]}:{sys.argv[2]}"

acc = esi.Accelerator("cosim", conn)

assert acc.sysinfo().esi_version() == 1
assert acc.manifest._manifest['api_version'] == 1
