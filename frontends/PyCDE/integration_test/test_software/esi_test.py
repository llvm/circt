import esiaccel as esi

import sys
from typing import Optional

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1
print(m.type_table)

d = acc.build_accelerator()

recv = d.ports[esi.AppID("loopback_add7")].read_port("result")
recv.connect()

send = d.ports[esi.AppID("loopback_add7")].write_port("arg")
send.connect()

data = 10234
send.write(data)
got_data = False
resp: Optional[int] = None
# Reads are non-blocking, so we need to poll.
while not got_data:
  (got_data, resp) = recv.read()

print(f"data: {data}")
print(f"resp: {resp}")
assert resp == data + 7

print("PASS")
