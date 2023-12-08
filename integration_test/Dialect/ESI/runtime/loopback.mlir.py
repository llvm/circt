from typing import List
import esi
import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1


def strType(t: esi.Type) -> str:
  if isinstance(t, esi.BundleType):
    return "bundle<[{}]>".format(", ".join([
        f"{name} {direction} {strType(ty)}" for (name, direction,
                                                 ty) in t.channels
    ]))
  if isinstance(t, esi.ChannelType):
    return f"channel<{strType(t.inner)}>"
  if isinstance(t, esi.ArrayType):
    return f"array<{strType(t.element)}, {t.size}>"
  if isinstance(t, esi.StructType):
    return "struct<{}>".format(", ".join(
        ["{name}: {strType(ty)}" for (name, ty) in t.fields]))
  if isinstance(t, esi.BitsType):
    return f"bits<{t.width}>"
  if isinstance(t, esi.UIntType):
    return f"uint<{t.width}>"
  if isinstance(t, esi.SIntType):
    return f"sint<{t.width}>"
  assert False, f"unknown type: {t}"


for esiType in m.type_table:
  print(f"{esiType}:")
  print(f"  {strType(esiType)}")

d = m.build_accelerator(acc)

loopback = d.children[esi.AppID("loopback_inst", 0)]
appid = loopback.id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0

mysvc_send = loopback.ports[esi.AppID("mysvc_recv")].channels["recv"]
mysvc_send.connect()
mysvc_send.write([0])
assert str(mysvc_send.type) == "<!esi.channel<i0>>"

mysvc_send = loopback.ports[esi.AppID("mysvc_send")].channels["send"]
mysvc_send.connect()
resp: List[int] = []
# Reads are non-blocking, so we need to poll.
while resp == []:
  print("i0 polling")
  resp = mysvc_send.read(1)
print(f"i0 resp: {resp}")

recv = loopback.ports[esi.AppID("loopback_tohw")].channels["recv"]
recv.connect()

send = loopback.ports[esi.AppID("loopback_fromhw")].channels["send"]
send.connect()

data = [24]
recv.write(data)
resp = []
# Reads are non-blocking, so we need to poll.
while resp == []:
  print("polling")
  resp = send.read(1)

# Trace platform intentionally produces random responses.
if platform != "trace":
  print(f"data: {data}")
  print(f"resp: {resp}")
  assert resp == data

# Placeholder until we have a runtime function API.
myfunc = d.ports[esi.AppID("func1")]
myfunc.channels["arg"].connect()
myfunc.channels["result"].connect()

print("PASS")
