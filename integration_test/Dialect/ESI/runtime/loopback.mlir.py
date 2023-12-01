import esi
import esi.types as types
import sys

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

assert acc.sysinfo().esi_version() == 1
m = acc.manifest()
assert m.api_version == 1


def strType(t: esi.Type) -> str:
  return str(t)
  # if isinstance(t, esi.BundleType):
  #   return "bundle<[{}]>".format(", ".join([
  #       f"{name} {direction} {strType(ty)}" for (name, direction,
  #                                                ty) in t.channels
  #   ]))
  # if isinstance(t, esi.ChannelType):
  #   return f"channel<{strType(t.inner)}>"
  # if isinstance(t, esi.ArrayType):
  #   return f"array<{strType(t.element)}, {t.size}>"
  # if isinstance(t, esi.StructType):
  #   return "struct<{}>".format(", ".join(
  #       ["{name}: {strType(ty)}" for (name, ty) in t.fields]))
  # if isinstance(t, esi.BitsType):
  #   return f"bits<{t.width}>"
  # if isinstance(t, esi.UIntType):
  #   return f"uint<{t.width}>"
  # if isinstance(t, esi.SIntType):
  #   return f"sint<{t.width}>"
  # if isinstance(t, esi.VoidType):
  #   return "void"
  # assert False, f"unknown type: {t}"


for esiType in m.type_table:
  print(f"{esiType}:")
  print(f"  {strType(esiType)}")

d = acc.build_accelerator()

loopback = d.children[esi.AppID("loopback_inst", 0)]
appid = loopback.id
print(appid)
assert appid.name == "loopback_inst"
assert appid.idx == 0

mysvc_send = loopback.ports[esi.AppID("mysvc_recv")].write_port("recv")
mysvc_send.connect()
mysvc_send.write(None)
print(f"mysvc_send.type: {mysvc_send.type}")
assert isinstance(mysvc_send.type, types.VoidType)

mysvc_send = loopback.ports[esi.AppID("mysvc_send")].read_port("send")
mysvc_send.connect()
resp: bool = False
# Reads are non-blocking, so we need to poll.
while not resp:
  print("i0 polling")
  (resp, _) = mysvc_send.read()
print(f"i0 resp: {resp}")

recv = loopback.ports[esi.AppID("loopback_tohw")].write_port("recv")
recv.connect()
assert isinstance(recv.type, types.BitsType)

send = loopback.ports[esi.AppID("loopback_fromhw")].read_port("send")
send.connect()

data = [24]
recv.write(bytearray(data))
resp = False
# Reads are non-blocking, so we need to poll.
while not resp:
  print("polling")
  (resp, _) = send.read()

# Trace platform intentionally produces random responses.
if platform != "trace":
  print(f"data: {data}")
  print(f"resp: {resp}")
  assert resp == data

# Placeholder until we have a runtime function API.
myfunc = d.ports[esi.AppID("func1")]
arg = myfunc.write_port("arg").connect()
result = myfunc.read_port("result").connect()


print("PASS")
