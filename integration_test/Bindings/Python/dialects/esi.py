# REQUIRES: bindings_python
# REQUIRES: capnp
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import esi

from circt.ir import *
from circt import passmanager

with Context() as ctx:
  circt.register_dialects(ctx)
  any_type = esi.AnyType.get()
  print(any_type)
  print()
  # CHECK: !esi.any

  list_type = esi.ListType.get(IntegerType.get_signless(3))
  print(list_type)
  print()
  # CHECK: !esi.list<i3>

  channel_type = esi.ChannelType.get(IntegerType.get_signless(16))
  print(channel_type)
  print()
  # CHECK: !esi.channel<i16>

  bundle_type = esi.BundleType.get(
      [("i16chan", esi.ChannelDirection.TO, channel_type)], False)
  print(bundle_type)
  # CHECK: !esi.bundle<[!esi.channel<i16> to "i16chan"]>
  assert (not bundle_type.resettable)
  for bchan in bundle_type.channels:
    print(bchan)
  # CHECK: ('i16chan', <ChannelDirection.TO: 1>, Type(!esi.channel<i16>))
  print()

  bundle_type = esi.BundleType.get(
      [("i16chan", esi.ChannelDirection.FROM, channel_type)], True)
  print(bundle_type)
  # CHECK: !esi.bundle<[!esi.channel<i16> from "i16chan"] reset >
  assert (bundle_type.resettable)
  print()


# CHECK-LABEL: === testGen called with op:
# CHECK:       %0:2 = esi.service.impl_req svc @HostComms impl as "test"(%clk) : (i1) -> (i8, !esi.bundle<[!esi.channel<i8> to "recv"]>) {
# CHECK:       %2 = esi.service.impl_req.req <@HostComms::@Recv>(["m1", "loopback_tohw"]) : !esi.bundle<[!esi.channel<i8> to "recv"]>
# CHECK:       esi.service.impl_req.output(%2) : !esi.bundle<[!esi.channel<i8> to "recv"]>
def testGen(reqOp: esi.ServiceImplementReqOp) -> bool:
  print("=== testGen called with op:")
  reqOp.print()
  print()
  return True


esi.registerServiceGenerator("test", testGen)

with Context() as ctx:
  circt.register_dialects(ctx)
  mod = Module.parse("""
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>

esi.service.decl @HostComms {
  esi.service.to_client @Recv : !recvI8
}

hw.module @MsTop (in %clk : i1, out chksum : i8) {
  %c = esi.service.instance svc @HostComms impl as  "test" (%clk) : (i1) -> (i8)
  hw.instance "m1" @MsLoopback (clk: %clk: i1) -> ()
  hw.output %c : i8
}

hw.module @MsLoopback (in %clk : i1) {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !recvI8
}
""")
  pm = passmanager.PassManager.parse("builtin.module(esi-connect-services)")
  pm.run(mod.operation)
