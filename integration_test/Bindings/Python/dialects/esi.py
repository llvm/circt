# REQUIRES: bindings_python
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
  # CHECK: ('i16chan', ChannelDirection.TO, Type(!esi.channel<i16>))
  print()

  bundle_type = esi.BundleType.get(
      [("i16chan", esi.ChannelDirection.FROM, channel_type)], True)
  print(bundle_type)
  # CHECK: !esi.bundle<[!esi.channel<i16> from "i16chan"] reset >
  assert (bundle_type.resettable)
  print()

# CHECK-LABEL: === testGen called with ops:
# CHECK-NEXT:  [[R0:%.+]]:2 = esi.service.impl_req #esi.appid<"mstop"> svc @HostComms impl as "test"(%clk) : (i1) -> (i8, !esi.bundle<[!esi.channel<i8> to "recv"]>) {
# CHECK-NEXT:    [[R2:%.+]] = esi.service.impl_req.req <@HostComms::@Recv>([#esi.appid<"loopback_tohw">]) : !esi.bundle<[!esi.channel<i8> to "recv"]>
# CHECK-NEXT:  }
# CHECK-NEXT:  esi.service.decl @HostComms {
# CHECK-NEXT:    esi.service.port @Recv : !esi.bundle<[!esi.channel<i8> to "recv"]>
# CHECK-NEXT:  }
# CHECK-NEXT:  esi.manifest.service_impl #esi.appid<"mstop"> svc @HostComms by "test" with {}


def testGen(reqOp: Operation, decl_op: Operation, rec_op: Operation) -> bool:
  print("=== testGen called with ops:")
  reqOp.print()
  print()
  decl_op.print()
  print()
  rec_op.print()
  print()
  return True


esi.registerServiceGenerator("test", testGen)

with Context() as ctx:
  circt.register_dialects(ctx)
  mod = Module.parse("""
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>

esi.service.decl @HostComms {
  esi.service.port @Recv : !recvI8
}

hw.module @MsTop (in %clk : i1, out chksum : i8) {
  %c = esi.service.instance #esi.appid<"mstop"> svc @HostComms impl as  "test" (%clk) : (i1) -> (i8)
  hw.instance "m1" @MsLoopback (clk: %clk: i1) -> ()
  hw.output %c : i8
}

hw.module @MsLoopback (in %clk : i1) {
  %dataIn = esi.service.req <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) : !recvI8
}
""")
  pm = passmanager.PassManager.parse("builtin.module(esi-connect-services)")
  pm.run(mod.operation)
