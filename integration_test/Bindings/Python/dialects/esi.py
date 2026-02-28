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

  # Test WindowFieldType
  field_name_attr = StringAttr.get("field1")
  window_field_type = esi.WindowFieldType.get(field_name_attr, 5)
  print(window_field_type)
  # CHECK: !esi.window.field<"field1", 5>
  print(window_field_type.field_name)
  # CHECK: "field1"
  print(window_field_type.num_items)
  # CHECK: 5
  print(window_field_type.bulk_count_width)
  # CHECK: 0
  print()

  # Test WindowFieldType with bulk_count_width (bulk transfer mode)
  bulk_field_name_attr = StringAttr.get("payload")
  bulk_window_field_type = esi.WindowFieldType.get(bulk_field_name_attr,
                                                   bulk_count_width=16)
  print(bulk_window_field_type)
  # CHECK: !esi.window.field<"payload" countWidth 16>
  print(bulk_window_field_type.field_name)
  # CHECK: "payload"
  print(bulk_window_field_type.num_items)
  # CHECK: 0
  print(bulk_window_field_type.bulk_count_width)
  # CHECK: 16
  print()

  # Test WindowFrameType
  frame_name_attr = StringAttr.get("frame1")
  field2_name_attr = StringAttr.get("field2")
  window_field2_type = esi.WindowFieldType.get(field2_name_attr)
  window_frame_type = esi.WindowFrameType.get(
      frame_name_attr, [window_field_type, window_field2_type])
  print(window_frame_type)
  # CHECK: !esi.window.frame<"frame1", [<"field1", 5>, <"field2">]>
  print(window_frame_type.name)
  # CHECK: "frame1"
  print(len(window_frame_type.members))
  # CHECK: 2
  for member in window_frame_type.members:
    print(member)
  # CHECK: !esi.window.field<"field1", 5>
  # CHECK: !esi.window.field<"field2">
  print()

  # Test WindowType - must use struct type with matching fields
  window_name_attr = StringAttr.get("window1")
  i32_type = IntegerType.get_signless(32)
  i8_type = IntegerType.get_signless(8)
  # field1 needs to be an array since it has numItems=5
  array_type = Type.parse("!hw.array<5xi32>")
  # Create a struct with field1 (array) and field2 (scalar)
  struct_type = Type.parse("!hw.struct<field1: !hw.array<5xi32>, field2: i8>")
  window_type = esi.WindowType.get(window_name_attr, struct_type,
                                   [window_frame_type])
  print(window_type)
  # CHECK: !esi.window<"window1", !hw.struct<field1: !hw.array<5xi32>, field2: i8>, [<"frame1", [<"field1", 5>, <"field2">]>]>
  print(window_type.name)
  # CHECK: "window1"
  print(window_type.into)
  # CHECK: !hw.struct<field1: !hw.array<5xi32>, field2: i8>
  print(len(window_type.frames))
  # CHECK: 1
  for frame in window_type.frames:
    print(frame)
  # CHECK: !esi.window.frame<"frame1", [<"field1", 5>, <"field2">]>
  print(window_type.get_lowered_type())
  # CHECK: !hw.union<frame1: !hw.struct<field1: !hw.array<5xi32>, field2: i8>>
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
