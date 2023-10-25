// RUN: circt-opt %s --esi-connect-services --esi-appid-hier=top=top | FileCheck --check-prefix=HIER %s
// RUN: circt-opt %s --esi-appid-hier=top=top --esi-build-manifest=to-file=%t1.json
// RUN: FileCheck --input-file=%t1.json %s

hw.type_scope @__hw_typedecls {
  hw.typedecl @foo, "Foo" : i1
}
!alias = !hw.typealias<@__hw_typedecls::@foo, i1>

!sendI8 = !esi.bundle<[!esi.channel<i8> to "send"]>
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>

esi.service.decl @HostComms {
  esi.service.to_server @Send : !sendI8
  esi.service.to_client @Recv : !recvI8
}

hw.module @Loopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req.to_client <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) {esi.appid=#esi.appid<"loopback_tohw">} : !recvI8
  %dataOut = esi.bundle.unpack from %dataInBundle : !recvI8
  %dataOutBundle = esi.bundle.pack %dataOut : !sendI8
  esi.service.req.to_server %dataOutBundle -> <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !sendI8
}

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  %0 = esi.null : !esi.channel<si14>
  esi.cosim %clk, %rst, %0, "t2" : !esi.channel<si14> -> !esi.channel<i32>

  %1 = esi.null : !esi.channel<!esi.list<ui14>>
  esi.cosim %clk, %rst, %1, "t1" : !esi.channel<!esi.list<ui14>> -> !esi.channel<!hw.array<3xi5>>

  %2 = esi.null : !esi.channel<!esi.any>
  esi.cosim %clk, %rst, %2, "t2" : !esi.channel<!esi.any> -> !esi.channel<!hw.struct<"foo": i5>>

  %3 = esi.null : !esi.channel<!alias>
  esi.cosim %clk, %rst, %3, "t3" : !esi.channel<!alias> -> !esi.channel<i32>

  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst">}
}

// HIER-LABEL:  esi.esi.manifest.hier_root @top {
// HIER:          esi.esi.manifest.service_impl #esi.appid<"cosim"> svc @HostComms by "cosim" with {} {
// HIER:            esi.esi.manifest.impl_conn [#esi.appid<"loopback_inst">, #esi.appid<"loopback_tohw">] req <@HostComms::@Recv>(!esi.bundle<[!esi.channel<i8> to "recv"]>) with {channel_assignments = {recv = "loopback_inst.loopback_tohw.recv"}}
// HIER:            esi.esi.manifest.impl_conn [#esi.appid<"loopback_inst">, #esi.appid<"loopback_fromhw">] req <@HostComms::@Send>(!esi.bundle<[!esi.channel<i8> from "send"]>) with {channel_assignments = {send = "loopback_inst.loopback_fromhw.send"}}
// HIER:          }
// HIER:          esi.esi.manifest.hier_node #esi.appid<"loopback_inst"> mod @Loopback {
// HIER:            esi.esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, toClient, !esi.bundle<[!esi.channel<i8> to "recv"]>
// HIER:            esi.esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, toServer, !esi.bundle<[!esi.channel<i8> to "send"]>
// HIER:          }
// HIER:        }

// CHECK:      {
// CHECK-NEXT:   "api_version": 1,
// CHECK-LABEL:  "types": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<si14>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "si14",
// CHECK-NEXT:         "dialect": "builtin",
// CHECK-NEXT:         "hw_bitwidth": 14,
// CHECK-NEXT:         "mnemonic": "int",
// CHECK-NEXT:         "signedness": "signed"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<i32>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "i32",
// CHECK-NEXT:         "dialect": "builtin",
// CHECK-NEXT:         "hw_bitwidth": 32,
// CHECK-NEXT:         "mnemonic": "int",
// CHECK-NEXT:         "signedness": "signless"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!esi.list<ui14>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!esi.list<ui14>",
// CHECK-NEXT:         "dialect": "esi",
// CHECK-NEXT:         "element": {
// CHECK-NEXT:           "circt_name": "ui14",
// CHECK-NEXT:           "dialect": "builtin",
// CHECK-NEXT:           "hw_bitwidth": 14,
// CHECK-NEXT:           "mnemonic": "int",
// CHECK-NEXT:           "signedness": "unsigned"
// CHECK-NEXT:         },
// CHECK-NEXT:         "mnemonic": "list"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!hw.array<3xi5>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!hw.array<3xi5>",
// CHECK-NEXT:         "dialect": "hw",
// CHECK-NEXT:         "element": {
// CHECK-NEXT:           "circt_name": "i5",
// CHECK-NEXT:           "dialect": "builtin",
// CHECK-NEXT:           "hw_bitwidth": 5,
// CHECK-NEXT:           "mnemonic": "int",
// CHECK-NEXT:           "signedness": "signless"
// CHECK-NEXT:         },
// CHECK-NEXT:         "hw_bitwidth": 15,
// CHECK-NEXT:         "mnemonic": "array",
// CHECK-NEXT:         "size": 3
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!esi.any>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!esi.any",
// CHECK-NEXT:         "dialect": "esi",
// CHECK-NEXT:         "mnemonic": "any"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!hw.struct<foo: i5>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!hw.struct<foo: i5>",
// CHECK-NEXT:         "dialect": "hw",
// CHECK-NEXT:         "fields": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "name": "foo",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:               "circt_name": "i5",
// CHECK-NEXT:               "dialect": "builtin",
// CHECK-NEXT:               "hw_bitwidth": 5,
// CHECK-NEXT:               "mnemonic": "int",
// CHECK-NEXT:               "signedness": "signless"
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         ],
// CHECK-NEXT:         "hw_bitwidth": 5,
// CHECK-NEXT:         "mnemonic": "struct"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "circt_name": "!esi.channel<!hw.typealias<@__hw_typedecls::@foo, i1>>",
// CHECK-NEXT:       "dialect": "esi",
// CHECK-NEXT:       "inner": {
// CHECK-NEXT:         "circt_name": "!hw.typealias<@__hw_typedecls::@foo, i1>",
// CHECK-NEXT:         "dialect": "hw",
// CHECK-NEXT:         "hw_bitwidth": 1,
// CHECK-NEXT:         "inner": {
// CHECK-NEXT:           "circt_name": "i1",
// CHECK-NEXT:           "dialect": "builtin",
// CHECK-NEXT:           "hw_bitwidth": 1,
// CHECK-NEXT:           "mnemonic": "int",
// CHECK-NEXT:           "signedness": "signless"
// CHECK-NEXT:         },
// CHECK-NEXT:         "mnemonic": "alias",
// CHECK-NEXT:         "name": "Foo"
// CHECK-NEXT:       },
// CHECK-NEXT:       "mnemonic": "channel"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
