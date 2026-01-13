// REQUIRES: zlib
// RUN: circt-opt %s --esi-connect-services --esi-appid-hier=top=top --esi-build-manifest="top=top" > %t1.mlir 
// RUN: circt-opt %t1.mlir | FileCheck --check-prefix=HIER %s
// RUN: FileCheck --input-file=esi_system_manifest.json %s
// RUN: circt-opt %t1.mlir --esi-clean-metadata --lower-esi-bundles --lower-esi-ports --lower-esi-to-hw=platform=cosim | FileCheck --check-prefix=HW %s

hw.type_scope @__hw_typedecls {
  hw.typedecl @foo, "Foo" : i1
}
!alias = !hw.typealias<@__hw_typedecls::@foo, i1>

!sendI8 = !esi.bundle<[!esi.channel<i8> from "send"]>
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>
!sendI0 = !esi.bundle<[!esi.channel<i0> from "send"]>

esi.service.decl @HostComms {
  esi.service.port @Send : !sendI8
  esi.service.port @Recv : !recvI8
  esi.service.port @SendI0 : !sendI0
}

hw.module @Loopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) {esi.appid=#esi.appid<"loopback_tohw">} : !recvI8
  %dataOut = esi.bundle.unpack from %dataInBundle : !recvI8
  esi.bundle.unpack %dataOut from %dataOutBundle : !sendI8
  %dataOutBundle = esi.service.req <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !sendI8

  %c0_0 = hw.constant 0 : i0
  %c0_1 = hw.constant 0 : i1
  %sendi0_channel, %ready = esi.wrap.vr %c0_0, %c0_1 : i0
  esi.bundle.unpack %sendi0_channel from %sendi0_bundle : !sendI0
  %sendi0_bundle = esi.service.req <@HostComms::@SendI0> (#esi.appid<"loopback_fromhw_i0">) : !sendI0
}

esi.manifest.sym @Loopback name "LoopbackIP" version "v0.0" summary "IP which simply echos bytes" {foo=1}
esi.manifest.constants @Loopback {depth=5:ui32}

esi.service.std.func @funcs

// CONN-LABEL:   hw.module @CallableFunc1(in %func1 : !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>) {
// CONN-NEXT:      esi.manifest.req #esi.appid<"func1">, <@funcs::@call> std "esi.service.std.func", toClient, !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
// CONN-NEXT:      %arg = esi.bundle.unpack %arg from %func1 : !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
!func1Signature = !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
hw.module @CallableFunc1() {
  %call = esi.service.req <@funcs::@call> (#esi.appid<"func1">) : !func1Signature
  %arg = esi.bundle.unpack %arg from %call : !func1Signature
}

// Test window types in manifest
!WindowType = !hw.struct<header: i8, data: i16>
!WindowDef = !esi.window<
  "TestWindow", !WindowType, [
    <"HeaderFrame", [<"header">]>,
    <"DataFrame", [<"data">]>
  ]>

!windowBundle = !esi.bundle<[!esi.channel<!WindowDef> to "window_in"]>

esi.service.decl @WindowService {
  esi.service.port @WindowPort : !windowBundle
}

hw.module.extern @WindowProducer(out window_port: !windowBundle)

hw.module @WindowConsumer() {
  %window_bundle = esi.service.req <@WindowService::@WindowPort> (#esi.appid<"window_consumer">) : !windowBundle
  // Unpack the bundle to receive the window_in channel
  %window_in = esi.bundle.unpack from %window_bundle : !windowBundle
}

// Test window with list and countWidth in manifest
!ListWindowType = !hw.struct<address: i32, data: !esi.list<i64>>
!ListWindowDef = !esi.window<
  "BulkTransferWindow", !ListWindowType, [
    <"HeaderFrame", [<"address">, <"data" countWidth 8>]>,
    <"DataFrame", [<"data", 4>]>
  ]>

!listWindowBundle = !esi.bundle<[!esi.channel<!ListWindowDef> to "bulk_in"]>

esi.service.decl @ListWindowService {
  esi.service.port @BulkPort : !listWindowBundle
}

hw.module @ListWindowConsumer() {
  %bulk_bundle = esi.service.req <@ListWindowService::@BulkPort> (#esi.appid<"bulk_consumer">) : !listWindowBundle
  %bulk_in = esi.bundle.unpack from %bulk_bundle : !listWindowBundle
}

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  esi.service.instance #esi.appid<"window_svc"> svc @WindowService impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  esi.service.instance #esi.appid<"list_window_svc"> svc @ListWindowService impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[0]>}
  hw.instance "m2" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[1]>}
  hw.instance "func1" @CallableFunc1() -> ()
  hw.instance "window_consumer" @WindowConsumer() -> () {esi.appid=#esi.appid<"window_consumer_inst">}
  hw.instance "list_window_consumer" @ListWindowConsumer() -> () {esi.appid=#esi.appid<"list_window_consumer_inst">}
}

// HIER-LABEL:  esi.manifest.compressed <"{{.+}}">
// HIER-LABEL:  esi.manifest.hier_root @top {
// HIER-NEXT:     esi.manifest.service_impl #esi.appid<"cosim"> svc @HostComms by "cosim" engine with {} {
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_tohw">] req <@HostComms::@Recv>(!esi.bundle<[!esi.channel<i8> to "recv"]>) channels {recv = {name = "loopback_inst[0].loopback_tohw.recv", type = "cosim"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_fromhw">] req <@HostComms::@Send>(!esi.bundle<[!esi.channel<i8> from "send"]>) channels {send = {name = "loopback_inst[0].loopback_fromhw.send", type = "cosim"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_fromhw_i0">] req <@HostComms::@SendI0>(!esi.bundle<[!esi.channel<i0> from "send"]>) channels {send = {name = "loopback_inst[0].loopback_fromhw_i0.send", type = "cosim"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_tohw">] req <@HostComms::@Recv>(!esi.bundle<[!esi.channel<i8> to "recv"]>) channels {recv = {name = "loopback_inst[1].loopback_tohw.recv", type = "cosim"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_fromhw">] req <@HostComms::@Send>(!esi.bundle<[!esi.channel<i8> from "send"]>) channels {send = {name = "loopback_inst[1].loopback_fromhw.send", type = "cosim"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_fromhw_i0">] req <@HostComms::@SendI0>(!esi.bundle<[!esi.channel<i0> from "send"]>) channels {send = {name = "loopback_inst[1].loopback_fromhw_i0.send", type = "cosim"}}
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.service_impl #esi.appid<"window_svc"> svc @WindowService by "cosim" engine with {} {
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"window_consumer_inst">, #esi.appid<"window_consumer">] req <@WindowService::@WindowPort>(!esi.bundle<[!esi.channel<!esi.window<"TestWindow", !hw.struct<header: i8, data: i16>, [<"HeaderFrame", [<"header">]>, <"DataFrame", [<"data">]>]>> to "window_in"]>) channels {window_in = {name = "window_consumer_inst.window_consumer.window_in", type = "cosim"}}
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.service_impl #esi.appid<"list_window_svc"> svc @ListWindowService by "cosim" engine with {} {
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"list_window_consumer_inst">, #esi.appid<"bulk_consumer">] req <@ListWindowService::@BulkPort>(!esi.bundle<[!esi.channel<!esi.window<"BulkTransferWindow", !hw.struct<address: i32, data: !esi.list<i64>>, [<"HeaderFrame", [<"address">, <"data" countWidth 8>]>, <"DataFrame", [<"data", 4>]>]>> to "bulk_in"]>) channels {bulk_in = {name = "list_window_consumer_inst.bulk_consumer.bulk_in", type = "cosim"}}
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.hier_node #esi.appid<"loopback_inst"[0]> mod @Loopback {
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, !esi.bundle<[!esi.channel<i8> to "recv"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, !esi.bundle<[!esi.channel<i8> from "send"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw_i0">, <@HostComms::@SendI0>, !esi.bundle<[!esi.channel<i0> from "send"]>
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.hier_node #esi.appid<"loopback_inst"[1]> mod @Loopback {
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, !esi.bundle<[!esi.channel<i8> to "recv"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, !esi.bundle<[!esi.channel<i8> from "send"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw_i0">, <@HostComms::@SendI0>, !esi.bundle<[!esi.channel<i0> from "send"]>
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.req #esi.appid<"func1">, <@funcs::@call> std "esi.service.std.func", !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
// HIER-NEXT:     esi.manifest.hier_node #esi.appid<"window_consumer_inst"> mod @WindowConsumer {
// HIER-NEXT:       esi.manifest.req #esi.appid<"window_consumer">, <@WindowService::@WindowPort>, !esi.bundle<[!esi.channel<!esi.window<"TestWindow", !hw.struct<header: i8, data: i16>, [<"HeaderFrame", [<"header">]>, <"DataFrame", [<"data">]>]>> to "window_in"]>
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.hier_node #esi.appid<"list_window_consumer_inst"> mod @ListWindowConsumer {
// HIER-NEXT:       esi.manifest.req #esi.appid<"bulk_consumer">, <@ListWindowService::@BulkPort>, !esi.bundle<[!esi.channel<!esi.window<"BulkTransferWindow", !hw.struct<address: i32, data: !esi.list<i64>>, [<"HeaderFrame", [<"address">, <"data" countWidth 8>]>, <"DataFrame", [<"data", 4>]>]>> to "bulk_in"]>
// HIER-NEXT:     }
// HIER-NEXT:   }

// HW-LABEL:    hw.module @__ESI_Manifest_ROM(in %clk : !seq.clock, in %address : i29, out data : i64) {
// HW:            [[R0:%.+]] = hw.aggregate_constant
// HW:            [[R1:%.+]] = sv.reg : !hw.inout<uarray<{{.*}}xi64>>
// HW:            sv.assign [[R1]], [[R0]] : !hw.uarray<{{.*}}xi64>
// HW:            [[R2:%.+]] = comb.extract %address from 0 : (i29) -> i8
// HW:            [[R3:%.+]] = seq.compreg  [[R2]], %clk : i8
// HW:            [[R4:%.+]] = sv.array_index_inout [[R1]][[[R3]]] : !hw.inout<uarray<{{.*}}xi64>>, i8
// HW:            [[R5:%.+]] = sv.read_inout [[R4]] : !hw.inout<i64>
// HW:            [[R6:%.+]] = seq.compreg  [[R5]], %clk : i64
// HW:            hw.output [[R6]] : i64

// HW-LABEL:    hw.module @top
// HW:            hw.instance "__cycle_counter" @Cosim_CycleCount<CORE_CLOCK_FREQUENCY_HZ: i64 = 0>(clk: %clk: !seq.clock, rst: %rst: i1) -> ()
// HW:            hw.instance "__manifest" @__ESIManifest() -> ()
// HW-LABEL:    hw.module.extern @Cosim_CycleCount<CORE_CLOCK_FREQUENCY_HZ: i64>(in %clk : !seq.clock, in %rst : i1) attributes {verilogName = "Cosim_CycleCount"}
// HW-LABEL:    hw.module.extern @Cosim_Manifest<COMPRESSED_MANIFEST_SIZE: i32>(in %compressed_manifest : !hw.array<#hw.param.decl.ref<"COMPRESSED_MANIFEST_SIZE">xi8>) attributes {verilogName = "Cosim_Manifest"}
// HW-LABEL:    hw.module @__ESIManifest()
// HW:            hw.instance "__manifest" @Cosim_Manifest<COMPRESSED_MANIFEST_SIZE: i32 = {{.+}}>(compressed_manifest: %{{.+}}: !hw.array<{{.+}}xi8>) -> ()

// CHECK-LABEL: {
// CHECK-LABEL:   "apiVersion": 0,

// CHECK-LABEL:   "design": {
// CHECK-NEXT:      "instanceOf": "@top",
// CHECK-NEXT:      "clientPorts": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "func1"
// CHECK-NEXT:          },
// CHECK-NEXT:          "typeID": "!esi.bundle<[!esi.channel<i16> to \"arg\", !esi.channel<i16> from \"result\"]>",
// CHECK-NEXT:          "servicePort": {
// CHECK-NEXT:            "port": "call",
// CHECK-NEXT:            "serviceName": "@funcs"
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      ],

// CHECK-LABEL:     "engines": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "cosim"
// CHECK-NEXT:          },
// CHECK-NEXT:          "service": "@HostComms",
// CHECK-NEXT:          "serviceImplName": "cosim",
// CHECK-NEXT:          "clientDetails": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "recv": {
// CHECK-NEXT:                  "name": "loopback_inst[0].loopback_tohw.recv",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "index": 0,
// CHECK-NEXT:                  "name": "loopback_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "loopback_tohw"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Recv",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "send": {
// CHECK-NEXT:                  "name": "loopback_inst[0].loopback_fromhw.send",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "index": 0,
// CHECK-NEXT:                  "name": "loopback_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "loopback_fromhw"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Send",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "send": {
// CHECK-NEXT:                  "name": "loopback_inst[0].loopback_fromhw_i0.send",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "index": 0,
// CHECK-NEXT:                  "name": "loopback_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "loopback_fromhw_i0"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "SendI0",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "recv": {
// CHECK-NEXT:                  "name": "loopback_inst[1].loopback_tohw.recv",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "index": 1,
// CHECK-NEXT:                  "name": "loopback_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "loopback_tohw"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Recv",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "send": {
// CHECK-NEXT:                  "name": "loopback_inst[1].loopback_fromhw.send",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "index": 1,
// CHECK-NEXT:                  "name": "loopback_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "loopback_fromhw"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Send",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "send": {
// CHECK-NEXT:                  "name": "loopback_inst[1].loopback_fromhw_i0.send",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "index": 1,
// CHECK-NEXT:                  "name": "loopback_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "loopback_fromhw_i0"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "SendI0",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "window_svc"
// CHECK-NEXT:          },
// CHECK-NEXT:          "service": "@WindowService",
// CHECK-NEXT:          "serviceImplName": "cosim",
// CHECK-NEXT:          "clientDetails": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "window_in": {
// CHECK-NEXT:                  "name": "window_consumer_inst.window_consumer.window_in",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "window_consumer_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "window_consumer"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "WindowPort",
// CHECK-NEXT:                "serviceName": "@WindowService"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "list_window_svc"
// CHECK-NEXT:          },
// CHECK-NEXT:          "service": "@ListWindowService",
// CHECK-NEXT:          "serviceImplName": "cosim",
// CHECK-NEXT:          "clientDetails": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "channelAssignments": {
// CHECK-NEXT:                "bulk_in": {
// CHECK-NEXT:                  "name": "list_window_consumer_inst.bulk_consumer.bulk_in",
// CHECK-NEXT:                  "type": "cosim"
// CHECK-NEXT:                }
// CHECK-NEXT:              },
// CHECK-NEXT:              "relAppIDPath": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "list_window_consumer_inst"
// CHECK-NEXT:                },
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "bulk_consumer"
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "BulkPort",
// CHECK-NEXT:                "serviceName": "@ListWindowService"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ],

// CHECK-LABEL:     "children": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "index": 0,
// CHECK-NEXT:            "name": "loopback_inst"
// CHECK-NEXT:          },
// CHECK-NEXT:          "instanceOf": "@Loopback",
// CHECK-NEXT:          "clientPorts": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_tohw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "typeID": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>",
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Recv",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "typeID": "!esi.bundle<[!esi.channel<i8> from \"send\"]>",
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Send",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw_i0"
// CHECK-NEXT:              },
// CHECK-NEXT:              "typeID": "!esi.bundle<[!esi.channel<i0> from \"send\"]>",
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "SendI0",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "children": []
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "index": 1,
// CHECK-NEXT:            "name": "loopback_inst"
// CHECK-NEXT:          },
// CHECK-NEXT:          "instanceOf": "@Loopback",
// CHECK-NEXT:          "clientPorts": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_tohw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "typeID": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>",
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Recv",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "typeID": "!esi.bundle<[!esi.channel<i8> from \"send\"]>",
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "Send",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw_i0"
// CHECK-NEXT:              },
// CHECK-NEXT:              "typeID": "!esi.bundle<[!esi.channel<i0> from \"send\"]>",
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "SendI0",
// CHECK-NEXT:                "serviceName": "@HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "children": []
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "window_consumer_inst"
// CHECK-NEXT:          },
// CHECK-NEXT:          "instanceOf": "@WindowConsumer",
// CHECK-NEXT:          "clientPorts": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "window_consumer"
// CHECK-NEXT:              },
// CHECK:               "typeID": "!esi.bundle<[!esi.channel<!esi.window<\"TestWindow\"
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "WindowPort",
// CHECK-NEXT:                "serviceName": "@WindowService"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "children": []
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "list_window_consumer_inst"
// CHECK-NEXT:          },
// CHECK-NEXT:          "instanceOf": "@ListWindowConsumer",
// CHECK-NEXT:          "clientPorts": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "bulk_consumer"
// CHECK-NEXT:              },
// CHECK:               "typeID": "!esi.bundle<[!esi.channel<!esi.window<\"BulkTransferWindow\"
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "port": "BulkPort",
// CHECK-NEXT:                "serviceName": "@ListWindowService"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "children": []
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    },

// CHECK-LABEL:   "serviceDeclarations": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "@HostComms",
// CHECK-NEXT:        "ports": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "Send",
// CHECK-NEXT:            "typeID": "!esi.bundle<[!esi.channel<i8> from \"send\"]>"
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "Recv",
// CHECK-NEXT:            "typeID": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>"
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "SendI0",
// CHECK-NEXT:            "typeID": "!esi.bundle<[!esi.channel<i0> from \"send\"]>"
// CHECK-NEXT:          }
// CHECK-NEXT:        ]
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "@funcs",
// CHECK-NEXT:        "serviceName": "esi.service.std.func",
// CHECK-NEXT:        "ports": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "call",
// CHECK-NEXT:            "typeID": "!esi.bundle<[!esi.channel<!esi.any> to \"arg\", !esi.channel<!esi.any> from \"result\"]>"
// CHECK-NEXT:          }
// CHECK-NEXT:        ]
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "@WindowService",
// CHECK-NEXT:        "ports": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "WindowPort",
// CHECK-NEXT:            "typeID": "!esi.bundle<[!esi.channel<!esi.window<\"TestWindow\"
// CHECK-NEXT:          }
// CHECK-NEXT:        ]
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "@ListWindowService",
// CHECK-NEXT:        "ports": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "BulkPort",
// CHECK-NEXT:            "typeID": "!esi.bundle<[!esi.channel<!esi.window<\"BulkTransferWindow\"
// CHECK-NEXT:          }
// CHECK-NEXT:        ]
// CHECK-NEXT:      }
// CHECK-NEXT:    ],

// CHECK-LABEL:   "modules": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "@top"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "@Loopback",
// CHECK-NEXT:        "symInfo": {
// CHECK-NEXT:          "foo": {
// CHECK-NEXT:            "type": "i64",
// CHECK-NEXT:            "value": 1
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "LoopbackIP",
// CHECK-NEXT:          "summary": "IP which simply echos bytes",
// CHECK-NEXT:          "version": "v0.0"
// CHECK-NEXT:        },
// CHECK-NEXT:        "symConsts": {
// CHECK-NEXT:          "depth": {
// CHECK-NEXT:            "type": "ui32",
// CHECK-NEXT:            "value": 5
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    ],

// CHECK-LABEL:   "types": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "arg",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hwBitwidth": 16,
// CHECK-NEXT:              "id": "!esi.channel<i16>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hwBitwidth": 16,
// CHECK-NEXT:                "id": "i16",
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "from",
// CHECK-NEXT:            "name": "result",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hwBitwidth": 16,
// CHECK-NEXT:              "id": "!esi.channel<i16>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hwBitwidth": 16,
// CHECK-NEXT:                "id": "i16",
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "hwBitwidth": 36,
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<i16> to \"arg\", !esi.channel<i16> from \"result\"]>",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "recv",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hwBitwidth": 8,
// CHECK-NEXT:              "id": "!esi.channel<i8>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hwBitwidth": 8,
// CHECK-NEXT:                "id": "i8",
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "hwBitwidth": 10,
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "from",
// CHECK-NEXT:            "name": "send",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hwBitwidth": 8,
// CHECK-NEXT:              "id": "!esi.channel<i8>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hwBitwidth": 8,
// CHECK-NEXT:                "id": "i8",
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "hwBitwidth": 10,
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<i8> from \"send\"]>",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "from",
// CHECK-NEXT:            "name": "send",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hwBitwidth": 0,
// CHECK-NEXT:              "id": "!esi.channel<i0>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hwBitwidth": 0,
// CHECK-NEXT:                "id": "i0",
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "hwBitwidth": 2,
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<i0> from \"send\"]>",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "window_in",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hwBitwidth": 16,
// CHECK-NEXT:              "id": "!esi.channel<!esi.window<\"TestWindow\", !hw.struct<header: i8, data: i16>, [<\"HeaderFrame\", [<\"header\">]>, <\"DataFrame\", [<\"data\">]>]>>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "esi",
// CHECK-NEXT:                "frames": [
// CHECK-NEXT:                  {
// CHECK-NEXT:                    "fields": [
// CHECK-NEXT:                      {
// CHECK-NEXT:                        "name": "header"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    ],
// CHECK-NEXT:                    "name": "HeaderFrame"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  {
// CHECK-NEXT:                    "fields": [
// CHECK-NEXT:                      {
// CHECK-NEXT:                        "name": "data"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    ],
// CHECK-NEXT:                    "name": "DataFrame"
// CHECK-NEXT:                  }
// CHECK-NEXT:                ],
// CHECK-NEXT:                "hwBitwidth": 16,
// CHECK-NEXT:                "id": "!esi.window<\"TestWindow\", !hw.struct<header: i8, data: i16>, [<\"HeaderFrame\", [<\"header\">]>, <\"DataFrame\", [<\"data\">]>]>",
// CHECK-NEXT:                "into": {
// CHECK-NEXT:                  "dialect": "hw",
// CHECK-NEXT:                  "fields": [
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "header",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "builtin",
// CHECK-NEXT:                        "hwBitwidth": 8,
// CHECK-NEXT:                        "id": "i8",
// CHECK-NEXT:                        "mnemonic": "int",
// CHECK-NEXT:                        "signedness": "signless"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "data",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "builtin",
// CHECK-NEXT:                        "hwBitwidth": 16,
// CHECK-NEXT:                        "id": "i16",
// CHECK-NEXT:                        "mnemonic": "int",
// CHECK-NEXT:                        "signedness": "signless"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    }
// CHECK-NEXT:                  ],
// CHECK-NEXT:                  "hwBitwidth": 24,
// CHECK-NEXT:                  "id": "!hw.struct<header: i8, data: i16>",
// CHECK-NEXT:                  "mnemonic": "struct"
// CHECK-NEXT:                },
// CHECK-NEXT:                "loweredType": {
// CHECK-NEXT:                  "dialect": "hw",
// CHECK-NEXT:                  "fields": [
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "HeaderFrame",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "hw",
// CHECK-NEXT:                        "fields": [
// CHECK-NEXT:                          {
// CHECK-NEXT:                            "name": "header",
// CHECK-NEXT:                            "type": {
// CHECK-NEXT:                              "dialect": "builtin",
// CHECK-NEXT:                              "hwBitwidth": 8,
// CHECK-NEXT:                              "id": "i8",
// CHECK-NEXT:                              "mnemonic": "int",
// CHECK-NEXT:                              "signedness": "signless"
// CHECK-NEXT:                            }
// CHECK-NEXT:                          }
// CHECK-NEXT:                        ],
// CHECK-NEXT:                        "hwBitwidth": 8,
// CHECK-NEXT:                        "id": "!hw.struct<header: i8>",
// CHECK-NEXT:                        "mnemonic": "struct"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "DataFrame",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "hw",
// CHECK-NEXT:                        "fields": [
// CHECK-NEXT:                         {
// CHECK-NEXT:                           "name": "data",
// CHECK-NEXT:                           "type": {
// CHECK-NEXT:                             "dialect": "builtin",
// CHECK-NEXT:                             "hwBitwidth": 16,
// CHECK-NEXT:                             "id": "i16",
// CHECK-NEXT:                             "mnemonic": "int",
// CHECK-NEXT:                             "signedness": "signless"
// CHECK-NEXT:                           }
// CHECK-NEXT:                         }
// CHECK-NEXT:                       ],
// CHECK-NEXT:                       "hwBitwidth": 16,
// CHECK-NEXT:                       "id": "!hw.struct<data: i16>",
// CHECK-NEXT:                       "mnemonic": "struct"
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 ],
// CHECK-NEXT:                 "hwBitwidth": 16,
// CHECK-NEXT:                 "id": "!hw.union<HeaderFrame: !hw.struct<header: i8>, DataFrame: !hw.struct<data: i16>>",
// CHECK-NEXT:                 "mnemonic": "union"
// CHECK-NEXT:               },
// CHECK-NEXT:               "mnemonic": "window",
// CHECK-NEXT:               "name": "TestWindow"
// CHECK-NEXT:               },
// CHECK-NEXT:               "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "hwBitwidth": 18,
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<!esi.window<\"TestWindow\", !hw.struct<header: i8, data: i16>, [<\"HeaderFrame\", [<\"header\">]>, <\"DataFrame\", [<\"data\">]>]>> to \"window_in\"]>",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },

// Check for list window bundle type with countWidth and numItems
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "bulk_in",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hwBitwidth": 256,
// CHECK-NEXT:              "id": "!esi.channel<!esi.window<\"BulkTransferWindow\", !hw.struct<address: i32, data: !esi.list<i64>>, [<\"HeaderFrame\", [<\"address\">, <\"data\" countWidth 8>]>, <\"DataFrame\", [<\"data\", 4>]>]>>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "esi",
// CHECK-NEXT:                "frames": [
// CHECK-NEXT:                  {
// CHECK-NEXT:                    "fields": [
// CHECK-NEXT:                      {
// CHECK-NEXT:                        "name": "address"
// CHECK-NEXT:                      },
// CHECK-NEXT:                      {
// CHECK-NEXT:                        "bulkCountWidth": 8,
// CHECK-NEXT:                        "name": "data"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    ],
// CHECK-NEXT:                    "name": "HeaderFrame"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  {
// CHECK-NEXT:                    "fields": [
// CHECK-NEXT:                      {
// CHECK-NEXT:                        "name": "data",
// CHECK-NEXT:                        "numItems": 4
// CHECK-NEXT:                      }
// CHECK-NEXT:                    ],
// CHECK-NEXT:                    "name": "DataFrame"
// CHECK-NEXT:                  }
// CHECK-NEXT:                ],
// CHECK-NEXT:                "hwBitwidth": 256,
// CHECK-NEXT:                "id": "!esi.window<\"BulkTransferWindow\", !hw.struct<address: i32, data: !esi.list<i64>>, [<\"HeaderFrame\", [<\"address\">, <\"data\" countWidth 8>]>, <\"DataFrame\", [<\"data\", 4>]>]>",
// CHECK-NEXT:                "into": {
// CHECK-NEXT:                  "dialect": "hw",
// CHECK-NEXT:                  "fields": [
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "address",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "builtin",
// CHECK-NEXT:                        "hwBitwidth": 32,
// CHECK-NEXT:                        "id": "i32",
// CHECK-NEXT:                        "mnemonic": "int",
// CHECK-NEXT:                        "signedness": "signless"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "data",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "esi",
// CHECK-NEXT:                        "element": {
// CHECK-NEXT:                          "dialect": "builtin",
// CHECK-NEXT:                          "hwBitwidth": 64,
// CHECK-NEXT:                          "id": "i64",
// CHECK-NEXT:                          "mnemonic": "int",
// CHECK-NEXT:                          "signedness": "signless"
// CHECK-NEXT:                        },
// CHECK-NEXT:                        "id": "!esi.list<i64>",
// CHECK-NEXT:                        "mnemonic": "list"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    }
// CHECK-NEXT:                  ],
// CHECK-NEXT:                  "id": "!hw.struct<address: i32, data: !esi.list<i64>>",
// CHECK-NEXT:                  "mnemonic": "struct"
// CHECK-NEXT:                },
// CHECK-NEXT:                "loweredType": {
// CHECK-NEXT:                  "dialect": "hw",
// CHECK-NEXT:                  "fields": [
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "HeaderFrame",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "hw",
// CHECK-NEXT:                        "fields": [
// CHECK-NEXT:                          {
// CHECK-NEXT:                            "name": "address",
// CHECK-NEXT:                            "type": {
// CHECK-NEXT:                              "dialect": "builtin",
// CHECK-NEXT:                              "hwBitwidth": 32,
// CHECK-NEXT:                              "id": "i32",
// CHECK-NEXT:                              "mnemonic": "int",
// CHECK-NEXT:                              "signedness": "signless"
// CHECK-NEXT:                            }
// CHECK-NEXT:                          },
// CHECK-NEXT:                          {
// CHECK-NEXT:                            "name": "data_count",
// CHECK-NEXT:                            "type": {
// CHECK-NEXT:                              "dialect": "builtin",
// CHECK-NEXT:                              "hwBitwidth": 8,
// CHECK-NEXT:                              "id": "i8",
// CHECK-NEXT:                              "mnemonic": "int",
// CHECK-NEXT:                              "signedness": "signless"
// CHECK-NEXT:                            }
// CHECK-NEXT:                          }
// CHECK-NEXT:                        ],
// CHECK-NEXT:                        "hwBitwidth": 40,
// CHECK-NEXT:                        "id": "!hw.struct<address: i32, data_count: i8>",
// CHECK-NEXT:                        "mnemonic": "struct"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    {
// CHECK-NEXT:                      "name": "DataFrame",
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                        "dialect": "hw",
// CHECK-NEXT:                        "fields": [
// CHECK-NEXT:                          {
// CHECK-NEXT:                            "name": "data",
// CHECK-NEXT:                            "type": {
// CHECK-NEXT:                              "dialect": "hw",
// CHECK-NEXT:                              "element": {
// CHECK-NEXT:                                "dialect": "builtin",
// CHECK-NEXT:                                "hwBitwidth": 64,
// CHECK-NEXT:                                "id": "i64",
// CHECK-NEXT:                                "mnemonic": "int",
// CHECK-NEXT:                                "signedness": "signless"
// CHECK-NEXT:                              },
// CHECK-NEXT:                              "hwBitwidth": 256,
// CHECK-NEXT:                              "id": "!hw.array<4xi64>",
// CHECK-NEXT:                              "mnemonic": "array",
// CHECK-NEXT:                              "size": 4
// CHECK-NEXT:                            }
// CHECK-NEXT:                          }
// CHECK-NEXT:                        ],
// CHECK-NEXT:                        "hwBitwidth": 256,
// CHECK-NEXT:                        "id": "!hw.struct<data: !hw.array<4xi64>>",
// CHECK-NEXT:                        "mnemonic": "struct"
// CHECK-NEXT:                      }
// CHECK-NEXT:                    }
// CHECK-NEXT:                  ],
// CHECK-NEXT:                  "hwBitwidth": 256,
// CHECK-NEXT:                  "id": "!hw.union<HeaderFrame: !hw.struct<address: i32, data_count: i8>, DataFrame: !hw.struct<data: !hw.array<4xi64>>>",
// CHECK-NEXT:                  "mnemonic": "union"
// CHECK-NEXT:                },
// CHECK-NEXT:                "mnemonic": "window",
// CHECK-NEXT:                "name": "BulkTransferWindow"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "hwBitwidth": 258,
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<!esi.window<\"BulkTransferWindow\", !hw.struct<address: i32, data: !esi.list<i64>>, [<\"HeaderFrame\", [<\"address\">, <\"data\" countWidth 8>]>, <\"DataFrame\", [<\"data\", 4>]>]>> to \"bulk_in\"]>",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "arg",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "id": "!esi.channel<!esi.any>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "esi",
// CHECK-NEXT:                "id": "!esi.any",
// CHECK-NEXT:                "mnemonic": "any"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "from",
// CHECK-NEXT:            "name": "result",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "id": "!esi.channel<!esi.any>",
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "dialect": "esi",
// CHECK-NEXT:                "id": "!esi.any",
// CHECK-NEXT:                "mnemonic": "any"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<!esi.any> to \"arg\", !esi.channel<!esi.any> from \"result\"]>",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "dialect": "builtin",
// CHECK-NEXT:        "hwBitwidth": 64,
// CHECK-NEXT:        "id": "i64",
// CHECK-NEXT:        "mnemonic": "int",
// CHECK-NEXT:        "signedness": "signless"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "dialect": "builtin",
// CHECK-NEXT:        "hwBitwidth": 32,
// CHECK-NEXT:        "id": "ui32",
// CHECK-NEXT:        "mnemonic": "int",
// CHECK-NEXT:        "signedness": "unsigned"
// CHECK-NEXT:      }
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
