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

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[0]>}
  hw.instance "m2" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[1]>}
  hw.instance "func1" @CallableFunc1() -> ()
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
// HW:            hw.instance "__manifest" @__ESIManifest() -> ()
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
// CHECK-NEXT:        "id": "!esi.bundle<[!esi.channel<i0> from \"send\"]>",
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
