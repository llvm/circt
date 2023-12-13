// REQUIRES: zlib
// RUN: circt-opt %s --esi-connect-services --esi-appid-hier=top=top --esi-build-manifest="top=top" > %t1.mlir 
// RUN: circt-opt %t1.mlir | FileCheck --check-prefix=HIER %s
// RUN: FileCheck --input-file=esi_system_manifest.json %s
// RUN: circt-opt %t1.mlir --esi-clean-metadata --lower-esi-bundles --lower-esi-ports --lower-esi-to-hw=platform=cosim | FileCheck --check-prefix=HW %s

hw.type_scope @__hw_typedecls {
  hw.typedecl @foo, "Foo" : i1
}
!alias = !hw.typealias<@__hw_typedecls::@foo, i1>

!sendI8 = !esi.bundle<[!esi.channel<i8> to "send"]>
!recvI8 = !esi.bundle<[!esi.channel<i8> to "recv"]>
!sendI0 = !esi.bundle<[!esi.channel<i0> to "send"]>

esi.service.decl @HostComms {
  esi.service.to_server @Send : !sendI8
  esi.service.to_client @Recv : !recvI8
  esi.service.to_server @SendI0 : !sendI0
}

hw.module @Loopback (in %clk: !seq.clock) {
  %dataInBundle = esi.service.req.to_client <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) {esi.appid=#esi.appid<"loopback_tohw">} : !recvI8
  %dataOut = esi.bundle.unpack from %dataInBundle : !recvI8
  %dataOutBundle = esi.bundle.pack %dataOut : !sendI8
  esi.service.req.to_server %dataOutBundle -> <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !sendI8

  %c0_0 = hw.constant 0 : i0
  %c0_1 = hw.constant 0 : i1
  %sendi0_channel, %ready = esi.wrap.vr %c0_0, %c0_1 : i0
  %sendi0_bundle = esi.bundle.pack %sendi0_channel : !sendI0
  esi.service.req.to_server %sendi0_bundle -> <@HostComms::@SendI0> (#esi.appid<"loopback_fromhw_i0">) : !sendI0
}

esi.manifest.sym @Loopback name "LoopbackIP" version "v0.0" summary "IP which simply echos bytes" {foo=1}

esi.service.std.func @funcs

// CONN-LABEL:   hw.module @CallableFunc1(in %func1 : !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>) {
// CONN-NEXT:      esi.manifest.req #esi.appid<"func1">, <@funcs::@call> std "esi.service.std.func", toClient, !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
// CONN-NEXT:      %arg = esi.bundle.unpack %arg from %func1 : !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
!func1Signature = !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
hw.module @CallableFunc1() {
  %call = esi.service.req.to_client <@funcs::@call> (#esi.appid<"func1">) : !func1Signature
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
// HIER-NEXT:     esi.manifest.service_impl #esi.appid<"cosim"> svc @HostComms by "cosim" with {} {
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_tohw">] req <@HostComms::@Recv>(!esi.bundle<[!esi.channel<i8> to "recv"]>) with {channel_assignments = {recv = "loopback_inst[0].loopback_tohw.recv"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_fromhw">] req <@HostComms::@Send>(!esi.bundle<[!esi.channel<i8> from "send"]>) with {channel_assignments = {send = "loopback_inst[0].loopback_fromhw.send"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_fromhw_i0">] req <@HostComms::@SendI0>(!esi.bundle<[!esi.channel<i0> from "send"]>) with {channel_assignments = {send = "loopback_inst[0].loopback_fromhw_i0.send"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_tohw">] req <@HostComms::@Recv>(!esi.bundle<[!esi.channel<i8> to "recv"]>) with {channel_assignments = {recv = "loopback_inst[1].loopback_tohw.recv"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_fromhw">] req <@HostComms::@Send>(!esi.bundle<[!esi.channel<i8> from "send"]>) with {channel_assignments = {send = "loopback_inst[1].loopback_fromhw.send"}}
// HIER-NEXT:       esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_fromhw_i0">] req <@HostComms::@SendI0>(!esi.bundle<[!esi.channel<i0> from "send"]>) with {channel_assignments = {send = "loopback_inst[1].loopback_fromhw_i0.send"}}
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.hier_node #esi.appid<"loopback_inst"[0]> mod @Loopback {
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, toClient, !esi.bundle<[!esi.channel<i8> to "recv"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, toServer, !esi.bundle<[!esi.channel<i8> to "send"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw_i0">, <@HostComms::@SendI0>, toServer, !esi.bundle<[!esi.channel<i0> to "send"]>
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.hier_node #esi.appid<"loopback_inst"[1]> mod @Loopback {
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, toClient, !esi.bundle<[!esi.channel<i8> to "recv"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, toServer, !esi.bundle<[!esi.channel<i8> to "send"]>
// HIER-NEXT:       esi.manifest.req #esi.appid<"loopback_fromhw_i0">, <@HostComms::@SendI0>, toServer, !esi.bundle<[!esi.channel<i0> to "send"]>
// HIER-NEXT:     }
// HIER-NEXT:     esi.manifest.req #esi.appid<"func1">, <@funcs::@call> std "esi.service.std.func", toClient, !esi.bundle<[!esi.channel<i16> to "arg", !esi.channel<i16> from "result"]>
// HIER-NEXT:   }

// HW-LABEL:    hw.module @top
// HW:            hw.instance "__manifest" @__ESIManifest() -> ()
// HW-LABEL:    hw.module.extern @Cosim_Manifest<COMPRESSED_MANIFEST_SIZE: i32>(in %compressed_manifest : !hw.uarray<#hw.param.decl.ref<"COMPRESSED_MANIFEST_SIZE">xi8>) attributes {verilogName = "Cosim_Manifest"}
// HW-LABEL:    hw.module @__ESIManifest()
// HW:            hw.instance "__manifest" @Cosim_Manifest<COMPRESSED_MANIFEST_SIZE: i32 = {{.+}}>(compressed_manifest: %{{.+}}: !hw.uarray<{{.+}}xi8>) -> ()

// CHECK:       {
// CHECK-LABEL:   "api_version": 1,

// CHECK-LABEL:   "symbols": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "foo": 1,
// CHECK-NEXT:        "name": "LoopbackIP",
// CHECK-NEXT:        "summary": "IP which simply echos bytes",
// CHECK-NEXT:        "symbolRef": "@Loopback",
// CHECK-NEXT:        "version": "v0.0"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],

// CHECK-LABEL:   "design": {
// CHECK-NEXT:      "inst_of": "@top",
// CHECK-NEXT:      "contents": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "class": "service",
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "cosim"
// CHECK-NEXT:          },
// CHECK-NEXT:          "service": "@HostComms",
// CHECK-NEXT:          "serviceImplName": "cosim",
// CHECK-NEXT:          "client_details": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "channel_assignments": {
// CHECK-NEXT:                "recv": "loopback_inst[0].loopback_tohw.recv"
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
// CHECK-NEXT:                "inner": "Recv",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channel_assignments": {
// CHECK-NEXT:                "send": "loopback_inst[0].loopback_fromhw.send"
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
// CHECK-NEXT:                "inner": "Send",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channel_assignments": {
// CHECK-NEXT:                "send": "loopback_inst[0].loopback_fromhw_i0.send"
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
// CHECK-NEXT:                "inner": "SendI0",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channel_assignments": {
// CHECK-NEXT:                "recv": "loopback_inst[1].loopback_tohw.recv"
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
// CHECK-NEXT:                "inner": "Recv",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channel_assignments": {
// CHECK-NEXT:                "send": "loopback_inst[1].loopback_fromhw.send"
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
// CHECK-NEXT:                "inner": "Send",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "channel_assignments": {
// CHECK-NEXT:                "send": "loopback_inst[1].loopback_fromhw_i0.send"
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
// CHECK-NEXT:                "inner": "SendI0",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "class": "client_port",
// CHECK-NEXT:          "appID": {
// CHECK-NEXT:            "name": "func1"
// CHECK-NEXT:          },
// CHECK-NEXT:          "direction": "toClient",
// CHECK-NEXT:          "bundleType": {
// CHECK-NEXT:            "circt_name": "!esi.bundle<[!esi.channel<i16> to \"arg\", !esi.channel<i16> from \"result\"]>"
// CHECK-NEXT:          },
// CHECK-NEXT:          "servicePort": {
// CHECK-NEXT:            "inner": "call",
// CHECK-NEXT:            "outer_sym": "funcs"
// CHECK-NEXT:          },
// CHECK-NEXT:          "stdService": "esi.service.std.func"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "children": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "app_id": {
// CHECK-NEXT:            "index": 0,
// CHECK-NEXT:            "name": "loopback_inst"
// CHECK-NEXT:          },
// CHECK-NEXT:          "inst_of": "@Loopback",
// CHECK-NEXT:          "contents": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "class": "client_port",
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_tohw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "direction": "toClient",
// CHECK-NEXT:              "bundleType": {
// CHECK-NEXT:                "circt_name": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "inner": "Recv",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "class": "client_port",
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "direction": "toServer",
// CHECK-NEXT:              "bundleType": {
// CHECK-NEXT:                "circt_name": "!esi.bundle<[!esi.channel<i8> to \"send\"]>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "inner": "Send",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "class": "client_port",
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw_i0"
// CHECK-NEXT:              },
// CHECK-NEXT:              "direction": "toServer",
// CHECK-NEXT:              "bundleType": {
// CHECK-NEXT:                "circt_name": "!esi.bundle<[!esi.channel<i0> to \"send\"]>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "inner": "SendI0",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "children": []
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "app_id": {
// CHECK-NEXT:            "index": 1,
// CHECK-NEXT:            "name": "loopback_inst"
// CHECK-NEXT:          },
// CHECK-NEXT:          "inst_of": "@Loopback",
// CHECK-NEXT:          "contents": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "class": "client_port",
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_tohw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "direction": "toClient",
// CHECK-NEXT:              "bundleType": {
// CHECK-NEXT:                "circt_name": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "inner": "Recv",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "class": "client_port",
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw"
// CHECK-NEXT:              },
// CHECK-NEXT:              "direction": "toServer",
// CHECK-NEXT:              "bundleType": {
// CHECK-NEXT:                "circt_name": "!esi.bundle<[!esi.channel<i8> to \"send\"]>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "inner": "Send",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            },
// CHECK-NEXT:            {
// CHECK-NEXT:              "class": "client_port",
// CHECK-NEXT:              "appID": {
// CHECK-NEXT:                "name": "loopback_fromhw_i0"
// CHECK-NEXT:              },
// CHECK-NEXT:              "direction": "toServer",
// CHECK-NEXT:              "bundleType": {
// CHECK-NEXT:                "circt_name": "!esi.bundle<[!esi.channel<i0> to \"send\"]>"
// CHECK-NEXT:              },
// CHECK-NEXT:              "servicePort": {
// CHECK-NEXT:                "inner": "SendI0",
// CHECK-NEXT:                "outer_sym": "HostComms"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "children": []
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    },

// CHECK-LABEL:   "service_decls": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "HostComms",
// CHECK-NEXT:        "ports": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "Send",
// CHECK-NEXT:            "direction": "toServer",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.bundle<[!esi.channel<i8> to \"send\"]>"
// CHECK-NEXT:            }
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "SendI0",
// CHECK-NEXT:            "direction": "toServer",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.bundle<[!esi.channel<i0> to \"send\"]>"
// CHECK-NEXT:            }
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "Recv",
// CHECK-NEXT:            "direction": "toClient",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ]
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "symbol": "funcs",
// CHECK-NEXT:        "ports": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "name": "call",
// CHECK-NEXT:            "direction": "toClient",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "type": {
// CHECK-NEXT:                "channels": [
// CHECK-NEXT:                  {
// CHECK-NEXT:                    "direction": "to",
// CHECK-NEXT:                    "name": "arg",
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                      "circt_name": "!esi.channel<!esi.any>",
// CHECK-NEXT:                      "dialect": "esi",
// CHECK-NEXT:                      "inner": {
// CHECK-NEXT:                        "circt_name": "!esi.any",
// CHECK-NEXT:                        "dialect": "esi",
// CHECK-NEXT:                        "mnemonic": "any"
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "mnemonic": "channel"
// CHECK-NEXT:                    }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  {
// CHECK-NEXT:                    "direction": "from",
// CHECK-NEXT:                    "name": "result",
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                      "circt_name": "!esi.channel<!esi.any>",
// CHECK-NEXT:                      "dialect": "esi",
// CHECK-NEXT:                      "inner": {
// CHECK-NEXT:                        "circt_name": "!esi.any",
// CHECK-NEXT:                        "dialect": "esi",
// CHECK-NEXT:                        "mnemonic": "any"
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "mnemonic": "channel"
// CHECK-NEXT:                    }
// CHECK-NEXT:                  }
// CHECK-NEXT:                ],
// CHECK-NEXT:                "circt_name": "!esi.bundle<[!esi.channel<!esi.any> to \"arg\", !esi.channel<!esi.any> from \"result\"]>",
// CHECK-NEXT:                "dialect": "esi",
// CHECK-NEXT:                "mnemonic": "bundle"
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ]
// CHECK-NEXT:      }
// CHECK-NEXT:    ],

// CHECK-LABEL:   "types": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "recv",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.channel<i8>",
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hw_bitwidth": 8,
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "circt_name": "i8",
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hw_bitwidth": 8,
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "circt_name": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>",
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "send",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.channel<i8>",
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hw_bitwidth": 8,
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "circt_name": "i8",
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hw_bitwidth": 8,
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "circt_name": "!esi.bundle<[!esi.channel<i8> to \"send\"]>",
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "send",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.channel<i0>",
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hw_bitwidth": 0,
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "circt_name": "i0",
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hw_bitwidth": 0,
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "circt_name": "!esi.bundle<[!esi.channel<i0> to \"send\"]>",
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "channels": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "direction": "to",
// CHECK-NEXT:            "name": "arg",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.channel<i16>",
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hw_bitwidth": 16,
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "circt_name": "i16",
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hw_bitwidth": 16,
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
// CHECK-NEXT:              "circt_name": "!esi.channel<i16>",
// CHECK-NEXT:              "dialect": "esi",
// CHECK-NEXT:              "hw_bitwidth": 16,
// CHECK-NEXT:              "inner": {
// CHECK-NEXT:                "circt_name": "i16",
// CHECK-NEXT:                "dialect": "builtin",
// CHECK-NEXT:                "hw_bitwidth": 16,
// CHECK-NEXT:                "mnemonic": "int",
// CHECK-NEXT:                "signedness": "signless"
// CHECK-NEXT:              },
// CHECK-NEXT:              "mnemonic": "channel"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "circt_name": "!esi.bundle<[!esi.channel<i16> to \"arg\", !esi.channel<i16> from \"result\"]>",
// CHECK-NEXT:        "dialect": "esi",
// CHECK-NEXT:        "mnemonic": "bundle"
// CHECK-NEXT:      }
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
