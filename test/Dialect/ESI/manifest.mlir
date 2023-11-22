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

esi.manifest.sym @Loopback name "LoopbackIP" version "v0.0" summary "IP which simply echos bytes" {foo=1}

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[0]>}
  hw.instance "m2" @Loopback (clk: %clk: !seq.clock) -> () {esi.appid=#esi.appid<"loopback_inst"[1]>}
}

// HIER-LABEL:  esi.manifest.compressed <"{{.+}}">
// HIER-LABEL:  esi.manifest.hier_root @top {
// HIER:          esi.manifest.service_impl #esi.appid<"cosim"> svc @HostComms by "cosim" with {} {
// HIER:            esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_tohw">] req <@HostComms::@Recv>(!esi.bundle<[!esi.channel<i8> to "recv"]>) with {channel_assignments = {recv = "loopback_inst[0].loopback_tohw.recv"}}
// HIER:            esi.manifest.impl_conn [#esi.appid<"loopback_inst"[0]>, #esi.appid<"loopback_fromhw">] req <@HostComms::@Send>(!esi.bundle<[!esi.channel<i8> from "send"]>) with {channel_assignments = {send = "loopback_inst[0].loopback_fromhw.send"}}
// HIER:            esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_tohw">] req <@HostComms::@Recv>(!esi.bundle<[!esi.channel<i8> to "recv"]>) with {channel_assignments = {recv = "loopback_inst[1].loopback_tohw.recv"}}
// HIER:            esi.manifest.impl_conn [#esi.appid<"loopback_inst"[1]>, #esi.appid<"loopback_fromhw">] req <@HostComms::@Send>(!esi.bundle<[!esi.channel<i8> from "send"]>) with {channel_assignments = {send = "loopback_inst[1].loopback_fromhw.send"}}
// HIER:          }
// HIER:          esi.manifest.hier_node #esi.appid<"loopback_inst"[0]> mod @Loopback {
// HIER:            esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, toClient, !esi.bundle<[!esi.channel<i8> to "recv"]>
// HIER:            esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, toServer, !esi.bundle<[!esi.channel<i8> to "send"]>
// HIER:          }
// HIER:          esi.manifest.hier_node #esi.appid<"loopback_inst"[1]> mod @Loopback {
// HIER:            esi.manifest.req #esi.appid<"loopback_tohw">, <@HostComms::@Recv>, toClient, !esi.bundle<[!esi.channel<i8> to "recv"]>
// HIER:            esi.manifest.req #esi.appid<"loopback_fromhw">, <@HostComms::@Send>, toServer, !esi.bundle<[!esi.channel<i8> to "send"]>
// HIER:          }
// HIER:        }

// HW-LABEL:    hw.module @top
// HW:            hw.instance "__manifest" @Cosim_Manifest<COMPRESSED_MANIFEST_SIZE: i32 = {{.+}}>(compressed_manifest: %{{.+}}: !hw.uarray<{{.+}}xi8>) -> ()
// HW-LABEL:    hw.module.extern @Cosim_Manifest<COMPRESSED_MANIFEST_SIZE: i32>(in %compressed_manifest : !hw.uarray<#hw.param.decl.ref<"COMPRESSED_MANIFEST_SIZE">xi8>) attributes {verilogName = "Cosim_Manifest"}

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
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
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
// CHECK-NEXT:            "name": "Recv",
// CHECK-NEXT:            "direction": "toClient",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:              "circt_name": "!esi.bundle<[!esi.channel<i8> to \"recv\"]>"
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
// CHECK-NEXT:      }
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
