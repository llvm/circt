// REQUIRES: capnp
// RUN: circt-opt --esi-connect-services --esi-emit-collateral=tops=LoopbackCosimTopWrapper --lower-msft-to-hw --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw  --export-verilog %s -o %t1.mlir  | FileCheck %s

// CHECK-LABEL: // ----- 8< ----- FILE "services.json" ----- 8< -----

// CHECK-LABEL: "declarations": [ 

// CHECK-LABEL:    "name": "HostComms" 
// CHECK:          "ports": [
// CHECK:            [
// CHECK:              {
// CHECK:                "name": "Send",
// CHECK:                "to-server-type": {
// CHECK:                  "dialect": "esi",
// CHECK:                  "inner": {
// CHECK:                    "dialect": "esi",
// CHECK:                    "mnemonic": "any"
// CHECK:                  },
// CHECK:                  "mnemonic": "channel"
// CHECK:                }
// CHECK:              },
// CHECK:              {
// CHECK:                "name": "Recv",
// CHECK:                "to-client-type": {
// CHECK:                  "dialect": "esi",
// CHECK:                  "inner": {
// CHECK:                    "dialect": "builtin",
// CHECK:                    "mnemonic": "i8"
// CHECK:                  },
// CHECK:                  "mnemonic": "channel"
// CHECK:                }
// CHECK:              },
// CHECK:              {
// CHECK:                "name": "ReqResp",
// CHECK:                "to-client-type": {
// CHECK:                  "dialect": "esi",
// CHECK:                  "inner": {
// CHECK:                    "dialect": "builtin",
// CHECK:                    "mnemonic": "i16"
// CHECK:                  },
// CHECK:                  "mnemonic": "channel"
// CHECK:                },
// CHECK:                "to-server-type": {
// CHECK:                  "dialect": "esi",
// CHECK:                  "inner": {
// CHECK:                    "dialect": "builtin",
// CHECK:                    "mnemonic": "i8"
// CHECK:                  },
// CHECK:                  "mnemonic": "channel"
// CHECK:                }
// CHECK:              }
// CHECK:            ]
esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
  esi.service.inout @ReqResp : !esi.channel<i8> -> !esi.channel<i16>
}

msft.module @InOutLoopback {} (%clk: i1) -> () {
  %dataIn = esi.service.req.inout %dataTrunc -> <@HostComms::@ReqResp> (["loopback_inout"]) : !esi.channel<i8> -> !esi.channel<i16>
  %unwrap, %valid = esi.unwrap.vr %dataIn, %rdy: i16
  %trunc = comb.extract %unwrap from 0 : (i16) -> (i8)
  %dataTrunc, %rdy = esi.wrap.vr %trunc, %valid : i8
  msft.output
}

msft.module @LoopbackCosimTop {} (%clk: i1, %rst: i1) {
  esi.service.instance @HostComms impl as "cosim" (%clk, %rst) : (i1, i1) -> ()
  msft.instance @m1 @InOutLoopback(%clk) : (i1) -> ()
  msft.output
}

// CHECK-LABEL: "top_levels": [

// CHECK-LABEL: "module": "@LoopbackCosimTopWrapper",
// CHECK:       "services": [
// CHECK:         {
// CHECK:           "service": "HostComms",
// CHECK:           "path": [
// CHECK:             "top"
// CHECK:           ],
// CHECK:           "impl_type": "cosim",
// CHECK:           "clients": [
// CHECK:             {
// CHECK:               "client_name": [
// CHECK:                 "m1",
// CHECK:                 "loopback_inout"
// CHECK:               ],
// CHECK:               "port": "#hw.innerNameRef<@HostComms::@ReqResp>",
// CHECK:               "to_client_type": {
// CHECK:                 "dialect": "esi",
// CHECK:                 "inner": {
// CHECK:                   "dialect": "builtin",
// CHECK:                   "mnemonic": "i16"
// CHECK:                 },
// CHECK:                 "mnemonic": "channel"
// CHECK:               }
// CHECK:             },
// CHECK:             {
// CHECK:               "client_name": [
// CHECK:                 "m1",
// CHECK:                 "loopback_inout"
// CHECK:               ],
// CHECK:               "port": "#hw.innerNameRef<@HostComms::@ReqResp>",
// CHECK:               "to_server_type": {
// CHECK:                 "dialect": "esi",
// CHECK:                 "inner": {
// CHECK:                   "dialect": "builtin",
// CHECK:                   "mnemonic": "i8"
// CHECK:                 },
// CHECK:                 "mnemonic": "channel"


msft.module @LoopbackCosimTopWrapper {} (%clk: i1, %rst: i1) {
  msft.instance @top @LoopbackCosimTop(%clk, %rst) : (i1, i1) -> ()
  msft.output
}
