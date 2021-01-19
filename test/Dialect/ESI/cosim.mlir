// REQUIRES: capnp
// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-ports --lower-esi-to-rtl -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=COSIM %s
// RUN: circt-opt %s --lower-esi-ports --lower-esi-to-rtl | circt-translate --emit-verilog | FileCheck --check-prefix=SV %s
// RUN: circt-translate %s -emit-esi-capnp -verify-diagnostics | FileCheck --check-prefix=CAPNP %s

rtl.externmodule @Sender() -> ( !esi.channel<si14> { rtl.name = "x"})
rtl.externmodule @Reciever(%a: !esi.channel<i32>)

// CHECK-LABEL: rtl.externmodule @Sender() -> (%x: !esi.channel<si14>)
// CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i32> {rtl.name = "a"})

rtl.module @top(%clk:i1, %rstn:i1) -> () {
  rtl.instance "recv" @Reciever (%cosimRecv) : (!esi.channel<i32>) -> ()
  // CHECK:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i32>) -> ()

  %send.x = rtl.instance "send" @Sender () : () -> (!esi.channel<si14>)
  // CHECK:  %send.x = rtl.instance "send" @Sender() : () -> !esi.channel<si14>

  %cosimRecv = esi.cosim %clk, %rstn, %send.x, 1 {name="TestEP"} : !esi.channel<si14> -> !esi.channel<i32>
  // CHECK:  %0 = esi.cosim %clk, %rstn, %send.x, 1 {name = "TestEP"} : !esi.channel<si14> -> !esi.channel<i32>

  // Ensure that the file hash is deterministic.
  // CAPNP: 0xc6085f4b7a4688f8;
  // CAPNP-LABEL: struct Si14 @0x9bd5e507cce05cc1
  // CAPNP:         i @0 :Int16;
  // CAPNP-LABEL: struct I32 @0x92cd59dfefaacbdb
  // CAPNP:         i @0 :UInt32;
  // Ensure the standard RPC interface is tacked on.
  // CAPNP: interface CosimDpiServer
  // CAPNP: list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
  // CAPNP: open @1 [S, T] (iface :EsiDpiInterfaceDesc) -> (iface :EsiDpiEndpoint(S, T));

  // COSIM: rtl.instance "TestEP" @Cosim_Endpoint(%clk, %rstn, %{{.+}}, %{{.+}}, %{{.+}}) {parameters = {ENDPOINT_ID = 1 : i32, RECV_TYPE_ID = 10578209918096690139 : ui64, RECV_TYPE_SIZE_BITS = 128 : i32, SEND_TYPE_ID = 11229133067582987457 : ui64, SEND_TYPE_SIZE_BITS = 128 : i32}} : (i1, i1, i1, i1, i128) -> (i1, !rtl.array<128xi1>, i1)

  // SV: assign _T.valid = TestEP_DataOutValid;
  // SV: assign _T.data = dataSection[_T_3+:32];
  // SV: Reciever recv (
  // SV:   .a ({{.+}}.source)
  // SV: );
  // SV: {{.+}}.ready = TestEP_DataInReady;
  // SV: Sender send (
  // SV:   .x ({{.+}}.sink)
  // SV: );
  // SV: Cosim_Endpoint #(
  // SV:   .ENDPOINT_ID(32'd1),
  // SV:   .RECV_TYPE_ID(64'd10578209918096690139),
  // SV:   .RECV_TYPE_SIZE_BITS(32'd128),
  // SV:   .SEND_TYPE_ID(64'd11229133067582987457),
  // SV:   .SEND_TYPE_SIZE_BITS(32'd128)
  // SV: ) TestEP (
  // SV:   .clk (clk),
  // SV:   .rstn (rstn),
  // SV:   .DataOutReady ({{.+}}.ready),
  // SV:   .DataInValid ({{.+}}.valid),
  // SV:   .DataIn ({{[{]}}{50'h0, {{.+}}}, {16'h0, 16'h1, 32'h0}}),
  // SV:   .DataOutValid (TestEP_DataOutValid),
  // SV:   .DataOut (TestEP_DataOut),
  // SV:   .DataInReady (TestEP_DataInReady)
  // SV: );
  // SV: rootPointer = TestEP_DataOut[{{.+}}+:64];
  // SV: dataSection = TestEP_DataOut[{{.+}}+:64];
}
