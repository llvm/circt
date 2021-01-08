// REQUIRES: capnp
// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-to-rtl -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=COSIM %s
// RUN: circt-translate %s -emit-esi-capnp -verify-diagnostics | FileCheck --check-prefix=CAPNP %s

module {
  rtl.externmodule @Sender() -> ( !esi.channel<si14> { rtl.name = "x"})
  rtl.externmodule @Reciever(%a: !esi.channel<i32>)

  // CHECK-LABEL: rtl.externmodule @Sender() -> (%x: !esi.channel<si14>)
  // CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i32> {rtl.name = "a"})

  rtl.module @Top(%clk:i1, %rstn:i1) -> () {
    rtl.instance "recv" @Reciever (%cosimRecv) : (!esi.channel<i32>) -> ()
    // CHECK:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i32>) -> ()

    %send.x = rtl.instance "send" @Sender () : () -> (!esi.channel<si14>)
    // CHECK:  %send.x = rtl.instance "send" @Sender() : () -> !esi.channel<si14>

    %cosimRecv = esi.cosim %clk, %rstn, %send.x, 1 {name="TestEP"} : !esi.channel<si14> -> !esi.channel<i32>
    // CHECK:  %0 = esi.cosim %clk, %rstn, %send.x, 1 {name = "TestEP"} : !esi.channel<si14> -> !esi.channel<i32>

    // Ensure that the file hash is deterministic.
    // CAPNP: @0xdc4810c19280c1d2;
    // CAPNP-LABEL: struct TYi32 @0xc24ad8e97bb0ff57
    // CAPNP:         i @0 :UInt32;
    // CAPNP-LABEL: struct TYsi14 @0x8ee0bd493e80e8a1
    // CAPNP:         i @0 :Int16;

    // COSIM: %rawOutput, %valid = esi.unwrap.vr %send.x, %TestEP.DataInReady : si14
    // COSIM: %0 = esi.encode.capnp %rawOutput : si14 -> !rtl.array<192xi1>
    // COSIM: %TestEP.DataOutValid, %TestEP.DataOut, %TestEP.DataInReady = rtl.instance "TestEP" @Cosim_Endpoint(%clk, %rstn, %ready, %valid, %0) {parameters = {ENDPOINT_ID = 1 : i32, RECVTYPE_SIZE_BITS = 192 : i64, RECV_TYPE_ID = 14000240888948784983 : ui64, SEND_TYPE_ID = 10295436870447851681 : ui64, SEND_TYPE_SIZE_BITS = 192 : i64}} : (i1, i1, i1, i1, !rtl.array<192xi1>) -> (i1, !rtl.array<192xi1>, i1)
    // COSIM: %1 = esi.decode.capnp %TestEP.DataOut : !rtl.array<192xi1> -> !esi.channel<i32>
  }
}
