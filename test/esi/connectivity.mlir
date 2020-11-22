// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @Sender() -> ( !esi.channel<i1> { rtl.name = "x"}) {
    %0 = constant 0 : i1
    %1 = esi.wrap %0 : i1 -> !esi.channel<i1>
    rtl.output %1 : !esi.channel<i1>
  }
  rtl.module @Reciever(%a: !esi.channel<i1>) {
    %0 = esi.unwrap %a : !esi.channel<i1> -> i1
  }

  // CHECK-LABEL: rtl.module @Sender() -> (!esi.channel<i1> {rtl.name = "x"}) {
  // CHECK:         %0 = esi.wrap %false : i1 -> !esi.channel<i1>
  // CHECK-LABEL: rtl.module @Reciever(%arg0: !esi.channel<i1> {rtl.name = "a"}) {
  // CHECK:         %0 = esi.unwrap %arg0 : !esi.channel<i1> -> i1

  rtl.module @test() {
    %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan = esi.buffer %esiChan { } : i1
    rtl.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // CHECK:  %esiChan = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %0 = esi.buffer %esiChan {} : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

    %esiChan2 = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan2 = esi.buffer %esiChan2 { stages = 4 } : i1
    rtl.instance "recv" @Reciever (%bufferedChan2) : (!esi.channel<i1>) -> ()

    // CHECK-NEXT:  %esiChan2 = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %1 = esi.buffer %esiChan2 {stages = 4 : i64} : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%1)  : (!esi.channel<i1>) -> ()
  }
}
