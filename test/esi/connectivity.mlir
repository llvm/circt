// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @Sender() -> (%x: !esi.channel<i1>) {
    %0 = constant 0 : i1
    // Don't transmit any data.
    %ch, %rcvrRdy = esi.wrap.vr %0, %0 : i1
    rtl.output %ch : !esi.channel<i1>
  }
  rtl.module @Reciever(%a: !esi.channel<i1>) {
    %rdy = constant 1 : i1
    // Recieve bits.
    %data, %valid = esi.unwrap.vr %a, %rdy : i1
  }

  // CHECK-LABEL: rtl.module @Sender() -> (%x: !esi.channel<i1>) {
  // CHECK:        %output, %ready = esi.wrap.vr %false, %false : i1
  // CHECK-LABEL: rtl.module @Reciever(%a: !esi.channel<i1>) {
  // CHECK:        %output, %valid = esi.unwrap.vr %a, %true : i1

  rtl.module @test() {
    %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan = esi.buffer %esiChan { } : i1
    rtl.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // CHECK:  %sender.x = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %0 = esi.buffer %sender.x {} : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

    %esiChan2 = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan2 = esi.buffer %esiChan2 { stages = 4 } : i1
    rtl.instance "recv" @Reciever (%bufferedChan2) : (!esi.channel<i1>) -> ()

    // CHECK-NEXT:  %sender.x_0 = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %1 = esi.buffer %sender.x_0 {stages = 4 : i64} : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%1)  : (!esi.channel<i1>) -> ()
  }
}
