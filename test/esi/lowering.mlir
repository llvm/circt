// RUN: circt-opt %s --lower-esi-to-physical -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-to-rtl -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=RTL %s

module {
  rtl.externmodule @Sender() -> ( %x: !esi.channel<i4> )
  rtl.externmodule @Reciever(%a: !esi.channel<i4>)

  // CHECK-LABEL: rtl.externmodule @Sender() -> (%x: !esi.channel<i4>)
  // CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i4> {rtl.name = "a"})

  // RTL-NOT: esi.stage

  rtl.module @test(%clk:i1, %rstn:i1) {
    %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i4>)
    %bufferedChan = esi.buffer %clk, %rstn, %esiChan { } : i4
    rtl.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i4>) -> ()

    // CHECK:  %sender.x = rtl.instance "sender" @Sender()  : () -> !esi.channel<i4>
    // CHECK-NEXT:  %0 = esi.stage %clk, %rstn, %sender.x : i4
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i4>) -> ()

    %esiChan2 = rtl.instance "sender2" @Sender () : () -> (!esi.channel<i4>)
    %bufferedChan2 = esi.buffer %clk, %rstn, %esiChan2 { stages = 4 } : i4
    rtl.instance "recv2" @Reciever (%bufferedChan2) : (!esi.channel<i4>) -> ()

    // CHECK-NEXT:  %sender2.x = rtl.instance "sender2" @Sender()  : () -> !esi.channel<i4>
    // CHECK-NEXT:  %1 = esi.stage %clk, %rstn, %sender2.x : i4
    // CHECK-NEXT:  %2 = esi.stage %clk, %rstn, %1 : i4
    // CHECK-NEXT:  %3 = esi.stage %clk, %rstn, %2 : i4
    // CHECK-NEXT:  %4 = esi.stage %clk, %rstn, %3 : i4
    // CHECK-NEXT:  rtl.instance "recv2" @Reciever(%4)  : (!esi.channel<i4>) -> ()
  }
}
