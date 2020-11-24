// RUN: circt-opt %s --lower-esi-to-physical -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.externmodule @Sender() -> ( %x: !esi.channel<i1> )
  rtl.externmodule @Reciever(%a: !esi.channel<i1>)

  // CHECK-LABEL: rtl.externmodule @Sender() -> (%x: !esi.channel<i1>)
  // CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i1> {rtl.name = "a"})

  rtl.module @test() {
    %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan = esi.buffer %esiChan { } : i1
    rtl.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // CHECK:  %esiChan = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %0 = esi.stage %esiChan : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

    %esiChan2 = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan2 = esi.buffer %esiChan2 { stages = 4 } : i1
    rtl.instance "recv" @Reciever (%bufferedChan2) : (!esi.channel<i1>) -> ()

    // CHECK-NEXT:  %esiChan2 = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %1 = esi.stage %esiChan2 : i1
    // CHECK-NEXT:  %2 = esi.stage %1 : i1
    // CHECK-NEXT:  %3 = esi.stage %2 : i1
    // CHECK-NEXT:  %4 = esi.stage %3 : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%4)  : (!esi.channel<i1>) -> ()
  }
}
