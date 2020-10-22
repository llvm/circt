// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.externmodule @Sender() -> ( !esi.channel<i1> { rtl.name = "x"})
  rtl.externmodule @Reciever(%a: !esi.channel<i1>)

  %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
  %bufferedChan = esi.buffer %esiChan {  } : i1
  rtl.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

  // CHECK-LABEL: rtl.externmodule @Sender() -> (!esi.channel<i1> {rtl.name = "x"})
  // CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i1> {rtl.name = "a"})
  // CHECK-NEXT:  %esiChan = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
  // CHECK-NEXT:  %0 = esi.buffer %esiChan {} : i1
  // CHECK-NEXT:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

}
