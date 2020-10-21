// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.externmodule @Sender() -> ( !esi.channel<i1> { rtl.name = "x"})
  rtl.externmodule @Reciever(%a: !esi.channel<i1>)

  %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
  rtl.instance "recv" @Reciever (%esiChan) : (!esi.channel<i1>) -> ()

  // CHECK-LABEL: rtl.externmodule @Sender() -> (!esi.channel<i1> {rtl.name = "x"})
  // CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i1> {rtl.name = "a"})
  // CHECK:       %esiChan = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
  // CHECK:       rtl.instance "recv" @Reciever(%esiChan)  : (!esi.channel<i1>) -> ()

}
