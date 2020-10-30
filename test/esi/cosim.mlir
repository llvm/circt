// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

esi.snippet "testSnip" {
  rtl.externmodule @Sender() -> ( !esi.channel<i1> { rtl.name = "x"} )
  rtl.externmodule @Reciever(%a: !esi.channel<i1>)

  // CHECK-LABEL: rtl.externmodule @Sender() -> (!esi.channel<i1> {rtl.name = "x"})
  // CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i1> {rtl.name = "a"})

  rtl.instance "recv" @Reciever (%cosimRecv) : (!esi.channel<i1>) -> ()
  // CHECK-NEXT:  rtl.instance "recv" @Reciever(%1)  : (!esi.channel<i1>) -> ()

  %cosimSend = esi.instantiated "send" @Sender () : () -> (!esi.channel<i1>)
  // CHECK-NEXT:  %0 = esi.instantiated "send" @Sender() : () -> !esi.channel<i1>

  %cosimRecv = esi.cosim (%cosimSend) {} : !esi.channel<i1> -> !esi.channel<i1>
  // CHECK-NEXT:  %1 = esi.cosim(%0) {} : !esi.channel<i1> -> !esi.channel<i1>
}
