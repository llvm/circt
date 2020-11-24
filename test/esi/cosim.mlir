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

  // CHECK-LABEL: rtl.module @Sender() -> (%x: !esi.channel<i1>)
  // CHECK-LABEL: rtl.module @Reciever(%a: !esi.channel<i1>)

  rtl.module @Top() -> () {
    rtl.instance "recv" @Reciever (%cosimRecv) : (!esi.channel<i1>) -> ()
    // CHECK:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

    %cosimSend = rtl.instance "send" @Sender () : () -> (!esi.channel<i1>)
    // CHECK:  %cosimSend = rtl.instance "send" @Sender() : () -> !esi.channel<i1>

    %cosimRecv = esi.cosim (%cosimSend) {} : !esi.channel<i1> -> !esi.channel<i1>
    // CHECK:  %0 = esi.cosim(%cosimSend) : !esi.channel<i1> -> !esi.channel<i1>
  }
}
