// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @Sender() -> ( !esi.channel<i1> { rtl.name = "x"}) {
    %0 = constant 0 : i1
    // Don't transmit.
    %1, %rcvrRdy = esi.wrap.vr %0, %0 : i1
    rtl.output %1 : !esi.channel<i1>
  }
  rtl.module @Reciever(%a: !esi.channel<i1>) {
    %false = constant 0 : i1
    %0, %valid = esi.unwrap.vr %a, %false : i1
  }

  // CHECK-LABEL: rtl.module @Sender() -> (%x: !esi.channel<i1>)
  // CHECK-LABEL: rtl.module @Reciever(%a: !esi.channel<i1>)

  rtl.module @Top() -> () {
    rtl.instance "recv" @Reciever (%cosimRecv) : (!esi.channel<i1>) -> ()
    // CHECK:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

    %send.x = rtl.instance "send" @Sender () : () -> (!esi.channel<i1>)
    // CHECK:  %send.x = rtl.instance "send" @Sender() : () -> !esi.channel<i1>

    %cosimRecv = esi.cosim (%send.x) {} : !esi.channel<i1> -> !esi.channel<i1>
    // CHECK:  %0 = esi.cosim(%send.x) : !esi.channel<i1> -> !esi.channel<i1>
  }
}
