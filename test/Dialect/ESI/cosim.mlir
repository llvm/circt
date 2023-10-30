// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module.extern @Sender(out x: !esi.channel<si14>)
hw.module.extern @Reciever(in %a: !esi.channel<i32>)
hw.module.extern @ArrReciever(in %x: !esi.channel<!hw.array<4xsi64>>)

// CHECK-LABEL: hw.module.extern @Sender(out x : !esi.channel<si14>)
// CHECK-LABEL: hw.module.extern @Reciever(in %a : !esi.channel<i32>)
// CHECK-LABEL: hw.module.extern @ArrReciever(in %x : !esi.channel<!hw.array<4xsi64>>)

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  hw.instance "recv" @Reciever (a: %cosimRecv: !esi.channel<i32>) -> ()
  // CHECK:  hw.instance "recv" @Reciever(a: %0: !esi.channel<i32>) -> ()

  %send.x = hw.instance "send" @Sender () -> (x: !esi.channel<si14>)
  // CHECK:  %send.x = hw.instance "send" @Sender() -> (x: !esi.channel<si14>)

  %cosimRecv = esi.cosim %clk, %rst, %send.x, "TestEP" : !esi.channel<si14> -> !esi.channel<i32>
  // CHECK:  esi.cosim %clk, %rst, %send.x, "TestEP" : !esi.channel<si14> -> !esi.channel<i32>

  %send2.x = hw.instance "send2" @Sender () -> (x: !esi.channel<si14>)
  // CHECK:  %send2.x = hw.instance "send2" @Sender() -> (x: !esi.channel<si14>)

  %cosimArrRecv = esi.cosim %clk, %rst, %send2.x, "ArrTestEP" : !esi.channel<si14> -> !esi.channel<!hw.array<4xsi64>>
  // CHECK:  esi.cosim %clk, %rst, %send2.x, "ArrTestEP" : !esi.channel<si14> -> !esi.channel<!hw.array<4xsi64>>

  hw.instance "arrRecv" @ArrReciever (x: %cosimArrRecv: !esi.channel<!hw.array<4 x si64>>) -> ()
}
