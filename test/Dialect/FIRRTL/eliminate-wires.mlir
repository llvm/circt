// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-eliminate-wires)))' %s | FileCheck %s

firrtl.circuit "TopLevel" {

  // CHECK-LABEL: @TopLevel
  firrtl.module @TopLevel(in %source: !firrtl.uint<1>, 
                             out %sink: !firrtl.uint<1>) {
    // CHECK-NOT: firrtl.wire
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %w, %source : !firrtl.uint<1>
    %wn = firrtl.not %w : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %x = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %x, %wn : !firrtl.uint<1>
    firrtl.matchingconnect %sink, %x : !firrtl.uint<1>
    firrtl.matchingconnect %sink, %w : !firrtl.uint<1>
  }

  // CHECK-LABEL: @Foo
  firrtl.module private @Foo() {
    %a = firrtl.wire : !firrtl.uint<3>
    %b = firrtl.wire : !firrtl.uint<3>
    %invalid_ui3 = firrtl.invalidvalue : !firrtl.uint<3>
    firrtl.matchingconnect %b, %invalid_ui3 : !firrtl.uint<3>
    firrtl.matchingconnect %a, %b : !firrtl.uint<3>
    // CHECK: %[[inv:.*]] = firrtl.invalidvalue : !firrtl.uint<3>
    // CHECK-NEXT:  %b = firrtl.node %[[inv]] : !firrtl.uint<3>
    // CHECK-NEXT:  %a = firrtl.node %b : !firrtl.uint<3>
  }
}