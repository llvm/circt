// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-drop-name))' %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %val: !firrtl.uint<2>) {
    // CHECK-NEXT:  %a = firrtl.wire droppable_name  : !firrtl.uint<1>
    // CHECK-NEXT:  %b = firrtl.reg droppable_name %clock  : !firrtl.uint<1>
    // CHECK-NEXT:  %c = firrtl.regreset droppable_name %clock, %reset, %reset  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT:  %d = firrtl.node droppable_name %reset  : !firrtl.uint<1>
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.reg %clock : !firrtl.uint<1>
    %c = firrtl.regreset %clock, %reset, %reset : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %d = firrtl.node %reset : !firrtl.uint<1>
  }
}
