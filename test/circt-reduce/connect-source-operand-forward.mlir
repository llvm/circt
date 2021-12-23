// RUN: circt-reduce %s --test %S/test.sh --test-arg cat --test-arg "firrtl.module @Foo" --keep-best=0 --include connect-source-operand-0-forwarder | FileCheck %s
firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %val: !firrtl.uint<2>) {
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.reg %clock : !firrtl.uint<1>
    %c = firrtl.regreset %clock, %reset, %reset : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.bits %val 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT:   %0 = firrtl.wire  : !firrtl.uint<2>
    // CHECK-NEXT:   %1 = firrtl.reg %clock  : !firrtl.uint<2>
    // CHECK-NEXT:   %2 = firrtl.reg %clock  : !firrtl.uint<2>
    // CHECK-NEXT:   firrtl.connect %0, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT:   firrtl.connect %1, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT:   firrtl.connect %2, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT: }
  }
}
