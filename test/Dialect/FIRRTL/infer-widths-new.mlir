// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-widths-new))' --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  
  //===----------------------------------------------------------------------===//
  // New Test: issue9140_0
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: @issue9140_0
  // CHECK-SAME: out %out: !firrtl.uint<5>
  firrtl.module @issue9140_0(in %in: !firrtl.uint<4>, in %clock: !firrtl.clock, out %out: !firrtl.uint) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: %x = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<5>
    // CHECK: %0 = firrtl.tail {{.*}} -> !firrtl.uint<4>
    // CHECK: %1 = firrtl.add {{.*}} -> !firrtl.uint<5>
    %x = firrtl.reg %clock : !firrtl.clock, !firrtl.uint
    %0 = firrtl.tail %x, 1 : (!firrtl.uint) -> !firrtl.uint
    %1 = firrtl.add %0, %in : (!firrtl.uint, !firrtl.uint<4>) -> !firrtl.uint
    firrtl.connect %x, %1 : !firrtl.uint
    firrtl.connect %out, %x : !firrtl.uint
  }

  //===----------------------------------------------------------------------===//
  // New Test: issue9140_1
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: @issue9140_1
  // CHECK-SAME: out %out: !firrtl.uint<4>
  firrtl.module @issue9140_1(in %in: !firrtl.uint<4>, in %clock: !firrtl.clock, out %out: !firrtl.uint) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: %x1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<4>
    // CHECK: %x2 = firrtl.wire : !firrtl.uint<0>
    // CHECK: %x3 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %0 = firrtl.mul {{.*}} -> !firrtl.uint<4>
    // CHECK: %1 = firrtl.mul {{.*}} -> !firrtl.uint<4>
    // CHECK: %2 = firrtl.shr {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.pad {{.*}} -> !firrtl.uint<2>
    // CHECK: %4 = firrtl.tail {{.*}} -> !firrtl.uint<0>
    %x1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint
    %x2 = firrtl.wire : !firrtl.uint
    %x3 = firrtl.wire : !firrtl.uint
    %0 = firrtl.mul %x2, %in : (!firrtl.uint, !firrtl.uint<4>) -> !firrtl.uint
    %1 = firrtl.mul %0, %x2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %x1, %1 : !firrtl.uint
    %2 = firrtl.shr %x1, 2 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.pad %2, 1 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %x3, %3 : !firrtl.uint
    %4 = firrtl.tail %x3, 2 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %x2, %4 : !firrtl.uint
    firrtl.connect %out, %x1 : !firrtl.uint
  }
}
