// RUN: circt-translate --export-firrtl --verify-diagnostics %s --target-fir-version=5.0.0 -o %t
// RUN: cat %t | FileCheck %s --strict-whitespace
// RUN: circt-translate --import-firrtl %t --mlir-print-debuginfo | circt-translate --export-firrtl --target-fir-version=5.0.0 | diff - %t

// CHECK-LABEL: FIRRTL version 5.0.0
// CHECK-LABEL: circuit VersionTest :
firrtl.circuit "VersionTest" {
  // CHECK-LABEL: public module VersionTest :
  firrtl.module @VersionTest(in %x: !firrtl.uint<1>, in %y: !firrtl.sint<1>, in %z: !firrtl.uint<1>) {
    // Check that variadic cat ops are properly emitted before 5.1.0.
    %cat_0_tmp = firrtl.cat  : () -> !firrtl.uint<0>
    %cat_1_tmp = firrtl.cat %x : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %cat_1_signed_tmp = firrtl.cat %y : (!firrtl.sint<1>) -> !firrtl.uint<1>
    %cat_2_tmp = firrtl.cat %x, %z : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %cat_3_tmp = firrtl.cat %x, %z, %x : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<3>
    // CHECK:      node cat_0_node = UInt<0>(0)
    // CHECK-NEXT: node cat_1_node = x
    // CHECK-NEXT: node cat_1_signed_node = cat(y, SInt<0>(0))
    // CHECK-NEXT: node cat_2_node = cat(x, z)
    // CHECK-NEXT: node cat_3_node = cat(x, cat(z, x))
    %cat_0_node = firrtl.node %cat_0_tmp : !firrtl.uint<0>
    %cat_1_node = firrtl.node %cat_1_tmp : !firrtl.uint<1>
    %cat_1_signed_node = firrtl.node %cat_1_signed_tmp : !firrtl.uint<1>
    %cat_2_node = firrtl.node %cat_2_tmp : !firrtl.uint<2>
    %cat_3_node = firrtl.node %cat_3_tmp : !firrtl.uint<3>
  }
}
