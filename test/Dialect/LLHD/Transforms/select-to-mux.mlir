// RUN: circt-opt --llhd-select-to-mux %s | FileCheck %s

// CHECK-LABEL: @Foo(
hw.module @Foo(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK-NEXT: hw.constant 9001 : i42
  // CHECK-NOT: arith.constant
  arith.constant 9001 : i42
  // CHECK-NEXT: comb.mux %c, %a, %b : i42
  // CHECK-NOT: arith.select
  arith.select %c, %a, %b : i42

  // CHECK-NEXT: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: hw.constant 9001 : i42
    // CHECK-NOT: arith.constant
    arith.constant 9001 : i42
    // CHECK-NEXT: comb.mux %c, %a, %b : i42
    // CHECK-NOT: arith.select
    arith.select %c, %a, %b : i42
    llhd.yield
  }
}
