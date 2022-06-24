// RUN: circt-opt %s -test-prepare-for-emission --split-input-file | FileCheck %s

// CHECK: @namehint_variadic
hw.module @namehint_variadic(%a: i3) -> (b: i3) {
  // CHECK-NEXT: %0 = comb.add %a, %a : i3
  // CHECK-NEXT: %1 = comb.add %a, %0 {sv.namehint = "bar"} : i3
  // CHECK-NEXT: hw.output %1
  %0 = comb.add %a, %a, %a { sv.namehint = "bar" } : i3
  hw.output %0 : i3
}

// -----

module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  // CHECK: @test_hoist
  hw.module @test_hoist(%a: i3) -> () {
    // CHECK-NEXT: %reg = sv.reg
    %reg = sv.reg : !hw.inout<i3>
    // CHECK-NEXT: %0 = comb.add
    // CHECK-NEXT: sv.initial
    sv.initial {
      %0 = comb.add %a, %a : i3
      sv.passign %reg, %0 : i3
    }
  }
}
