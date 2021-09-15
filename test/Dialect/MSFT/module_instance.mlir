// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --msft-lower-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=HWLOW

hw.module.extern @fooMod () -> (x: i32)

// CHECK-LABEL: hw.module @top
// HWLOW-LABEL: hw.module @top
hw.module @top () {
  msft.instance "foo" @fooMod () : () -> (i32)
  // CHECK: %foo.x = msft.instance "foo" @fooMod() : () -> i32
  // HWLOW: %foo.x = hw.instance "foo" @fooMod() -> (x: i32)
}

msft.module @B { "WIDTH" = 1 } (%a: i1) -> (nameOfPortInSV: i1) {
  %0 = comb.or %a, %a : i1
  %1 = comb.and %a, %a : i1
  msft.output %0, %1: i1, i1
}

msft.module @UnGenerated { DEPTH = 3 } (%a: i1) -> (nameOfPortInSV: i1)
