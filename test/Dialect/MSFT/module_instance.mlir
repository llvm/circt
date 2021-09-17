// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --msft-lower-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=HWLOW

msft.module @Foo { "WIDTH" = 1 } () -> (x: i32) {
  %c0 = hw.constant 0 : i32
  msft.output %c0 : i32
}

// CHECK-LABEL: hw.module @top
// HWLOW-LABEL: hw.module @top
hw.module @top () {
  msft.instance "foo" @Foo() : () -> (i32)
  // CHECK: %foo.x = msft.instance "foo" @Foo() : () -> i32
  // HWLOW: %foo.x = hw.instance "foo" @Foo() -> (x: i32)
}

// CHECK-LABEL: msft.module @B {WIDTH = 1 : i64} (%a: i4) -> (nameOfPortInSV: i4) {
// HWLOW-LABEL: hw.module @B(%a: i4) -> (nameOfPortInSV: i4) {
msft.module @B { "WIDTH" = 1 } (%a: i4) -> (nameOfPortInSV: i4) {
  %0 = comb.add %a, %a : i4
  // CHECK: comb.add %a, %a : i4
  // HWLOW: comb.add %a, %a : i4
  msft.output %0: i4
}

// CHECK-LABEL: msft.module @UnGenerated {DEPTH = 3 : i64} (%a: i1) -> (nameOfPortInSV: i1)
// HWLOW-LABEL: Module not generated: \22UnGenerated\22 params {DEPTH = 3 : i64}
msft.module @UnGenerated { DEPTH = 3 } (%a: i1) -> (nameOfPortInSV: i1)
