// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-msft-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=HWLOW

// CHECK-LABEL: hw.module @top
// HWLOW-LABEL: hw.module @top
hw.module @top () {
  msft.instance @foo @Foo() : () -> (i32)
  // CHECK: %foo.x = msft.instance @foo @Foo() : () -> i32
  // HWLOW: %foo.x = hw.instance "foo" sym @foo @Foo() -> (x: i32)

  %true = hw.constant true
  %extern.out = msft.instance @extern @Extern(%true)<param: i1 = false> : (i1) -> i1
  // CHECK: %extern.out = msft.instance @extern @Extern(%true) <param: i1 = false> : (i1) -> i1
  // HWLOW: %extern.out = hw.instance "extern" sym @extern @Extern<param: i1 = false>(in: %true: i1) -> (out: i1)
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

msft.module @Foo { "WIDTH" = 1 } () -> (x: i32) {
  %c0 = hw.constant 0 : i32
  msft.output %c0 : i32
}

// CHECK-LABEL: msft.module.extern @Extern<param: i1>(%in: i1) -> (out: i1)
// HWLOW-LABEL: hw.module.extern @Extern<param: i1>(%in: i1) -> (out: i1)
msft.module.extern @Extern<param: i1> (%in: i1) -> (out: i1)
