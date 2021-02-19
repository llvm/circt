// RUN: circt-opt %s -inline | FileCheck %s

rtl.module @empty() -> () {
  rtl.output
}

rtl.module @test0() -> () {
  rtl.instance "inline-me" @empty() {inline} : () -> ()
  rtl.output
}

// CHECK-LABEL: rtl.module @test0() {
// CHECK-NEXT:     rtl.output
// CHECK-NEXT: }

rtl.module @simple(%a: i2, %b : i2) -> (i2, i2) {
  %0 = comb.or %a, %b : i2
  %1 = comb.and %a, %b : i2
  rtl.output %0, %1: i2, i2
}

rtl.module @test1(%a: i2, %b : i2) -> (i2, i2) {
  %0, %1 = rtl.instance "inline-me" @simple(%a, %b) {inline} : (i2, i2) -> (i2, i2)
  rtl.output %0, %1: i2, i2
}

// CHECK-LABEL: rtl.module @test1(%a: i2, %b: i2) -> (i2, i2) {
// CHECK-NEXT:   %0 = comb.or %a, %b : i2
// CHECK-NEXT:   %1 = comb.and %a, %b : i2
// CHECK-NEXT:   rtl.output %0, %1 : i2, i2
// CHECK-NEXT: }
