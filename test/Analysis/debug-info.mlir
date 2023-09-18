// RUN: circt-translate --dump-di %s | FileCheck %s

// CHECK-LABEL: Module "Foo" for hw.module
// CHECK: Variable "a"
// CHECK:   Arg 0 of hw.module of type i32
// CHECK: Variable "b"
// CHECK:   Result 0 of hw.instance of type i32
// CHECK: Instance "b0" of "Bar" for hw.instance
// CHECK: Instance "b1" of "Bar" for hw.instance
hw.module @Foo(in %a: i32, out b: i32) {
  %b0.y = hw.instance "b0" @Bar(x: %a: i32) -> (y: i32)
  %b1.y = hw.instance "b1" @Bar(x: %b0.y: i32) -> (y: i32)
  hw.output %b1.y : i32
}

// CHECK-LABEL: Module "Bar" for hw.module
// CHECK: Variable "x"
// CHECK:   Arg 0 of hw.module of type i32
// CHECK: Variable "y"
// CHECK:   Result 0 of hw.wire of type i32
// CHECK: Variable "z"
// CHECK:   Result 0 of hw.wire of type i32
hw.module @Bar(in %x: i32, out y: i32) {
  %0 = comb.mul %x, %x : i32
  %z = hw.wire %0 : i32
  hw.output %z : i32
}
