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

// CHECK-LABEL: Module "Vars" for hw.module
// CHECK-NEXT: Variable "inA"
// CHECK-NEXT:   Arg 0 of hw.module of type i32
// CHECK-NEXT: Variable "outB"
// CHECK-NEXT:   Result 0 of comb.add of type i32
// CHECK-NOT: Variable "a"
// CHECK-NOT: Variable "b"
hw.module @Vars(in %a: i32, out b: i32) {
  dbg.variable "inA", %a : i32
  dbg.variable "outB", %0 : i32
  %0 = comb.add %a, %a : i32
  hw.output %0 : i32
}

// CHECK-LABEL: Module "Aggregates" for hw.module
// CHECK-NEXT: Variable "data"
// CHECK-NEXT:   Result 0 of dbg.struct of type !dbg.struct
hw.module @Aggregates(in %data_a: i32, in %data_b: index, in %data_c_0: i17, in %data_c_1: i17) {
  %0 = dbg.array [%data_c_0, %data_c_1] : i17
  %1 = dbg.struct {"a": %data_a, "b": %data_b, "c": %0} : i32, index, !dbg.array<i17>
  dbg.variable "data", %1 : !dbg.struct<i32, index, !dbg.array<i17>>
}

// CHECK-LABEL: Module "InlineScopes" for hw.module
// CHECK:       Variable "a"
// CHECK-NEXT:    Arg 0 of hw.module of type i42
// CHECK:       Instance "inner" of "InnerModule" for dbg.scope
// CHECK:         Module "InnerModule" for dbg.scope
// CHECK:           Variable "b"
// CHECK-NEXT:        Result 0 of comb.mul of type i42
hw.module @InlineScopes(in %a: i42) {
  dbg.variable "a", %a : i42
  %0 = comb.mul %a, %a : i42
  %1 = dbg.scope "inner", "InnerModule"
  dbg.variable "b", %0 scope %1 : i42
}

// CHECK-LABEL: Module "NestedScopes" for hw.module
// CHECK:       Instance "foo" of "Foo" for dbg.scope
// CHECK:         Module "Foo" for dbg.scope
// CHECK:           Instance "bar" of "Bar" for dbg.scope
// CHECK:             Module "Bar" for dbg.scope
// CHECK:               Variable "a"
hw.module @NestedScopes(in %a: i42) {
  %0 = dbg.scope "foo", "Foo"
  %1 = dbg.scope "bar", "Bar" scope %0
  dbg.variable "a", %a scope %1 : i42
}
