// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-types,firrtl-imconstprop),canonicalize' %s | FileCheck %s
// Tests extracted from:
// - test/scala/firrtlTests/transforms/RemoveResetSpec.scala

firrtl.circuit "Foo" {
firrtl.module @Foo() {}

// Should not generate a reset mux for an invalid init.
// CHECK-LABEL: firrtl.module @NoMuxForInvalid
firrtl.module @NoMuxForInvalid(
  in %clk: !firrtl.clock,
  in %rst: !firrtl.reset,
  in %arst: !firrtl.asyncreset,
  in %srst: !firrtl.uint<1>
) {
  // CHECK: %foo0 = firrtl.reg %clk :
  // CHECK: %foo1 = firrtl.reg %clk :
  // CHECK: %foo2 = firrtl.reg %clk :
  %invalid_ui42 = firrtl.invalidvalue : !firrtl.uint<42>
  %foo0 = firrtl.regreset %clk, %rst, %invalid_ui42 : !firrtl.reset, !firrtl.uint<42>, !firrtl.uint<42>
  %foo1 = firrtl.regreset %clk, %arst, %invalid_ui42 : !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
  %foo2 = firrtl.regreset %clk, %srst, %invalid_ui42 : !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
}

// Should not generate a reset mux for an invalid init.
// CHECK-LABEL: firrtl.module @NoMuxForInvalidWire
firrtl.module @NoMuxForInvalidWire(
  in %clk: !firrtl.clock,
  in %rst: !firrtl.reset,
  in %arst: !firrtl.asyncreset,
  in %srst: !firrtl.uint<1>
) {
  // CHECK: %foo0 = firrtl.reg %clk :
  // CHECK: %foo1 = firrtl.reg %clk :
  // CHECK: %foo2 = firrtl.reg %clk :
  %bar = firrtl.wire  : !firrtl.uint<42>
  %invalid_ui42 = firrtl.invalidvalue : !firrtl.uint<42>
  firrtl.connect %bar, %invalid_ui42 : !firrtl.uint<42>, !firrtl.uint<42>
  %foo0 = firrtl.regreset %clk, %rst, %bar : !firrtl.reset, !firrtl.uint<42>, !firrtl.uint<42>
  %foo1 = firrtl.regreset %clk, %arst, %bar : !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
  %foo2 = firrtl.regreset %clk, %srst, %bar : !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
}

// Should generate a reset mux for only the portion of an invalid aggregate that
// is reset.
// CHECK-LABEL: firrtl.module @PartiallyNoMuxInAggregate
firrtl.module @PartiallyNoMuxInAggregate(
  in %clk: !firrtl.clock,
  in %rst: !firrtl.reset,
  in %arst: !firrtl.asyncreset,
  in %srst: !firrtl.uint<1>
) {
  %invalid = firrtl.invalidvalue : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

  %bar = firrtl.wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>

  firrtl.connect %bar, %invalid : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>
  %0 = firrtl.subfield %bar(0) : (!firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>) -> !firrtl.vector<uint<1>, 2>
  %1 = firrtl.subindex %0[1] : !firrtl.vector<uint<1>, 2>
  firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: %foo0_a_0 = firrtl.reg %clk :
  // CHECK: %foo0_a_1 = firrtl.regreset %clk, %rst, %bar_a_1 :
  // CHECK: %foo0_b = firrtl.reg %clk :
  // CHECK: %foo1_a_0 = firrtl.reg %clk :
  // CHECK: %foo1_a_1 = firrtl.regreset %clk, %arst, %bar_a_1 :
  // CHECK: %foo1_b = firrtl.reg %clk :
  // CHECK: %foo2_a_0 = firrtl.reg %clk :
  // CHECK: %foo2_a_1 = firrtl.regreset %clk, %srst, %bar_a_1 :
  // CHECK: %foo2_b = firrtl.reg %clk :
  %foo0 = firrtl.regreset %clk, %rst, %bar  : !firrtl.reset, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>
  %foo1 = firrtl.regreset %clk, %arst, %bar  : !firrtl.asyncreset, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>
  %foo2 = firrtl.regreset %clk, %srst, %bar  : !firrtl.uint<1>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>
}

// Should propagate invalidations across connects.
// CHECK-LABEL: firrtl.module @PropagateInvalidAcrossConnects
firrtl.module @PropagateInvalidAcrossConnects(
  in %clk: !firrtl.clock,
  in %rst: !firrtl.reset,
  in %arst: !firrtl.asyncreset,
  in %srst: !firrtl.uint<1>
) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %invalid = firrtl.invalidvalue : !firrtl.bundle<a: uint<1>, b: uint<1>>

  %bar = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
  %baz = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>

  firrtl.connect %bar, %invalid : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  %bar_a = firrtl.subfield %bar(0) : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<1>
  firrtl.connect %bar_a, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

  firrtl.connect %baz, %invalid : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  firrtl.connect %baz, %bar : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>

  // CHECK: %foo0_a = firrtl.regreset %clk, %rst, %bar_a :
  // CHECK: %foo0_b = firrtl.reg %clk :
  // CHECK: %foo1_a = firrtl.regreset %clk, %arst, %bar_a :
  // CHECK: %foo1_b = firrtl.reg %clk :
  // CHECK: %foo2_a = firrtl.regreset %clk, %srst, %bar_a :
  // CHECK: %foo2_b = firrtl.reg %clk :
  %foo0 = firrtl.regreset %clk, %rst, %bar  : !firrtl.reset, !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  %foo1 = firrtl.regreset %clk, %arst, %bar  : !firrtl.asyncreset, !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  %foo2 = firrtl.regreset %clk, %srst, %bar  : !firrtl.uint<1>, !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
}

// Should convert a reset wired to UInt(0) to a canonical non-reset.
// CHECK-LABEL: firrtl.module @TreatUInt0ResetAsNonReset
firrtl.module @TreatUInt0ResetAsNonReset(
  in %clk: !firrtl.clock
) {
  %rst = firrtl.specialconstant 0 : !firrtl.reset
  %arst = firrtl.specialconstant 0 : !firrtl.asyncreset
  %srst = firrtl.constant 0 : !firrtl.uint<1>
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  // CHECK: %foo0 = firrtl.reg %clk :
  // CHECK: %foo1 = firrtl.reg %clk :
  // CHECK: %foo2 = firrtl.reg %clk :
  %foo0 = firrtl.regreset %clk, %rst, %c3_ui2  : !firrtl.reset, !firrtl.uint<2>, !firrtl.uint<2>
  %foo1 = firrtl.regreset %clk, %arst, %c3_ui2  : !firrtl.asyncreset, !firrtl.uint<2>, !firrtl.uint<2>
  %foo2 = firrtl.regreset %clk, %srst, %c3_ui2  : !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>
}

}
