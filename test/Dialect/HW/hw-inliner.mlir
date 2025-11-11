// RUN: circt-opt %s --hw-inliner | FileCheck %s

// Test basic inlining of a small private module
// CHECK-LABEL: hw.module @InlineSmall
hw.module @InlineSmall(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: hw.output %[[V0]]
  %0 = hw.instance "small" @SmallModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @SmallModule
hw.module private @SmallModule(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  hw.output %0 : i4
}

// Test inlining of an empty module
// CHECK-LABEL: hw.module @InlineEmpty
hw.module @InlineEmpty(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: hw.output %x
  hw.instance "empty" @EmptyModule() -> ()
  hw.output %x : i4
}
// CHECK-NOT: hw.module private @EmptyModule
hw.module private @EmptyModule() {
  hw.output
}

// Test inlining of a module with no outputs
// CHECK-LABEL: hw.module @InlineNoOutputs
hw.module @InlineNoOutputs(in %x: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: hw.output
  hw.instance "noout" @NoOutputModule(in: %x: i4) -> ()
  hw.output
}
// CHECK-NOT: hw.module private @NoOutputModule
hw.module private @NoOutputModule(in %in: i4) {
  %0 = comb.add %in, %in : i4
  hw.output
}

// Test that public modules are not inlined
// CHECK-LABEL: hw.module @DontInlinePublic
hw.module @DontInlinePublic(in %x: i4, out y: i4) {
  // CHECK-NEXT: hw.instance "pub" @PublicModule
  %0 = hw.instance "pub" @PublicModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK: hw.module @PublicModule
hw.module @PublicModule(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  hw.output %0 : i4
}

// Test inlining with single use
// CHECK-LABEL: hw.module @InlineSingleUse
hw.module @InlineSingleUse(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.mul %x, %x
  // CHECK-NEXT: hw.output %[[V0]]
  %0 = hw.instance "single" @SingleUseModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @SingleUseModule
hw.module private @SingleUseModule(in %a: i4, out b: i4) {
  %0 = comb.mul %a, %a : i4
  hw.output %0 : i4
}

// Test that modules with multiple uses and state are not inlined
// CHECK-LABEL: hw.module @DontInlineMultiUseWithState
hw.module @DontInlineMultiUseWithState(in %clk: !seq.clock, in %x: i4, out y: i4, out z: i4) {
  // CHECK-NEXT: hw.instance "inst1" @ModuleWithState
  // CHECK-NEXT: hw.instance "inst2" @ModuleWithState
  %0 = hw.instance "inst1" @ModuleWithState(clk: %clk: !seq.clock, a: %x: i4) -> (b: i4)
  %1 = hw.instance "inst2" @ModuleWithState(clk: %clk: !seq.clock, a: %x: i4) -> (b: i4)
  hw.output %0, %1 : i4, i4
}
// CHECK: hw.module private @ModuleWithState
hw.module private @ModuleWithState(in %clk: !seq.clock, in %a: i4, out b: i4) {
  %0 = seq.firreg %a clock %clk : i4
  hw.output %0 : i4
}

// Test name prefixing during inlining
// CHECK-LABEL: hw.module @InlineWithNames
hw.module @InlineWithNames(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[C:.+]] = hw.constant 1 : i4 {name = "named.const_val"}
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %[[C]]
  // CHECK-NEXT: hw.output %[[V0]]
  %0 = hw.instance "named" @ModuleWithNames(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @ModuleWithNames
hw.module private @ModuleWithNames(in %a: i4, out b: i4) {
  %c = hw.constant 1 : i4 {name = "const_val"}
  %0 = comb.add %a, %c : i4
  hw.output %0 : i4
}

// Test inlining with multiple operations
// CHECK-LABEL: hw.module @InlineMultiOp
hw.module @InlineMultiOp(in %x: i4, in %y: i4, out z: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %y
  // CHECK-NEXT: %[[V1:.+]] = comb.mul %[[V0]], %x
  // CHECK-NEXT: %[[V2:.+]] = comb.xor %[[V1]], %y
  // CHECK-NEXT: hw.output %[[V2]]
  %0 = hw.instance "multi" @MultiOpModule(a: %x: i4, b: %y: i4) -> (c: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @MultiOpModule
hw.module private @MultiOpModule(in %a: i4, in %b: i4, out c: i4) {
  %0 = comb.add %a, %b : i4
  %1 = comb.mul %0, %a : i4
  %2 = comb.xor %1, %b : i4
  hw.output %2 : i4
}

// Test that large modules with state are not inlined
// CHECK-LABEL: hw.module @DontInlineLargeWithState
hw.module @DontInlineLargeWithState(in %clk: !seq.clock, in %x: i4, out y: i4) {
  // CHECK-NEXT: hw.instance "large" @LargeModuleWithState
  %0 = hw.instance "large" @LargeModuleWithState(clk: %clk: !seq.clock, a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK: hw.module private @LargeModuleWithState
hw.module private @LargeModuleWithState(in %clk: !seq.clock, in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = comb.mul %0, %a : i4
  %2 = comb.xor %1, %a : i4
  %3 = comb.or %2, %a : i4
  %4 = comb.and %3, %a : i4
  %5 = comb.add %4, %a : i4
  %6 = comb.mul %5, %a : i4
  %7 = comb.xor %6, %a : i4
  %reg = seq.firreg %7 clock %clk : i4
  hw.output %reg : i4
}

// Test inlining with nested instances
// CHECK-LABEL: hw.module @InlineNested
hw.module @InlineNested(in %x: i4, out y: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[V1:.+]] = comb.mul %[[V0]], %[[V0]]
  // CHECK-NEXT: hw.output %[[V1]]
  %0 = hw.instance "outer" @OuterModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @OuterModule
hw.module private @OuterModule(in %a: i4, out b: i4) {
  %0 = hw.instance "inner" @InnerModule(x: %a: i4) -> (y: i4)
  hw.output %0 : i4
}
// CHECK-NOT: hw.module private @InnerModule
hw.module private @InnerModule(in %x: i4, out y: i4) {
  %0 = comb.add %x, %x : i4
  %1 = comb.mul %0, %0 : i4
  hw.output %1 : i4
}

// Test that small modules without state are inlined even with multiple uses
// CHECK-LABEL: hw.module @InlineSmallMultiUse
hw.module @InlineSmallMultiUse(in %x: i4, in %y: i4, out a: i4, out b: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[V1:.+]] = comb.add %y, %y
  // CHECK-NEXT: hw.output %[[V0]], %[[V1]]
  %0 = hw.instance "inst1" @TinyModule(in: %x: i4) -> (out: i4)
  %1 = hw.instance "inst2" @TinyModule(in: %y: i4) -> (out: i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NOT: hw.module private @TinyModule
hw.module private @TinyModule(in %in: i4, out out: i4) {
  %0 = comb.add %in, %in : i4
  hw.output %0 : i4
}

// Test multiple outputs
// CHECK-LABEL: hw.module @InlineMultiOutput
hw.module @InlineMultiOutput(in %x: i4, out y: i4, out z: i4) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: %[[V0:.+]] = comb.add %x, %x
  // CHECK-NEXT: %[[V1:.+]] = comb.mul %x, %x
  // CHECK-NEXT: hw.output %[[V0]], %[[V1]]
  %0, %1 = hw.instance "multi_out" @MultiOutputModule(a: %x: i4) -> (b: i4, c: i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NOT: hw.module private @MultiOutputModule
hw.module private @MultiOutputModule(in %a: i4, out b: i4, out c: i4) {
  %0 = comb.add %a, %a : i4
  %1 = comb.mul %a, %a : i4
  hw.output %0, %1 : i4, i4
}

