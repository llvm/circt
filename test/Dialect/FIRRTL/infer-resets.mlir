// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets))' --verify-diagnostics --split-input-file %s | FileCheck %s

// Tests extracted from:
// - github.com/chipsalliance/firrtl:
//   - test/scala/firrtlTests/InferResetsSpec.scala
// - github.com/sifive/$internal:
//   - test/scala/firrtl/FullAsyncResetTransform.scala

firrtl.circuit "Foo" {
firrtl.module @Foo() {}


//===----------------------------------------------------------------------===//
// Reset Inference
//===----------------------------------------------------------------------===//

// Provoke two existing reset networks being merged.
// CHECK-LABEL: firrtl.module @MergeNetsChild1
// CHECK-SAME: in %reset: !firrtl.asyncreset
firrtl.module @MergeNetsChild1(in %reset: !firrtl.reset) {
  // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
  %localReset = firrtl.wire : !firrtl.reset
  firrtl.connect %localReset, %reset : !firrtl.reset, !firrtl.reset
}
// CHECK-LABEL: firrtl.module @MergeNetsChild2
// CHECK-SAME: in %reset: !firrtl.asyncreset
firrtl.module @MergeNetsChild2(in %reset: !firrtl.reset) {
  // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
  %localReset = firrtl.wire : !firrtl.reset
  firrtl.connect %localReset, %reset : !firrtl.reset, !firrtl.reset
}
// CHECK-LABEL: firrtl.module @MergeNetsTop
firrtl.module @MergeNetsTop(in %reset: !firrtl.asyncreset) {
  // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
  %localReset = firrtl.wire : !firrtl.reset
  firrtl.connect %localReset, %reset : !firrtl.reset, !firrtl.asyncreset
  // CHECK: %c1_reset = firrtl.instance c1 @MergeNetsChild1(in reset: !firrtl.asyncreset)
  // CHECK: %c2_reset = firrtl.instance c2 @MergeNetsChild2(in reset: !firrtl.asyncreset)
  %c1_reset = firrtl.instance c1 @MergeNetsChild1(in reset: !firrtl.reset)
  %c2_reset = firrtl.instance c2 @MergeNetsChild2(in reset: !firrtl.reset)
  firrtl.connect %c1_reset, %localReset : !firrtl.reset, !firrtl.reset
  firrtl.connect %c2_reset, %localReset : !firrtl.reset, !firrtl.reset
}

// Should support casting to other types
// CHECK-LABEL: firrtl.module @CastingToOtherTypes
firrtl.module @CastingToOtherTypes(in %a: !firrtl.uint<1>, out %v: !firrtl.uint<1>, out %w: !firrtl.sint<1>, out %x: !firrtl.clock, out %y: !firrtl.asyncreset) {
  // CHECK: %r = firrtl.wire : !firrtl.uint<1>
  %r = firrtl.wire : !firrtl.reset
  %0 = firrtl.asUInt %r : (!firrtl.reset) -> !firrtl.uint<1>
  %1 = firrtl.asSInt %r : (!firrtl.reset) -> !firrtl.sint<1>
  %2 = firrtl.asClock %r : (!firrtl.reset) -> !firrtl.clock
  %3 = firrtl.asAsyncReset %r : (!firrtl.reset) -> !firrtl.asyncreset
  firrtl.connect %r, %a : !firrtl.reset, !firrtl.uint<1>
  firrtl.connect %v, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %w, %1 : !firrtl.sint<1>, !firrtl.sint<1>
  firrtl.connect %x, %2 : !firrtl.clock, !firrtl.clock
  firrtl.connect %y, %3 : !firrtl.asyncreset, !firrtl.asyncreset
}

// Should work across Module boundaries
// CHECK-LABEL: firrtl.module @ModuleBoundariesChild
// CHECK-SAME: in %childReset: !firrtl.uint<1>
firrtl.module @ModuleBoundariesChild(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = firrtl.constant 123 : !firrtl.uint
  // CHECK: %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.connect %r, %x : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}
// CHECK-LABEL: firrtl.module @ModuleBoundariesTop
firrtl.module @ModuleBoundariesTop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  // CHECK: {{.*}} = firrtl.instance c @ModuleBoundariesChild(in clock: !firrtl.clock, in childReset: !firrtl.uint<1>, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  %c_clock, %c_childReset, %c_x, %c_z = firrtl.instance c @ModuleBoundariesChild(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  firrtl.connect %c_clock, %clock : !firrtl.clock, !firrtl.clock
  firrtl.connect %c_childReset, %reset : !firrtl.reset, !firrtl.uint<1>
  firrtl.connect %c_x, %x : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %z, %c_z : !firrtl.uint<8>, !firrtl.uint<8>
}

// Should work across multiple Module boundaries
// CHECK-LABEL: firrtl.module @MultipleModuleBoundariesChild
// CHECK-SAME: in %resetIn: !firrtl.uint<1>
// CHECK-SAME: out %resetOut: !firrtl.uint<1>
firrtl.module @MultipleModuleBoundariesChild(in %resetIn: !firrtl.reset, out %resetOut: !firrtl.reset) {
  firrtl.connect %resetOut, %resetIn : !firrtl.reset, !firrtl.reset
}
// CHECK-LABEL: firrtl.module @MultipleModuleBoundariesTop
firrtl.module @MultipleModuleBoundariesTop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  // CHECK: {{.*}} = firrtl.instance c @MultipleModuleBoundariesChild(in resetIn: !firrtl.uint<1>, out resetOut: !firrtl.uint<1>)
  %c_resetIn, %c_resetOut = firrtl.instance c @MultipleModuleBoundariesChild(in resetIn: !firrtl.reset, out resetOut: !firrtl.reset)
  firrtl.connect %c_resetIn, %reset : !firrtl.reset, !firrtl.uint<1>
  %c123_ui = firrtl.constant 123 : !firrtl.uint
  // CHECK: %r = firrtl.regreset %clock, %c_resetOut, %c123_ui : !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %c_resetOut, %c123_ui : !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.connect %r, %x : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// Should work in nested and flipped aggregates with connect
// CHECK-LABEL: firrtl.module @NestedAggregates
// CHECK-SAME: out %buzz: !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: asyncreset, c: uint<1>>, 2>, bar: vector<bundle<a: asyncreset, b flip: asyncreset, c: uint<8>>, 2>>
firrtl.module @NestedAggregates(out %buzz: !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>) {
  %0 = firrtl.subfield %buzz[bar] : !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>
  %1 = firrtl.subfield %buzz[foo] : !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>
  firrtl.connect %0, %1 : !firrtl.vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>, !firrtl.vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>
}

// Should work with deeply nested aggregates.
// CHECK-LABEL: firrtl.module @DeeplyNestedAggregates(in %reset: !firrtl.uint<1>, out %buzz: !firrtl.bundle<a: bundle<b: uint<1>>>) {
firrtl.module @DeeplyNestedAggregates(in %reset: !firrtl.uint<1>, out %buzz: !firrtl.bundle<a: bundle<b: reset>>) {
  %0 = firrtl.subfield %buzz[a] : !firrtl.bundle<a: bundle<b : reset>>
  %1 = firrtl.subfield %0[b] : !firrtl.bundle<b: reset>
  // CHECK: firrtl.connect %1, %reset : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %1, %reset : !firrtl.reset, !firrtl.uint<1>
}


// Should not crash if a ResetType has no drivers
// CHECK-LABEL: firrtl.module @DontCrashIfNoDrivers
// CHECK-SAME: out %out: !firrtl.uint<1>
firrtl.module @DontCrashIfNoDrivers(out %out: !firrtl.reset) {
  %c1_ui = firrtl.constant 1 : !firrtl.uint
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK: %w = firrtl.wire : !firrtl.uint<1>
  %w = firrtl.wire : !firrtl.reset
  firrtl.connect %out, %w : !firrtl.reset, !firrtl.reset
  // TODO: Enable the following once #1303 is fixed.
  // firrtl.connect %out, %c1_ui : !firrtl.reset, !firrtl.uint
  firrtl.connect %out, %c1_ui1 : !firrtl.reset, !firrtl.uint<1>
}

// Should allow concrete reset types to overrule invalidation
// CHECK-LABEL: firrtl.module @ConcreteResetOverruleInvalid
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @ConcreteResetOverruleInvalid(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %out, %invalid_reset : !firrtl.reset, !firrtl.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should default to BoolType for Resets that are only invalidated
// CHECK-LABEL: firrtl.module @DefaultToBool
// CHECK-SAME: out %out: !firrtl.uint<1>
firrtl.module @DefaultToBool(out %out: !firrtl.reset) {
  // CHECK: %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %out, %invalid_reset : !firrtl.reset, !firrtl.reset
}

// Should not error if component of ResetType is invalidated and connected to an AsyncResetType
// CHECK-LABEL: firrtl.module @OverrideInvalidWithDifferentResetType
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @OverrideInvalidWithDifferentResetType(in %cond: !firrtl.uint<1>, in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %out, %invalid_reset : !firrtl.reset, !firrtl.reset
  firrtl.when %cond  {
    firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
  }
}

// Should allow ResetType to drive AsyncResets or UInt<1>
// CHECK-LABEL: firrtl.module @ResetDrivesAsyncResetOrBool1
firrtl.module @ResetDrivesAsyncResetOrBool1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK: %w = firrtl.wire : !firrtl.uint<1>
  %w = firrtl.wire : !firrtl.reset
  firrtl.connect %w, %in : !firrtl.reset, !firrtl.uint<1>
  firrtl.connect %out, %w : !firrtl.uint<1>, !firrtl.reset
}
// CHECK-LABEL: firrtl.module @ResetDrivesAsyncResetOrBool2
firrtl.module @ResetDrivesAsyncResetOrBool2(out %foo: !firrtl.bundle<a flip: uint<1>>, in %bar: !firrtl.bundle<a flip: uint<1>>) {
  // CHECK: %w = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
  %w = firrtl.wire : !firrtl.bundle<a flip: reset>
  firrtl.connect %foo, %w : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: reset>
  firrtl.connect %w, %bar : !firrtl.bundle<a flip: reset>, !firrtl.bundle<a flip: uint<1>>
}
// CHECK-LABEL: firrtl.module @ResetDrivesAsyncResetOrBool3
firrtl.module @ResetDrivesAsyncResetOrBool3(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK: %w = firrtl.wire : !firrtl.uint<1>
  %w = firrtl.wire : !firrtl.reset
  firrtl.connect %w, %in : !firrtl.reset, !firrtl.uint<1>
  firrtl.connect %out, %w : !firrtl.uint<1>, !firrtl.reset
}

// Should support inferring modules that would dedup differently
// CHECK-LABEL: firrtl.module @DedupDifferentlyChild1
// CHECK-SAME: in %childReset: !firrtl.uint<1>
firrtl.module @DedupDifferentlyChild1(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = firrtl.constant 123 : !firrtl.uint
  // CHECK: %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.connect %r, %x : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}
// CHECK-LABEL: firrtl.module @DedupDifferentlyChild2
// CHECK-SAME: in %childReset: !firrtl.asyncreset
firrtl.module @DedupDifferentlyChild2(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = firrtl.constant 123 : !firrtl.uint
  // CHECK: %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.asyncreset, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.connect %r, %x : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}
// CHECK-LABEL: firrtl.module @DedupDifferentlyTop
firrtl.module @DedupDifferentlyTop(in %clock: !firrtl.clock, in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.asyncreset, in %x: !firrtl.vector<uint<8>, 2>, out %z: !firrtl.vector<uint<8>, 2>) {
  // CHECK: {{.*}} = firrtl.instance c1 @DedupDifferentlyChild1(in clock: !firrtl.clock, in childReset: !firrtl.uint<1>
  %c1_clock, %c1_childReset, %c1_x, %c1_z = firrtl.instance c1 @DedupDifferentlyChild1(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  firrtl.connect %c1_clock, %clock : !firrtl.clock, !firrtl.clock
  firrtl.connect %c1_childReset, %reset1 : !firrtl.reset, !firrtl.uint<1>
  %0 = firrtl.subindex %x[0] : !firrtl.vector<uint<8>, 2>
  firrtl.connect %c1_x, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  %1 = firrtl.subindex %z[0] : !firrtl.vector<uint<8>, 2>
  firrtl.connect %1, %c1_z : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: {{.*}} = firrtl.instance c2 @DedupDifferentlyChild2(in clock: !firrtl.clock, in childReset: !firrtl.asyncreset
  %c2_clock, %c2_childReset, %c2_x, %c2_z = firrtl.instance c2 @DedupDifferentlyChild2(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  firrtl.connect %c2_clock, %clock : !firrtl.clock, !firrtl.clock
  firrtl.connect %c2_childReset, %reset2 : !firrtl.reset, !firrtl.asyncreset
  %2 = firrtl.subindex %x[1] : !firrtl.vector<uint<8>, 2>
  firrtl.connect %c2_x, %2 : !firrtl.uint<8>, !firrtl.uint<8>
  %3 = firrtl.subindex %z[1] : !firrtl.vector<uint<8>, 2>
  firrtl.connect %3, %c2_z : !firrtl.uint<8>, !firrtl.uint<8>
}

// Should infer based on what a component *drives* not just what drives it
// CHECK-LABEL: firrtl.module @InferBasedOnDriven
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @InferBasedOnDriven(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %w = firrtl.wire : !firrtl.asyncreset
  // CHECK: %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset
  %w = firrtl.wire : !firrtl.reset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %w, %invalid_reset : !firrtl.reset, !firrtl.reset
  firrtl.connect %out, %w : !firrtl.reset, !firrtl.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should infer from connections, ignoring the fact that the invalidation wins
// CHECK-LABEL: firrtl.module @InferIgnoreInvalidation
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @InferIgnoreInvalidation(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
  firrtl.connect %out, %invalid_reset : !firrtl.reset, !firrtl.reset
}

// Should not propagate type info from downstream across a cast
// CHECK-LABEL: firrtl.module @DontPropagateUpstreamAcrossCast
// CHECK-SAME: out %out0: !firrtl.asyncreset
// CHECK-SAME: out %out1: !firrtl.uint<1>
firrtl.module @DontPropagateUpstreamAcrossCast(in %in0: !firrtl.asyncreset, in %in1: !firrtl.uint<1>, out %out0: !firrtl.reset, out %out1: !firrtl.reset) {
  // CHECK: %w = firrtl.wire : !firrtl.uint<1>
  %w = firrtl.wire : !firrtl.reset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %w, %invalid_reset : !firrtl.reset, !firrtl.reset
  %0 = firrtl.asAsyncReset %w : (!firrtl.reset) -> !firrtl.asyncreset
  firrtl.connect %out0, %0 : !firrtl.reset, !firrtl.asyncreset
  firrtl.connect %out1, %w : !firrtl.reset, !firrtl.reset
  firrtl.connect %out0, %in0 : !firrtl.reset, !firrtl.asyncreset
  firrtl.connect %out1, %in1 : !firrtl.reset, !firrtl.uint<1>
}

// Should take into account both internal and external constraints on Module port types
// CHECK-LABEL: firrtl.module @InternalAndExternalChild
// CHECK-SAME: out %o: !firrtl.asyncreset
firrtl.module @InternalAndExternalChild(in %i: !firrtl.asyncreset, out %o: !firrtl.reset) {
  firrtl.connect %o, %i : !firrtl.reset, !firrtl.asyncreset
}
// CHECK-LABEL: firrtl.module @InternalAndExternalTop
firrtl.module @InternalAndExternalTop(in %in: !firrtl.asyncreset, out %out: !firrtl.asyncreset) {
  // CHECK: {{.*}} = firrtl.instance c @InternalAndExternalChild(in i: !firrtl.asyncreset, out o: !firrtl.asyncreset)
  %c_i, %c_o = firrtl.instance c @InternalAndExternalChild(in i: !firrtl.asyncreset, out o: !firrtl.reset)
  firrtl.connect %c_i, %in : !firrtl.asyncreset, !firrtl.asyncreset
  firrtl.connect %out, %c_o : !firrtl.asyncreset, !firrtl.reset
}

// Should not crash on combinational loops
// CHECK-LABEL: firrtl.module @NoCrashOnCombLoop
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @NoCrashOnCombLoop(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  %w0 = firrtl.wire : !firrtl.reset
  %w1 = firrtl.wire : !firrtl.reset
  firrtl.connect %w0, %in : !firrtl.reset, !firrtl.asyncreset
  firrtl.connect %w0, %w1 : !firrtl.reset, !firrtl.reset
  firrtl.connect %w1, %w0 : !firrtl.reset, !firrtl.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should not treat a single `invalidvalue` connected to different resets as a
// connection of the resets themselves.
// CHECK-LABEL: firrtl.module @InvalidValueShouldNotConnect
// CHECK-SAME: out %r0: !firrtl.asyncreset
// CHECK-SAME: out %r1: !firrtl.asyncreset
// CHECK-SAME: out %r2: !firrtl.uint<1>
// CHECK-SAME: out %r3: !firrtl.uint<1>
firrtl.module @InvalidValueShouldNotConnect(
  in %ar: !firrtl.asyncreset,
  in %sr: !firrtl.uint<1>,
  out %r0: !firrtl.reset,
  out %r1: !firrtl.reset,
  out %r2: !firrtl.reset,
  out %r3: !firrtl.reset
) {
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %r0, %invalid_reset : !firrtl.reset, !firrtl.reset
  firrtl.connect %r1, %invalid_reset : !firrtl.reset, !firrtl.reset
  firrtl.connect %r2, %invalid_reset : !firrtl.reset, !firrtl.reset
  firrtl.connect %r3, %invalid_reset : !firrtl.reset, !firrtl.reset
  firrtl.connect %r0, %ar : !firrtl.reset, !firrtl.asyncreset
  firrtl.connect %r1, %ar : !firrtl.reset, !firrtl.asyncreset
  firrtl.connect %r2, %sr : !firrtl.reset, !firrtl.uint<1>
  firrtl.connect %r3, %sr : !firrtl.reset, !firrtl.uint<1>
}

// Should properly adjust the type of external modules.
// CHECK-LABEL: firrtl.extmodule @ShouldAdjustExtModule1
// CHECK-SAME: in reset: !firrtl.uint<1>
firrtl.extmodule @ShouldAdjustExtModule1(in reset: !firrtl.reset)
// CHECK-LABEL: firrtl.module @ShouldAdjustExtModule2
// CHECK: %x_reset = firrtl.instance x @ShouldAdjustExtModule1(in reset: !firrtl.uint<1>)
firrtl.module @ShouldAdjustExtModule2() {
  %x_reset = firrtl.instance x @ShouldAdjustExtModule1(in reset: !firrtl.reset)
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  firrtl.connect %x_reset, %c1_ui1 : !firrtl.reset, !firrtl.uint<1>
}

// Should not crash if there are connects with foreign types.
// CHECK-LABEL: firrtl.module @ForeignTypes
firrtl.module @ForeignTypes(out %out: !firrtl.reset) {
  %0 = firrtl.wire : index
  %1 = firrtl.wire : index
  firrtl.strictconnect %0, %1 : index
  // CHECK-NEXT: [[W0:%.+]] = firrtl.wire : index
  // CHECK-NEXT: [[W1:%.+]] = firrtl.wire : index
  // CHECK-NEXT: firrtl.strictconnect [[W0]], [[W1]] : index
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  firrtl.connect %out, %c1_ui1 : !firrtl.reset, !firrtl.uint<1>
}


//===----------------------------------------------------------------------===//
// Full Async Reset
//===----------------------------------------------------------------------===//


// CHECK-LABEL: firrtl.module @ConsumeIgnoreAnno
// CHECK-NOT: IgnoreFullAsyncResetAnnotation
firrtl.module @ConsumeIgnoreAnno() attributes {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} {
}

// CHECK-LABEL: firrtl.module @ConsumeResetAnnoPort
// CHECK-NOT: FullAsyncResetAnnotation
firrtl.module @ConsumeResetAnnoPort(in %outerReset: !firrtl.asyncreset) attributes {portAnnotations = [[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
}

// CHECK-LABEL: firrtl.module @ConsumeResetAnnoWire
firrtl.module @ConsumeResetAnnoWire(in %outerReset: !firrtl.asyncreset) {
  // CHECK: %innerReset = firrtl.wire
  // CHECK-NOT: FullAsyncResetAnnotation
  %innerReset = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
}

} // firrtl.circuit

// -----
// Reset-less registers should inherit the annotated async reset signal.
firrtl.circuit "Top" {
  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %init: !firrtl.uint<1>, in %in: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset ) attributes {
    portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %reg1 = firrtl.regreset sym @reg1 %clock, %extraReset, %c0_ui8
    %reg1 = firrtl.reg sym @reg1 %clock : !firrtl.uint<8>
    firrtl.connect %reg1, %in : !firrtl.uint<8>, !firrtl.uint<8>

    // Existing async reset remains untouched.
    // CHECK: %reg2 = firrtl.regreset %clock, %reset, %c1_ui8
    %reg2 = firrtl.regreset %clock, %reset, %c1_ui8 : !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %reg2, %in : !firrtl.uint<8>, !firrtl.uint<8>

    // Existing sync reset is moved to mux.
    // CHECK: %reg3 = firrtl.regreset %clock, %extraReset, %c0_ui8
    // CHECK: %0 = firrtl.mux(%init, %c1_ui8, %reg3)
    // CHECK: %1 = firrtl.mux(%init, %c1_ui8, %in)
    // CHECK: firrtl.connect %reg3, %1
    %reg3 = firrtl.regreset %clock, %init, %c1_ui8 : !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %reg3, %in : !firrtl.uint<8>, !firrtl.uint<8>

    // Factoring of sync reset into mux works through subfield op.
    // CHECK: %reg4 = firrtl.regreset %clock, %extraReset, %2
    // CHECK: %4 = firrtl.mux(%init, %reset4, %reg4)
    // CHECK: %5 = firrtl.subfield %reset4[a]
    // CHECK: %6 = firrtl.subfield %reg4[a]
    // CHECK: %7 = firrtl.mux(%init, %5, %in)
    // CHECK: firrtl.connect %6, %7
    %reset4 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %reg4 = firrtl.regreset %clock, %init, %reset4 : !firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    %0 = firrtl.subfield %reg4[a] : !firrtl.bundle<a: uint<8>>
    firrtl.connect %0, %in : !firrtl.uint<8>, !firrtl.uint<8>

    // Factoring of sync reset into mux works through subindex op.
    // CHECK: %reg5 = firrtl.regreset %clock, %extraReset, %8
    // CHECK: %10 = firrtl.mux(%init, %reset5, %reg5)
    // CHECK: firrtl.connect %reg5, %10
    // CHECK: %11 = firrtl.subindex %reset5[0]
    // CHECK: %12 = firrtl.subindex %reg5[0]
    // CHECK: %13 = firrtl.mux(%init, %11, %in)
    // CHECK: firrtl.connect %12, %13
    %reset5 = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %reg5 = firrtl.regreset %clock, %init, %reset5 : !firrtl.uint<1>, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    %1 = firrtl.subindex %reg5[0] : !firrtl.vector<uint<8>, 1>
    firrtl.connect %1, %in : !firrtl.uint<8>, !firrtl.uint<8>

    // Factoring of sync reset into mux works through subaccess op.
    // CHECK: %reg6 = firrtl.regreset %clock, %extraReset, %14 
    // CHECK: %16 = firrtl.mux(%init, %reset6, %reg6)
    // CHECK: firrtl.connect %reg6, %16
    // CHECK: %17 = firrtl.subaccess %reset6[%in]
    // CHECK: %18 = firrtl.subaccess %reg6[%in]
    // CHECK: %19 = firrtl.mux(%init, %17, %in)
    // CHECK: firrtl.connect %18, %19
    %reset6 = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %reg6 = firrtl.regreset %clock, %init, %reset6 : !firrtl.uint<1>, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    %2 = firrtl.subaccess %reg6[%in] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    firrtl.connect %2, %in : !firrtl.uint<8>, !firrtl.uint<8>

    // Subfields that are never assigned to should not leave unused reset
    // subfields behind.
    // CHECK-NOT: firrtl.subfield %reset4[a]
    // CHECK: %20 = firrtl.subfield %reg4[a]
    %3 = firrtl.subfield %reg4[a] : !firrtl.bundle<a: uint<8>>
  }
}

// -----
// Async reset inference should be able to construct reset values for aggregate
// types.
firrtl.circuit "Top" {
  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // CHECK: %c0_ui = firrtl.constant 0 : !firrtl.uint
    // CHECK: %reg_uint = firrtl.regreset %clock, %reset, %c0_ui
    %reg_uint = firrtl.reg %clock : !firrtl.uint
    // CHECK: %c0_si = firrtl.constant 0 : !firrtl.sint
    // CHECK: %reg_sint = firrtl.regreset %clock, %reset, %c0_si
    %reg_sint = firrtl.reg %clock : !firrtl.sint
    // CHECK: %0 = firrtl.wire : !firrtl.bundle<a: uint<8>, b: bundle<x: uint<8>, y: uint<8>>>
    // CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %1 = firrtl.subfield %0[a]
    // CHECK: firrtl.strictconnect %1, %c0_ui8
    // CHECK: %2 = firrtl.wire : !firrtl.bundle<x: uint<8>, y: uint<8>>
    // CHECK: %3 = firrtl.subfield %2[x]
    // CHECK: firrtl.strictconnect %3, %c0_ui8
    // CHECK: %4 = firrtl.subfield %2[y]
    // CHECK: firrtl.strictconnect %4, %c0_ui8
    // CHECK: %5 = firrtl.subfield %0[b]
    // CHECK: firrtl.strictconnect %5, %2
    // CHECK: %reg_bundle = firrtl.regreset %clock, %reset, %0
    %reg_bundle = firrtl.reg %clock : !firrtl.bundle<a: uint<8>, b: bundle<x: uint<8>, y: uint<8>>>
    // CHECK: %6 = firrtl.wire : !firrtl.vector<uint<8>, 4>
    // CHECK: %c0_ui8_0 = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %7 = firrtl.subindex %6[0]
    // CHECK: firrtl.strictconnect %7, %c0_ui8_0
    // CHECK: %8 = firrtl.subindex %6[1]
    // CHECK: firrtl.strictconnect %8, %c0_ui8_0
    // CHECK: %9 = firrtl.subindex %6[2]
    // CHECK: firrtl.strictconnect %9, %c0_ui8_0
    // CHECK: %10 = firrtl.subindex %6[3]
    // CHECK: firrtl.strictconnect %10, %c0_ui8_0
    // CHECK: %reg_vector = firrtl.regreset %clock, %reset, %6
    %reg_vector = firrtl.reg %clock : !firrtl.vector<uint<8>, 4>
  }
}

// -----
// Reset should reuse ports if name and type matches.
firrtl.circuit "ReusePorts" {
  // CHECK-LABEL: firrtl.module @Child
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
  firrtl.module @Child(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %reg = firrtl.reg %clock : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @BadName
  // CHECK-SAME: in %reset: !firrtl.asyncreset,
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %existingReset: !firrtl.asyncreset
  // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
  firrtl.module @BadName(in %clock: !firrtl.clock, in %existingReset: !firrtl.asyncreset) {
    %reg = firrtl.reg %clock : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @BadType
  // CHECK-SAME: in %reset_0: !firrtl.asyncreset,
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %reset: !firrtl.uint<1>
  // CHECK: %reg = firrtl.regreset %clock, %reset_0, %c0_ui8
  firrtl.module @BadType(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ReusePorts
  firrtl.module @ReusePorts(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // CHECK: %child_clock, %child_reset = firrtl.instance child
    // CHECK: firrtl.strictconnect %child_reset, %reset
    // CHECK: %badName_reset, %badName_clock, %badName_existingReset = firrtl.instance badName
    // CHECK: firrtl.strictconnect %badName_reset, %reset
    // CHECK: %badType_reset_0, %badType_clock, %badType_reset = firrtl.instance badType
    // CHECK: firrtl.strictconnect %badType_reset_0, %reset
    %child_clock, %child_reset = firrtl.instance child @Child(in clock: !firrtl.clock, in reset: !firrtl.asyncreset)
    %badName_clock, %badName_existingReset = firrtl.instance badName @BadName(in clock: !firrtl.clock, in existingReset: !firrtl.asyncreset)
    %badType_clock, %badType_reset = firrtl.instance badType @BadType(in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
  }
}

// -----
// Infer async reset: nested
firrtl.circuit "FullAsyncNested" {
  // CHECK-LABEL: firrtl.module @FullAsyncNestedDeeper
  firrtl.module @FullAsyncNestedDeeper(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: %io_out_REG = firrtl.regreset %clock, %reset, %c1_ui1
    %io_out_REG = firrtl.regreset %clock, %reset, %c1_ui1 : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<8>
    firrtl.connect %io_out_REG, %io_in : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %io_out, %io_out_REG : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @FullAsyncNestedChild
  firrtl.module @FullAsyncNestedChild(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) {
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncNestedDeeper(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %inst_reset, %reset : !firrtl.asyncreset, !firrtl.asyncreset
    firrtl.connect %inst_io_in, %io_in : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: %io_out_REG = firrtl.regreset %clock, %reset, %c0_ui8
    %io_out_REG = firrtl.reg %clock : !firrtl.uint<8>
    // CHECK: %io_out_REG_NO = firrtl.reg %clock : !firrtl.uint<8>
    %io_out_REG_NO = firrtl.reg %clock {annotations = [{class = "sifive.enterprise.firrtl.ExcludeMemFromMemToRegOfVec"}]}: !firrtl.uint<8>
    firrtl.connect %io_out_REG, %io_in : !firrtl.uint<8>, !firrtl.uint<8>
    %0 = firrtl.add %io_out_REG, %inst_io_out : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    %1 = firrtl.bits %0 7 to 0 : (!firrtl.uint<9>) -> !firrtl.uint<8>
    firrtl.connect %io_out, %1 : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @FullAsyncNested
  firrtl.module @FullAsyncNested(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) attributes {
    portAnnotations=[[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}], [], []] } {
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncNestedChild(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %inst_reset, %reset : !firrtl.asyncreset, !firrtl.asyncreset
    firrtl.connect %io_out, %inst_io_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %inst_io_in, %io_in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}


// -----
// Infer async reset: excluded
// TODO: Check that no extraReset port present
firrtl.circuit "FullAsyncExcluded" {
  // CHECK-LABEL: firrtl.module @FullAsyncExcludedChild
  // CHECK-SAME: (in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>)
  firrtl.module @FullAsyncExcludedChild(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} {
    // CHECK: %io_out_REG = firrtl.reg %clock
    %io_out_REG = firrtl.reg %clock : !firrtl.uint<8>
    firrtl.connect %io_out_REG, %io_in : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %io_out, %io_out_REG : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @FullAsyncExcluded
  firrtl.module @FullAsyncExcluded(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset) attributes {
     portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // CHECK: %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncExcludedChild
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncExcludedChild(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %inst_reset, %reset : !firrtl.asyncreset, !firrtl.asyncreset
    firrtl.connect %io_out, %inst_io_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %inst_io_in, %io_in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}


// -----

// Local wire as async reset should be moved before all its uses.
firrtl.circuit "WireShouldDominate" {
  // CHECK-LABEL: firrtl.module @WireShouldDominate
  firrtl.module @WireShouldDominate(in %clock: !firrtl.clock) {
    %reg = firrtl.reg %clock : !firrtl.uint<8> // gets wired to localReset
    %localReset = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
  }
}

// -----

// Local node as async reset should be moved before all its uses if its input
// value dominates the target location in the module.
firrtl.circuit "MovableNodeShouldDominate" {
  // CHECK-LABEL: firrtl.module @MovableNodeShouldDominate
  firrtl.module @MovableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // does not block move of node
    %reg = firrtl.reg %clock : !firrtl.uint<8> // gets wired to localReset
    %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %0 = firrtl.asAsyncReset %ui1
    // CHECK-NEXT: %localReset = firrtl.node sym @theReset %0
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
  }
}

// -----

// Local node as async reset should be replaced by a wire and moved before all
// its uses if its input value does not dominate the target location in the
// module.
firrtl.circuit "UnmovableNodeShouldDominate" {
  // CHECK-LABEL: firrtl.module @UnmovableNodeShouldDominate
  firrtl.module @UnmovableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.uint<8> // gets wired to localReset
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = firrtl.wire sym @theReset
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: %0 = firrtl.asAsyncReset %ui1
    // CHECK-NEXT: firrtl.strictconnect %localReset, %0
  }
}

// -----

// Move of local async resets should work across blocks.
firrtl.circuit "MoveAcrossBlocks1" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks1
  firrtl.module @MoveAcrossBlocks1(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    firrtl.when %ui1 {
      %reg = firrtl.reg %clock : !firrtl.uint<8> // gets wired to localReset
    }
    firrtl.when %ui1 {
      %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    }
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: firrtl.when %ui1 {
    // CHECK-NEXT:   [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT:   %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
    // CHECK-NEXT: firrtl.when %ui1 {
    // CHECK-NEXT:   [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT:   firrtl.strictconnect %localReset, [[TMP]]
    // CHECK-NEXT: }
  }
}

// -----

firrtl.circuit "MoveAcrossBlocks2" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks2
  firrtl.module @MoveAcrossBlocks2(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    firrtl.when %ui1 {
      %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    }
    firrtl.when %ui1 {
      %reg = firrtl.reg %clock : !firrtl.uint<8> // gets wired to localReset
    }
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: firrtl.when %ui1 {
    // CHECK-NEXT:   [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT:   firrtl.strictconnect %localReset, [[TMP]]
    // CHECK-NEXT: }
    // CHECK-NEXT: firrtl.when %ui1 {
    // CHECK-NEXT:   [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT:   %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
  }
}

// -----

firrtl.circuit "MoveAcrossBlocks3" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks3
  firrtl.module @MoveAcrossBlocks3(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    %reg = firrtl.reg %clock : !firrtl.uint<8> // gets wired to localReset
    firrtl.when %ui1 {
      %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    }
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: firrtl.when %ui1 {
    // CHECK-NEXT:   [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT:   firrtl.strictconnect %localReset, [[TMP]]
    // CHECK-NEXT: }
  }
}

// -----

firrtl.circuit "MoveAcrossBlocks4" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks4
  firrtl.module @MoveAcrossBlocks4(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    firrtl.when %ui1 {
      %reg = firrtl.reg %clock : !firrtl.uint<8> // gets wired to localReset
    }
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: firrtl.when %ui1 {
    // CHECK-NEXT:   [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT:   %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
    // CHECK-NEXT: [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT: firrtl.strictconnect %localReset, [[TMP]]
  }
}

// -----

firrtl.circuit "SubAccess" {
  firrtl.module @SubAccess(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %init: !firrtl.uint<1>, in %in: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset ) attributes {
    // CHECK-LABEL: firrtl.module @SubAccess
    portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<2>
    %arr = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %reg6 = firrtl.regreset %clock, %init, %c1_ui8 : !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>
    %2 = firrtl.subaccess %arr[%reg6] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<2>
    firrtl.strictconnect %2, %in : !firrtl.uint<8>
    // CHECK:  %reg6 = firrtl.regreset %clock, %extraReset, %c0_ui2  : !firrtl.asyncreset, !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT: %0 = firrtl.mux(%init, %c1_ui2, %reg6)
    // CHECK: firrtl.connect %reg6, %0
    // CHECK-NEXT:  %[[v0:.+]] = firrtl.subaccess %arr[%reg6] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<2>
    // CHECK-NEXT:  firrtl.strictconnect %[[v0]], %in : !firrtl.uint<8>

  }
}

// -----

// This is a regression check to ensure that a zero-width register gets a proper
// reset value.
// CHECK-LABEL: firrtl.module @ZeroWidthRegister
firrtl.circuit "ZeroWidthRegister" {
  firrtl.module @ZeroWidthRegister(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    %reg = firrtl.reg %clock : !firrtl.uint<0>
    // CHECK-NEXT: [[TMP:%.+]] = firrtl.constant 0 : !firrtl.uint<0>
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %reset, [[TMP]]
  }
}

// -----

// Check that unaffected fields ("data") are not being affected by width
// inference. See https://github.com/llvm/circt/issues/2857.
// CHECK-LABEL: firrtl.module @ZeroLengthVectorInBundle1
firrtl.circuit "ZeroLengthVectorInBundle1"  {
  firrtl.module @ZeroLengthVectorInBundle1(out %out: !firrtl.bundle<resets: vector<reset, 0>, data flip: uint<3>>) {
    %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<reset, 0>, data flip: uint<3>>
    %invalid = firrtl.invalidvalue : !firrtl.vector<reset, 0>
    firrtl.strictconnect %0, %invalid : !firrtl.vector<reset, 0>
    // CHECK-NEXT: %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<uint<1>, 0>, data flip: uint<3>>
    // CHECK-NEXT: %invalid = firrtl.invalidvalue : !firrtl.vector<uint<1>, 0>
    // CHECK-NEXT: firrtl.strictconnect %0, %invalid : !firrtl.vector<uint<1>, 0>
  }
}

// -----

// CHECK-LABEL: firrtl.module @ZeroLengthVectorInBundle2
firrtl.circuit "ZeroLengthVectorInBundle2"  {
  firrtl.module @ZeroLengthVectorInBundle2(out %out: !firrtl.bundle<resets: vector<bundle<a: reset>, 0>, data flip: uint<3>>) {
    %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<bundle<a: reset>, 0>, data flip: uint<3>>
    %invalid = firrtl.invalidvalue : !firrtl.vector<bundle<a: reset>, 0>
    firrtl.strictconnect %0, %invalid : !firrtl.vector<bundle<a: reset>, 0>
    // CHECK-NEXT: %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<bundle<a: uint<1>>, 0>, data flip: uint<3>>
    // CHECK-NEXT: %invalid = firrtl.invalidvalue : !firrtl.vector<bundle<a: uint<1>>, 0>
    // CHECK-NEXT: firrtl.strictconnect %0, %invalid : !firrtl.vector<bundle<a: uint<1>>, 0>
  }
}

// -----

// Resets nested underneath a zero-length vector should infer to `UInt<1>`.
// CHECK-LABEL: firrtl.module @ZeroVecBundle
// CHECK-SAME: in %a: !firrtl.vector<bundle<x: uint<1>>, 0>
// CHECK-SAME: out %b: !firrtl.vector<bundle<x: uint<1>>, 0>
firrtl.circuit "ZeroVecBundle"  {
  firrtl.module @ZeroVecBundle(in %a: !firrtl.vector<bundle<x: uint<1>>, 0>, out %b: !firrtl.vector<bundle<x: reset>, 0>) {
    %w = firrtl.wire : !firrtl.vector<bundle<x: reset>, 0>
    firrtl.connect %b, %w : !firrtl.vector<bundle<x: reset>, 0>, !firrtl.vector<bundle<x: reset>, 0>
    // CHECK-NEXT: %w = firrtl.wire : !firrtl.vector<bundle<x: uint<1>>, 0>
    // CHECK-NEXT: firrtl.connect %b, %w : !firrtl.vector<bundle<x: uint<1>>, 0>, !firrtl.vector<bundle<x: uint<1>>, 0>
  }
}

// -----

// Resets directly in a zero-length vector should infer to `UInt<1>`.
// CHECK-LABEL: firrtl.module @ZeroVec
// CHECK-SAME: in %a: !firrtl.bundle<x: vector<uint<1>, 0>>
// CHECK-SAME: out %b: !firrtl.bundle<x: vector<uint<1>, 0>>
firrtl.circuit "ZeroVec"  {
  firrtl.module @ZeroVec(in %a: !firrtl.bundle<x: vector<reset, 0>>, out %b: !firrtl.bundle<x: vector<reset, 0>>) {
    firrtl.connect %b, %a : !firrtl.bundle<x: vector<reset, 0>>, !firrtl.bundle<x: vector<reset, 0>>
    // CHECK-NEXT: firrtl.connect %b, %a : !firrtl.bundle<x: vector<uint<1>, 0>>, !firrtl.bundle<x: vector<uint<1>, 0>>
  }
}

// -----

// CHECK-LABEL: "RefReset"
firrtl.circuit "RefReset" {
  // CHECK-LABEL: firrtl.module private @SendReset
  // CHECK-SAME: in %r: !firrtl.asyncreset
  // CHECK-SAME: out %ref: !firrtl.ref<asyncreset>
  // CHECK-NEXT: send %r : !firrtl.asyncreset
  // CHECK-NEXT: ref<asyncreset>
  firrtl.module private @SendReset(in %r: !firrtl.reset, out %ref: !firrtl.ref<reset>) {
    %ref_r = firrtl.ref.send %r : !firrtl.reset
    firrtl.strictconnect %ref, %ref_r : !firrtl.ref<reset>
  }
  // CHECK-LABEL: firrtl.module @RefReset
  // CHECK-NEXT: in r: !firrtl.asyncreset
  // CHECK-SAME: out ref: !firrtl.ref<asyncreset>
  // CHECK-NEXT: !firrtl.asyncreset, !firrtl.asyncreset
  // CHECK-NEXT: %s_ref : !firrtl.ref<asyncreset>
  firrtl.module @RefReset(in %r: !firrtl.asyncreset) {
    %s_r, %s_ref = firrtl.instance s @SendReset(in r: !firrtl.reset, out ref: !firrtl.ref<reset>)
    firrtl.connect %s_r, %r : !firrtl.reset, !firrtl.asyncreset
    %reset = firrtl.ref.resolve %s_ref : !firrtl.ref<reset>
  }
}
