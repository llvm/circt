// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets))' --verify-diagnostics --split-input-file %s | FileCheck %s

// Tests extracted from:
// - github.com/chipsalliance/firrtl:
//   - test/scala/firrtlTests/InferResetsSpec.scala

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
  firrtl.matchingconnect %localReset, %reset : !firrtl.reset
}
// CHECK-LABEL: firrtl.module @MergeNetsChild2
// CHECK-SAME: in %reset: !firrtl.asyncreset
firrtl.module @MergeNetsChild2(in %reset: !firrtl.reset) {
  // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
  %localReset = firrtl.wire : !firrtl.reset
  firrtl.matchingconnect %localReset, %reset : !firrtl.reset
}
// CHECK-LABEL: firrtl.module @MergeNetsTop
firrtl.module @MergeNetsTop(in %reset: !firrtl.asyncreset) {
  // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
  %localReset = firrtl.wire : !firrtl.reset
  %t = firrtl.resetCast %reset : (!firrtl.asyncreset) -> !firrtl.reset
  firrtl.matchingconnect %localReset, %t : !firrtl.reset
  // CHECK: %c1_reset = firrtl.instance c1 @MergeNetsChild1(in reset: !firrtl.asyncreset)
  // CHECK: %c2_reset = firrtl.instance c2 @MergeNetsChild2(in reset: !firrtl.asyncreset)
  %c1_reset = firrtl.instance c1 @MergeNetsChild1(in reset: !firrtl.reset)
  %c2_reset = firrtl.instance c2 @MergeNetsChild2(in reset: !firrtl.reset)
  firrtl.matchingconnect %c1_reset, %localReset : !firrtl.reset
  firrtl.matchingconnect %c2_reset, %localReset : !firrtl.reset
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
  %4 = firrtl.resetCast %a : (!firrtl.uint<1>) -> !firrtl.reset
  firrtl.matchingconnect %r, %4 : !firrtl.reset
  firrtl.matchingconnect %v, %0 : !firrtl.uint<1>
  firrtl.matchingconnect %w, %1 : !firrtl.sint<1>
  firrtl.matchingconnect %x, %2 : !firrtl.clock
  firrtl.matchingconnect %y, %3 : !firrtl.asyncreset
}

// Should support const-casts
// CHECK-LABEL: firrtl.module @ConstCast
firrtl.module @ConstCast(in %a: !firrtl.const.uint<1>) {
  // CHECK: %r = firrtl.wire : !firrtl.uint<1>
  %r = firrtl.wire : !firrtl.reset
  %0 = firrtl.resetCast %a : (!firrtl.const.uint<1>) -> !firrtl.const.reset
  %1 = firrtl.constCast %0 : (!firrtl.const.reset) -> !firrtl.reset
  firrtl.matchingconnect %r, %1 : !firrtl.reset
}

// Should work across Module boundaries
// CHECK-LABEL: firrtl.module @ModuleBoundariesChild
// CHECK-SAME: in %childReset: !firrtl.uint<1>
firrtl.module @ModuleBoundariesChild(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = firrtl.constant 123 : !firrtl.uint
  // CHECK: %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.matchingconnect %r, %x : !firrtl.uint<8>
  firrtl.matchingconnect %z, %r : !firrtl.uint<8>
}
// CHECK-LABEL: firrtl.module @ModuleBoundariesTop
firrtl.module @ModuleBoundariesTop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  // CHECK: {{.*}} = firrtl.instance c @ModuleBoundariesChild(in clock: !firrtl.clock, in childReset: !firrtl.uint<1>, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  %c_clock, %c_childReset, %c_x, %c_z = firrtl.instance c @ModuleBoundariesChild(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  firrtl.matchingconnect %c_clock, %clock : !firrtl.clock
  firrtl.connect %c_childReset, %reset : !firrtl.reset, !firrtl.uint<1>
  firrtl.matchingconnect %c_x, %x : !firrtl.uint<8>
  firrtl.matchingconnect %z, %c_z : !firrtl.uint<8>
}

// Should work across multiple Module boundaries
// CHECK-LABEL: firrtl.module @MultipleModuleBoundariesChild
// CHECK-SAME: in %resetIn: !firrtl.uint<1>
// CHECK-SAME: out %resetOut: !firrtl.uint<1>
firrtl.module @MultipleModuleBoundariesChild(in %resetIn: !firrtl.reset, out %resetOut: !firrtl.reset) {
  firrtl.matchingconnect %resetOut, %resetIn : !firrtl.reset
}
// CHECK-LABEL: firrtl.module @MultipleModuleBoundariesTop
firrtl.module @MultipleModuleBoundariesTop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  // CHECK: {{.*}} = firrtl.instance c @MultipleModuleBoundariesChild(in resetIn: !firrtl.uint<1>, out resetOut: !firrtl.uint<1>)
  %c_resetIn, %c_resetOut = firrtl.instance c @MultipleModuleBoundariesChild(in resetIn: !firrtl.reset, out resetOut: !firrtl.reset)
  firrtl.connect %c_resetIn, %reset : !firrtl.reset, !firrtl.uint<1>
  %c123_ui = firrtl.constant 123 : !firrtl.uint
  // CHECK: %r = firrtl.regreset %clock, %c_resetOut, %c123_ui : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %c_resetOut, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.matchingconnect %r, %x : !firrtl.uint<8>
  firrtl.matchingconnect %z, %r : !firrtl.uint<8>
}

// Should work in nested and flipped aggregates with connect
// CHECK-LABEL: firrtl.module @NestedAggregates
// CHECK-SAME: out %buzz: !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: asyncreset, c: uint<1>>, 2>, bar: vector<bundle<a: asyncreset, b flip: asyncreset, c: uint<8>>, 2>>
firrtl.module @NestedAggregates(out %buzz: !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>) {
  %0 = firrtl.subfield %buzz[bar] : !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>
  %1 = firrtl.subfield %buzz[foo] : !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>
  firrtl.connect %0, %1 :  !firrtl.vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>, !firrtl.vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>
}

// Should work with deeply nested aggregates.
// CHECK-LABEL: firrtl.module @DeeplyNestedAggregates(in %reset: !firrtl.uint<1>, out %buzz: !firrtl.bundle<a: bundle<b: uint<1>>>) {
firrtl.module @DeeplyNestedAggregates(in %reset: !firrtl.uint<1>, out %buzz: !firrtl.bundle<a: bundle<b: reset>>) {
  %0 = firrtl.subfield %buzz[a] : !firrtl.bundle<a: bundle<b : reset>>
  %1 = firrtl.subfield %0[b] : !firrtl.bundle<b: reset>
  // CHECK: firrtl.connect %1, %reset : !firrtl.uint<1>
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
  firrtl.matchingconnect %out, %w : !firrtl.reset
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
  firrtl.matchingconnect %out, %invalid_reset : !firrtl.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should default to BoolType for Resets that are only invalidated
// CHECK-LABEL: firrtl.module @DefaultToBool
// CHECK-SAME: out %out: !firrtl.uint<1>
firrtl.module @DefaultToBool(out %out: !firrtl.reset) {
  // CHECK: %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.matchingconnect %out, %invalid_reset : !firrtl.reset
}

// Should not error if component of ResetType is invalidated and connected to an AsyncResetType
// CHECK-LABEL: firrtl.module @OverrideInvalidWithDifferentResetType
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @OverrideInvalidWithDifferentResetType(in %cond: !firrtl.uint<1>, in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.matchingconnect %out, %invalid_reset : !firrtl.reset
  firrtl.when %cond : !firrtl.uint<1>  {
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
  // CHECK: %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.matchingconnect %r, %x : !firrtl.uint<8>
  firrtl.matchingconnect %z, %r : !firrtl.uint<8>
}
// CHECK-LABEL: firrtl.module @DedupDifferentlyChild2
// CHECK-SAME: in %childReset: !firrtl.asyncreset
firrtl.module @DedupDifferentlyChild2(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = firrtl.constant 123 : !firrtl.uint
  // CHECK: %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint<8>
  %r = firrtl.regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  firrtl.matchingconnect %r, %x : !firrtl.uint<8>
  firrtl.matchingconnect %z, %r : !firrtl.uint<8>
}
// CHECK-LABEL: firrtl.module @DedupDifferentlyTop
firrtl.module @DedupDifferentlyTop(in %clock: !firrtl.clock, in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.asyncreset, in %x: !firrtl.vector<uint<8>, 2>, out %z: !firrtl.vector<uint<8>, 2>) {
  // CHECK: {{.*}} = firrtl.instance c1 @DedupDifferentlyChild1(in clock: !firrtl.clock, in childReset: !firrtl.uint<1>
  %c1_clock, %c1_childReset, %c1_x, %c1_z = firrtl.instance c1 @DedupDifferentlyChild1(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  firrtl.matchingconnect %c1_clock, %clock : !firrtl.clock
  firrtl.connect %c1_childReset, %reset1 : !firrtl.reset, !firrtl.uint<1>
  %0 = firrtl.subindex %x[0] : !firrtl.vector<uint<8>, 2>
  firrtl.matchingconnect %c1_x, %0 : !firrtl.uint<8>
  %1 = firrtl.subindex %z[0] : !firrtl.vector<uint<8>, 2>
  firrtl.matchingconnect %1, %c1_z : !firrtl.uint<8>
  // CHECK: {{.*}} = firrtl.instance c2 @DedupDifferentlyChild2(in clock: !firrtl.clock, in childReset: !firrtl.asyncreset
  %c2_clock, %c2_childReset, %c2_x, %c2_z = firrtl.instance c2 @DedupDifferentlyChild2(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  firrtl.matchingconnect %c2_clock, %clock : !firrtl.clock
  firrtl.connect %c2_childReset, %reset2 : !firrtl.reset, !firrtl.asyncreset
  %2 = firrtl.subindex %x[1] : !firrtl.vector<uint<8>, 2>
  firrtl.matchingconnect %c2_x, %2 : !firrtl.uint<8>
  %3 = firrtl.subindex %z[1] : !firrtl.vector<uint<8>, 2>
  firrtl.matchingconnect %3, %c2_z : !firrtl.uint<8>
}

// Should infer based on what a component *drives* not just what drives it
// CHECK-LABEL: firrtl.module @InferBasedOnDriven
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @InferBasedOnDriven(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %w = firrtl.wire : !firrtl.asyncreset
  // CHECK: %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset
  %w = firrtl.wire : !firrtl.reset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.matchingconnect %w, %invalid_reset : !firrtl.reset
  firrtl.matchingconnect %out, %w : !firrtl.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should infer from connections, ignoring the fact that the invalidation wins
// CHECK-LABEL: firrtl.module @InferIgnoreInvalidation
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @InferIgnoreInvalidation(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.connect %out, %in : !firrtl.reset, !firrtl.asyncreset
  firrtl.matchingconnect %out, %invalid_reset : !firrtl.reset
}

// Should not propagate type info from downstream across a cast
// CHECK-LABEL: firrtl.module @DontPropagateUpstreamAcrossCast
// CHECK-SAME: out %out0: !firrtl.asyncreset
// CHECK-SAME: out %out1: !firrtl.uint<1>
firrtl.module @DontPropagateUpstreamAcrossCast(in %in0: !firrtl.asyncreset, in %in1: !firrtl.uint<1>, out %out0: !firrtl.reset, out %out1: !firrtl.reset) {
  // CHECK: %w = firrtl.wire : !firrtl.uint<1>
  %w = firrtl.wire : !firrtl.reset
  %invalid_reset = firrtl.invalidvalue : !firrtl.reset
  firrtl.matchingconnect %w, %invalid_reset : !firrtl.reset
  %0 = firrtl.asAsyncReset %w : (!firrtl.reset) -> !firrtl.asyncreset
  firrtl.connect %out0, %0 : !firrtl.reset, !firrtl.asyncreset
  firrtl.matchingconnect %out1, %w : !firrtl.reset
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
  firrtl.matchingconnect %c_i, %in : !firrtl.asyncreset
  firrtl.connect %out, %c_o : !firrtl.asyncreset, !firrtl.reset
}

// Should not crash on combinational loops
// CHECK-LABEL: firrtl.module @NoCrashOnCombLoop
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @NoCrashOnCombLoop(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  %w0 = firrtl.wire : !firrtl.reset
  %w1 = firrtl.wire : !firrtl.reset
  firrtl.connect %w0, %in : !firrtl.reset, !firrtl.asyncreset
  firrtl.matchingconnect %w0, %w1 : !firrtl.reset
  firrtl.matchingconnect %w1, %w0 : !firrtl.reset
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
  firrtl.matchingconnect %r0, %invalid_reset : !firrtl.reset
  firrtl.matchingconnect %r1, %invalid_reset : !firrtl.reset
  firrtl.matchingconnect %r2, %invalid_reset : !firrtl.reset
  firrtl.matchingconnect %r3, %invalid_reset : !firrtl.reset
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
  firrtl.matchingconnect %0, %1 : index
  // CHECK-NEXT: [[W0:%.+]] = firrtl.wire : index
  // CHECK-NEXT: [[W1:%.+]] = firrtl.wire : index
  // CHECK-NEXT: firrtl.matchingconnect [[W0]], [[W1]] : index
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  firrtl.connect %out, %c1_ui1 : !firrtl.reset, !firrtl.uint<1>
}
} // firrtl.circuit

// -----

// Check that unaffected fields ("data") are not being affected by width
// inference. See https://github.com/llvm/circt/issues/2857.
// CHECK-LABEL: firrtl.module @ZeroLengthVectorInBundle1
firrtl.circuit "ZeroLengthVectorInBundle1"  {
  firrtl.module @ZeroLengthVectorInBundle1(out %out: !firrtl.bundle<resets: vector<reset, 0>, data flip: uint<3>>) {
    %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<reset, 0>, data flip: uint<3>>
    %invalid = firrtl.invalidvalue : !firrtl.vector<reset, 0>
    firrtl.matchingconnect %0, %invalid : !firrtl.vector<reset, 0>
    // CHECK-NEXT: %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<uint<1>, 0>, data flip: uint<3>>
    // CHECK-NEXT: %invalid = firrtl.invalidvalue : !firrtl.vector<uint<1>, 0>
    // CHECK-NEXT: firrtl.matchingconnect %0, %invalid : !firrtl.vector<uint<1>, 0>
  }
}

// -----

// CHECK-LABEL: firrtl.module @ZeroLengthVectorInBundle2
firrtl.circuit "ZeroLengthVectorInBundle2"  {
  firrtl.module @ZeroLengthVectorInBundle2(out %out: !firrtl.bundle<resets: vector<bundle<a: reset>, 0>, data flip: uint<3>>) {
    %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<bundle<a: reset>, 0>, data flip: uint<3>>
    %invalid = firrtl.invalidvalue : !firrtl.vector<bundle<a: reset>, 0>
    firrtl.matchingconnect %0, %invalid : !firrtl.vector<bundle<a: reset>, 0>
    // CHECK-NEXT: %0 = firrtl.subfield %out[resets] : !firrtl.bundle<resets: vector<bundle<a: uint<1>>, 0>, data flip: uint<3>>
    // CHECK-NEXT: %invalid = firrtl.invalidvalue : !firrtl.vector<bundle<a: uint<1>>, 0>
    // CHECK-NEXT: firrtl.matchingconnect %0, %invalid : !firrtl.vector<bundle<a: uint<1>>, 0>
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
    firrtl.matchingconnect %b, %w : !firrtl.vector<bundle<x: reset>, 0>
    // CHECK-NEXT: %w = firrtl.wire : !firrtl.vector<bundle<x: uint<1>>, 0>
    // CHECK-NEXT: firrtl.matchingconnect %b, %w : !firrtl.vector<bundle<x: uint<1>>, 0>
  }
}

// -----

// Resets directly in a zero-length vector should infer to `UInt<1>`.
// CHECK-LABEL: firrtl.module @ZeroVec
// CHECK-SAME: in %a: !firrtl.bundle<x: vector<uint<1>, 0>>
// CHECK-SAME: out %b: !firrtl.bundle<x: vector<uint<1>, 0>>
firrtl.circuit "ZeroVec"  {
  firrtl.module @ZeroVec(in %a: !firrtl.bundle<x: vector<reset, 0>>, out %b: !firrtl.bundle<x: vector<reset, 0>>) {
    firrtl.matchingconnect %b, %a : !firrtl.bundle<x: vector<reset, 0>>
    // CHECK-NEXT: firrtl.matchingconnect %b, %a : !firrtl.bundle<x: vector<uint<1>, 0>>
  }
}

// -----

// CHECK-LABEL: "RefReset"
firrtl.circuit "RefReset" {
  // CHECK-LABEL: firrtl.module private @SendReset
  // CHECK-SAME: in %r: !firrtl.asyncreset
  // CHECK-SAME: out %ref: !firrtl.probe<asyncreset>
  // CHECK-NEXT: send %r : !firrtl.asyncreset
  // CHECK-NEXT: probe<asyncreset>
  firrtl.module private @SendReset(in %r: !firrtl.reset, out %ref: !firrtl.probe<reset>) {
    %ref_r = firrtl.ref.send %r : !firrtl.reset
    firrtl.ref.define %ref, %ref_r : !firrtl.probe<reset>
  }
  // CHECK-LABEL: firrtl.module @RefReset
  // CHECK-NEXT: in r: !firrtl.asyncreset
  // CHECK-SAME: out ref: !firrtl.probe<asyncreset>
  // CHECK-NEXT: !firrtl.asyncreset
  // CHECK-NEXT: %s_ref : !firrtl.probe<asyncreset>
  firrtl.module @RefReset(in %r: !firrtl.asyncreset) {
    %s_r, %s_ref = firrtl.instance s @SendReset(in r: !firrtl.reset, out ref: !firrtl.probe<reset>)
    firrtl.connect %s_r, %r : !firrtl.reset, !firrtl.asyncreset
    %reset = firrtl.ref.resolve %s_ref : !firrtl.probe<reset>
  }
}

// -----

// CHECK-LABEL: "RefCastReset"
firrtl.circuit "RefCastReset" {
  // CHECK-LABEL: firrtl.module private @SendCastReset
  // CHECK-SAME: in %r: !firrtl.asyncreset
  // CHECK-SAME: out %ref: !firrtl.probe<asyncreset>
  // CHECK-NEXT: send %r : !firrtl.asyncreset
  // CHECK-NEXT: probe<asyncreset>
  firrtl.module private @SendCastReset(in %r: !firrtl.asyncreset, out %ref: !firrtl.probe<reset>) {
    %ref_r = firrtl.ref.send %r : !firrtl.asyncreset
    %ref_r_cast = firrtl.ref.cast %ref_r : (!firrtl.probe<asyncreset>) -> !firrtl.probe<reset>
    firrtl.ref.define %ref, %ref_r_cast : !firrtl.probe<reset>
  }
  // CHECK-LABEL: firrtl.module @RefCastReset
  // CHECK-NEXT: in r: !firrtl.asyncreset
  // CHECK-SAME: out ref: !firrtl.probe<asyncreset>
  // CHECK-NEXT: : !firrtl.asyncreset
  // CHECK-NEXT: %s_ref : !firrtl.probe<asyncreset>
  firrtl.module @RefCastReset(in %r: !firrtl.asyncreset) {
    %s_r, %s_ref = firrtl.instance s @SendCastReset(in r: !firrtl.asyncreset, out ref: !firrtl.probe<reset>)
    firrtl.matchingconnect %s_r, %r : !firrtl.asyncreset
    %reset = firrtl.ref.resolve %s_ref : !firrtl.probe<reset>
  }
}

// -----

// CHECK-LABEL: "RefCastAggReset"
firrtl.circuit "RefCastAggReset" {
   // CHECK-LABEL: firrtl.module private @ResetAggSource
   // CHECK-SAME: in %r: !firrtl.asyncreset,
   // CHECK-SAME: out %p: !firrtl.rwprobe<bundle<a: asyncreset, b: uint<1>>>,
   // CHECK-SAME: out %pconst: !firrtl.probe<bundle<a: asyncreset, b: const.uint<1>>>)
  // CHECK-NOT: : {{(const\.)?reset}}
  firrtl.module private @ResetAggSource(in %r: !firrtl.asyncreset, out %p: !firrtl.rwprobe<bundle<a: reset, b: reset>>, out %pconst: !firrtl.probe<bundle<a: reset, b: const.reset>>) {
    %x = firrtl.wire : !firrtl.reset
    %0 = firrtl.resetCast %r : (!firrtl.asyncreset) -> !firrtl.reset
    firrtl.matchingconnect %x, %0 : !firrtl.reset
    %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
    %zero = firrtl.node %c0_ui1 : !firrtl.const.uint<1>
    %bundle, %bundle_ref = firrtl.wire forceable : !firrtl.bundle<a: reset, b: reset>, !firrtl.rwprobe<bundle<a: reset, b: reset>>
    %1 = firrtl.subfield %bundle[b] : !firrtl.bundle<a: reset, b: reset>
    %2 = firrtl.subfield %bundle[a] : !firrtl.bundle<a: reset, b: reset>
    firrtl.matchingconnect %2, %x : !firrtl.reset
    %3 = firrtl.resetCast %zero : (!firrtl.const.uint<1>) -> !firrtl.const.reset
    %4 = firrtl.constCast %3 : (!firrtl.const.reset) -> !firrtl.reset
    firrtl.matchingconnect %1, %4 : !firrtl.reset
    firrtl.ref.define %p, %bundle_ref : !firrtl.rwprobe<bundle<a: reset, b: reset>>
    %bundle_const = firrtl.wire : !firrtl.bundle<a: reset, b: const.reset>
    %5 = firrtl.subfield %bundle_const[b] : !firrtl.bundle<a: reset, b: const.reset>
    %6 = firrtl.subfield %bundle_const[a] : !firrtl.bundle<a: reset, b: const.reset>
    firrtl.matchingconnect %6, %x : !firrtl.reset
    firrtl.matchingconnect %5, %3 : !firrtl.const.reset
    %7 = firrtl.ref.send %bundle_const : !firrtl.bundle<a: reset, b: const.reset>
    firrtl.ref.define %pconst, %7 : !firrtl.probe<bundle<a: reset, b: const.reset>>
  }
  // CHECK-LABEL: firrtl.module @RefCastAggReset
  // CHECK-SAME: in %r: !firrtl.asyncreset,
  // CHECK-SAME: out %a: !firrtl.probe<asyncreset>,
  // CHECK-SAME: out %b: !firrtl.probe<uint<1>>,
  // CHECK-SAME: out %pconst: !firrtl.probe<bundle<a: asyncreset, b: const.uint<1>>>)
  // CHECK-NOT: : {{(const\.)?reset}}
  firrtl.module @RefCastAggReset(in %r: !firrtl.asyncreset, out %a: !firrtl.probe<reset>, out %b: !firrtl.probe<reset>, out %pconst: !firrtl.probe<bundle<a: reset, b: const.reset>>) {
    %s_r, %s_p, %s_pconst = firrtl.instance s @ResetAggSource(in r: !firrtl.asyncreset, out p: !firrtl.rwprobe<bundle<a: reset, b: reset>>, out pconst: !firrtl.probe<bundle<a: reset, b: const.reset>>)
    %0 = firrtl.ref.sub %s_p[1] : !firrtl.rwprobe<bundle<a: reset, b: reset>>
    %1 = firrtl.ref.sub %s_p[0] : !firrtl.rwprobe<bundle<a: reset, b: reset>>
    firrtl.matchingconnect %s_r, %r : !firrtl.asyncreset
    %2 = firrtl.ref.cast %1 : (!firrtl.rwprobe<reset>) -> !firrtl.probe<reset>
    firrtl.ref.define %a, %2 : !firrtl.probe<reset>
    %3 = firrtl.ref.cast %0 : (!firrtl.rwprobe<reset>) -> !firrtl.probe<reset>
    firrtl.ref.define %b, %3 : !firrtl.probe<reset>
    firrtl.ref.define %pconst, %s_pconst : !firrtl.probe<bundle<a: reset, b: const.reset>>
  }
}

// -----

// Check resets are inferred through references to bundles w/flips.

// CHECK-LABEL: "RefResetBundle"
firrtl.circuit "RefResetBundle" {
  // CHECK-LABEL: firrtl.module @RefResetBundle
  // CHECK-NOT: firrtl.reset
  firrtl.module @RefResetBundle(in %driver: !firrtl.asyncreset, out %out: !firrtl.bundle<a: reset, b: reset>) {
  %r = firrtl.wire : !firrtl.bundle<a: reset, b flip: reset>
  %ref_r = firrtl.ref.send %r : !firrtl.bundle<a: reset, b flip: reset>
  %reset = firrtl.ref.resolve %ref_r : !firrtl.probe<bundle<a: reset, b: reset>>
  firrtl.matchingconnect %out, %reset : !firrtl.bundle<a: reset, b: reset>

   %r_a = firrtl.subfield %r[a] : !firrtl.bundle<a: reset, b flip: reset>
   %r_b = firrtl.subfield %r[b] : !firrtl.bundle<a: reset, b flip: reset>
   firrtl.connect %r_a, %driver : !firrtl.reset, !firrtl.asyncreset
   firrtl.connect %r_b, %driver : !firrtl.reset, !firrtl.asyncreset
  }
}

// -----

// Check resets are inferred through ref.sub.

// CHECK-LABEL: "RefResetSub"
firrtl.circuit "RefResetSub" {
  // CHECK-LABEL: firrtl.module @RefResetSub
  // CHECK-NOT: firrtl.reset
  firrtl.module @RefResetSub(in %driver: !firrtl.asyncreset, out %out_a : !firrtl.reset, out %out_b: !firrtl.vector<reset,2>) {
  %r = firrtl.wire : !firrtl.bundle<a: reset, b flip: vector<reset, 2>>
  %ref_r = firrtl.ref.send %r : !firrtl.bundle<a: reset, b flip: vector<reset, 2>>
  %ref_r_a = firrtl.ref.sub %ref_r[0] : !firrtl.probe<bundle<a: reset, b : vector<reset, 2>>>
  %reset_a = firrtl.ref.resolve %ref_r_a : !firrtl.probe<reset>

  %ref_r_b = firrtl.ref.sub %ref_r[1] : !firrtl.probe<bundle<a: reset, b : vector<reset, 2>>>
  %reset_b = firrtl.ref.resolve %ref_r_b : !firrtl.probe<vector<reset, 2>>

  firrtl.matchingconnect %out_a, %reset_a : !firrtl.reset
  firrtl.matchingconnect %out_b, %reset_b : !firrtl.vector<reset, 2>

   %r_a = firrtl.subfield %r[a] : !firrtl.bundle<a: reset, b flip: vector<reset, 2>>
   %r_b = firrtl.subfield %r[b] : !firrtl.bundle<a: reset, b flip: vector<reset, 2>>
   %r_b_0 = firrtl.subindex %r_b[0] : !firrtl.vector<reset, 2>
   %r_b_1 = firrtl.subindex %r_b[1] : !firrtl.vector<reset, 2>
   firrtl.connect %r_a, %driver : !firrtl.reset, !firrtl.asyncreset
   firrtl.connect %r_b_0, %driver : !firrtl.reset, !firrtl.asyncreset
   firrtl.connect %r_b_1, %driver : !firrtl.reset, !firrtl.asyncreset
  }
}

// -----

// CHECK-LABEL: "ConstReset"
firrtl.circuit "ConstReset" {
  // CHECK-LABEL: firrtl.module private @InfersConstAsync(in %r: !firrtl.const.asyncreset)
  firrtl.module private @InfersConstAsync(in %r: !firrtl.const.reset) {}

  // CHECK-LABEL: firrtl.module private @InfersConstSync(in %r: !firrtl.const.uint<1>)
  firrtl.module private @InfersConstSync(in %r: !firrtl.const.reset) {}

  // CHECK-LABEL: firrtl.module private @InfersAsync(in %r: !firrtl.asyncreset)
  firrtl.module private @InfersAsync(in %r: !firrtl.reset) {}

  // CHECK-LABEL: firrtl.module private @InfersSync(in %r: !firrtl.uint<1>)
  firrtl.module private @InfersSync(in %r: !firrtl.reset) {}

  firrtl.module @ConstReset(in %async: !firrtl.const.asyncreset, in %sync: !firrtl.const.uint<1>) {
    %constAsyncTarget = firrtl.instance infersConstAsync @InfersConstAsync(in r: !firrtl.const.reset)
    %constSyncTarget = firrtl.instance infersConstSync @InfersConstSync(in r: !firrtl.const.reset)
    %asyncTarget = firrtl.instance infersAsync @InfersAsync(in r: !firrtl.reset)
    %syncTarget = firrtl.instance infersSync @InfersSync(in r: !firrtl.reset)

    firrtl.connect %constAsyncTarget, %async : !firrtl.const.reset, !firrtl.const.asyncreset
    firrtl.connect %constSyncTarget, %sync : !firrtl.const.reset, !firrtl.const.uint<1>
    firrtl.connect %asyncTarget, %async : !firrtl.reset, !firrtl.const.asyncreset
    firrtl.connect %syncTarget, %sync : !firrtl.reset, !firrtl.const.uint<1>
  }
}

// -----

// CHECK-LABEL: "ConstAggReset"
firrtl.circuit "ConstAggReset" {
  // CHECK-LABEL: module @ConstAggReset
  // CHECK-NOT: : reset
  firrtl.module @ConstAggReset(in %in: !firrtl.const.bundle<a: reset, b: uint<1>>, out %out: !firrtl.bundle<a: asyncreset>, out %out2: !firrtl.bundle<a: reset, b: uint<1>>) {
    %out_a = firrtl.subfield %out[a] : !firrtl.bundle<a: asyncreset>
    %in_a = firrtl.subfield %in[a] : !firrtl.const.bundle<a: reset, b: uint<1>>
    %in_a_asyncreset = firrtl.resetCast %in_a : (!firrtl.const.reset) -> !firrtl.const.asyncreset
    %in_a_asyncreset_noconst = firrtl.constCast %in_a_asyncreset : (!firrtl.const.asyncreset) -> !firrtl.asyncreset
    firrtl.matchingconnect %out_a, %in_a_asyncreset_noconst : !firrtl.asyncreset

    %in_noconst = firrtl.constCast %in : (!firrtl.const.bundle<a: reset, b: uint<1>>) -> !firrtl.bundle<a: reset, b : uint<1>>
    firrtl.matchingconnect %out2, %in_noconst : !firrtl.bundle<a: reset, b: uint<1>>
  }
}

// -----

// CHECK-LABEL: "ConstAggCastReset"
firrtl.circuit "ConstAggCastReset" {
  // CHECK-LABEL: module @ConstAggCastReset
  // CHECK-NOT: : reset
  firrtl.module @ConstAggCastReset(in %in: !firrtl.const.bundle<a: reset, b: uint<1>>, out %out: !firrtl.bundle<a: asyncreset>, out %out2: !firrtl.bundle<a: reset, b: uint<1>>) {
    %out_a = firrtl.subfield %out[a] : !firrtl.bundle<a: asyncreset>
    %in_a = firrtl.subfield %in[a] : !firrtl.const.bundle<a: reset, b: uint<1>>
    // CHECK: constCast %{{.+}} : (!firrtl.const.asyncreset) -> !firrtl.asyncreset
    %in_a_noconst = firrtl.constCast %in_a : (!firrtl.const.reset) -> !firrtl.reset
    %in_a_asyncreset = firrtl.resetCast %in_a_noconst : (!firrtl.reset) -> !firrtl.asyncreset
    // CHECK-NEXT: matchingconnect
    firrtl.matchingconnect %out_a, %in_a_asyncreset : !firrtl.asyncreset
    // CHECK-NOT: : reset
    %in_noconst = firrtl.constCast %in : (!firrtl.const.bundle<a: reset, b: uint<1>>) -> !firrtl.bundle<a: reset, b : uint<1>>
    firrtl.matchingconnect %out2, %in_noconst : !firrtl.bundle<a: reset, b: uint<1>>
  }
}

// -----

// Check resets are inferred for forceable ops.

// CHECK-LABEL: "InferToRWProbe"
firrtl.circuit "InferToRWProbe" {
  // CHECK-LABEL: firrtl.module @InferToRWProbe
  // CHECK-NOT: firrtl.reset
  firrtl.module @InferToRWProbe(in %driver: !firrtl.asyncreset, out %out: !firrtl.bundle<a: reset, b: reset>) {
  %r, %r_rw = firrtl.wire forceable : !firrtl.bundle<a: reset, b flip: reset>, !firrtl.rwprobe<bundle<a: reset, b : reset>>
  %reset = firrtl.ref.resolve %r_rw : !firrtl.rwprobe<bundle<a: reset, b: reset>>
  firrtl.matchingconnect %out, %reset : !firrtl.bundle<a: reset, b: reset>

   %r_a = firrtl.subfield %r[a] : !firrtl.bundle<a: reset, b flip: reset>
   %r_b = firrtl.subfield %r[b] : !firrtl.bundle<a: reset, b flip: reset>
   firrtl.connect %r_a, %driver : !firrtl.reset, !firrtl.asyncreset
   firrtl.connect %r_b, %driver : !firrtl.reset, !firrtl.asyncreset
  }
}

// -----

// Check resets are traced through nodes

// CHECK-LABEL: "TraceThroughNodes"
// CHECK-NOT: firrtl.reset
firrtl.circuit "TraceThroughNodes" {
  firrtl.module @TraceThroughNodes(in %reset: !firrtl.asyncreset) {
    // CHECK: %node = firrtl.node %localReset : !firrtl.asyncreset
    // CHECK-NOT: firrtl.reset
    %localReset = firrtl.wire : !firrtl.reset
    firrtl.connect %localReset, %reset : !firrtl.reset, !firrtl.asyncreset

    %node = firrtl.node %localReset : !firrtl.reset

    %localReset2 = firrtl.wire : !firrtl.reset
    firrtl.matchingconnect %localReset2, %node: !firrtl.reset

    // CHECK: %node2 = firrtl.node %localReset2 : !firrtl.asyncreset
    %node2 = firrtl.node %localReset2 : !firrtl.reset
  }
}

// -----

// CHECK-LABEL: "RWProbeOp"
firrtl.circuit "RWProbeOp" {
  // CHECK: %out: !firrtl.bundle<a: asyncreset, b: asyncreset>
  // CHECK-SAME: %out2: !firrtl.asyncreset
  firrtl.module @RWProbeOp(in %driver: !firrtl.asyncreset, out %out: !firrtl.bundle<a: reset, b: reset>, out %out2: !firrtl.reset) {
    %r = firrtl.wire sym [<@r,0,public>,<@r_a,1,public>,<@r_b,2,public>] : !firrtl.bundle<a: reset, b flip: reset>

    // CHECK-COUNT-2: <a: asyncreset, b flip: asyncreset>
    %r_a = firrtl.subfield %r[a] : !firrtl.bundle<a: reset, b flip: reset>
    %r_b = firrtl.subfield %r[b] : !firrtl.bundle<a: reset, b flip: reset>
    // CHECK-COUNT-2: %driver : !firrtl.asyncreset
    firrtl.connect %r_a, %driver : !firrtl.reset, !firrtl.asyncreset
    firrtl.connect %r_b, %driver : !firrtl.reset, !firrtl.asyncreset

    // CHECK-NEXT: rwprobe<asyncreset>
    %r_a_rw = firrtl.ref.rwprobe <@RWProbeOp::@r_a> : !firrtl.rwprobe<reset>
    // CHECK-NEXT: rwprobe<asyncreset>
    %r_b_rw = firrtl.ref.rwprobe <@RWProbeOp::@r_b> : !firrtl.rwprobe<reset>
    // CHECK-NEXT: rwprobe<bundle<a: asyncreset, b: asyncreset>>
    %r_rw = firrtl.ref.rwprobe <@RWProbeOp::@r> : !firrtl.rwprobe<bundle<a: reset, b: reset>>

    // CHECK-NEXT: rwprobe<asyncreset>
    %r_b_read = firrtl.ref.resolve %r_b_rw : !firrtl.rwprobe<reset>
    // CHECK-NEXT: rwprobe<bundle<a: asyncreset, b: asyncreset>>
    %r_read = firrtl.ref.resolve %r_rw : !firrtl.rwprobe<bundle<a: reset, b: reset>>

    // CHECK-NEXT: firrtl.asyncreset
    firrtl.matchingconnect %out2, %r_b_read : !firrtl.reset
    // CHECK-NEXT: bundle<a: asyncreset, b: asyncreset>
    firrtl.matchingconnect %out, %r_read : !firrtl.bundle<a: reset, b: reset>
  }
}

// -----
firrtl.circuit "AsResetInferSync" {
  // CHECK-LABEL: firrtl.module @AsResetInferSync
  firrtl.module @AsResetInferSync() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %syncReset = firrtl.wire : !firrtl.uint<1>
    %reset = firrtl.wire : !firrtl.reset
    // CHECK-NOT: firrtl.asReset
    // CHECK: firrtl.connect %reset, %c0_ui1
    %0 = firrtl.asReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.reset
    firrtl.connect %reset, %0 : !firrtl.reset, !firrtl.reset
    firrtl.connect %reset, %syncReset : !firrtl.reset, !firrtl.uint<1>
  }
}

// -----
firrtl.circuit "AsResetInferAsync" {
  // CHECK-LABEL: firrtl.module @AsResetInferAsync
  firrtl.module @AsResetInferAsync() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %asyncReset = firrtl.wire : !firrtl.asyncreset
    %reset = firrtl.wire : !firrtl.reset
    // CHECK-NOT: firrtl.asReset
    // CHECK: [[ASYNC:%.+]] = firrtl.asAsyncReset %c0_ui1
    // CHECK: firrtl.connect %reset, [[ASYNC]]
    %0 = firrtl.asReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.reset
    firrtl.connect %reset, %0 : !firrtl.reset, !firrtl.reset
    firrtl.connect %reset, %asyncReset : !firrtl.reset, !firrtl.asyncreset
  }
}

// -----
// A module instantiated in two contexts, one forcing a specific reset kind and
// the other adapting to that choice using an `asReset` cast should work.
firrtl.circuit "TopA" {
  // CHECK: firrtl.module private @ChildA({{.+}}: !firrtl.asyncreset)
  // CHECK: firrtl.module private @ChildB({{.+}}: !firrtl.uint<1>)
  firrtl.module private @ChildA(in %reset: !firrtl.reset) {}
  firrtl.module private @ChildB(in %reset: !firrtl.reset) {}

  // CHECK-LABEL: firrtl.module @TopA
  firrtl.module @TopA(in %asyncReset: !firrtl.asyncreset, in %syncReset: !firrtl.uint<1>) {
    %a_reset = firrtl.instance a @ChildA(in reset: !firrtl.reset)
    %b_reset = firrtl.instance b @ChildB(in reset: !firrtl.reset)
    firrtl.connect %a_reset, %asyncReset : !firrtl.reset, !firrtl.asyncreset
    firrtl.connect %b_reset, %syncReset : !firrtl.reset, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @TopB
  firrtl.module @TopB(in %x: !firrtl.uint<1>, in %y: !firrtl.uint<1>) {
    // CHECK-NEXT: [[A:%.+]] = firrtl.instance a @ChildA
    // CHECK-NEXT: [[B:%.+]] = firrtl.instance b @ChildB
    %a_reset = firrtl.instance a @ChildA(in reset: !firrtl.reset)
    %b_reset = firrtl.instance b @ChildB(in reset: !firrtl.reset)
    // CHECK-NEXT: [[ASYNC:%.+]] = firrtl.asAsyncReset %x
    // CHECK-NOT: firrtl.asReset %y
    %0 = firrtl.asReset %x : (!firrtl.uint<1>) -> !firrtl.reset
    %1 = firrtl.asReset %y : (!firrtl.uint<1>) -> !firrtl.reset
    // CHECK-NEXT: firrtl.connect [[A]], [[ASYNC]]
    // CHECK-NEXT: firrtl.connect [[B]], %y
    firrtl.connect %a_reset, %0 : !firrtl.reset, !firrtl.reset
    firrtl.connect %b_reset, %1 : !firrtl.reset, !firrtl.reset
  }
}

// -----
// Test that InstanceChoiceOp works with reset inference
firrtl.circuit "InstanceChoiceTest" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  // CHECK-LABEL: firrtl.module @DefaultTarget
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  firrtl.module @DefaultTarget(in %reset: !firrtl.reset) {
    // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
    %localReset = firrtl.wire : !firrtl.reset
    firrtl.matchingconnect %localReset, %reset : !firrtl.reset
  }

  // CHECK-LABEL: firrtl.module @FPGATarget
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  firrtl.module @FPGATarget(in %reset: !firrtl.reset) {
    // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
    %localReset = firrtl.wire : !firrtl.reset
    firrtl.matchingconnect %localReset, %reset : !firrtl.reset
  }

  // CHECK-LABEL: firrtl.module @InstanceChoiceTest
  firrtl.module @InstanceChoiceTest(in %reset: !firrtl.asyncreset) {
    // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
    %localReset = firrtl.wire : !firrtl.reset
    %t = firrtl.resetCast %reset : (!firrtl.asyncreset) -> !firrtl.reset
    firrtl.matchingconnect %localReset, %t : !firrtl.reset
    // CHECK: %inst_reset = firrtl.instance_choice inst @DefaultTarget alternatives @Platform
    // CHECK-SAME: @FPGA -> @FPGATarget
    // CHECK-SAME: (in reset: !firrtl.asyncreset)
    %inst_reset = firrtl.instance_choice inst @DefaultTarget alternatives @Platform {
      @FPGA -> @FPGATarget,
    } (in reset: !firrtl.reset)
    firrtl.matchingconnect %inst_reset, %localReset : !firrtl.reset
  }

  // CHECK-LABEL: firrtl.module @Child
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  firrtl.module @Child(in %reset: !firrtl.reset) {
    // CHECK: %localReset = firrtl.wire : !firrtl.asyncreset
    %localReset = firrtl.wire : !firrtl.reset
    firrtl.matchingconnect %localReset, %reset : !firrtl.reset
  }

  // CHECK-LABEL: firrtl.module @PropBetweenChoice
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  firrtl.module @PropBetweenChoice(in %reset: !firrtl.reset) {
    // Make sure FPGATarget -> Child propagates the reset type.
    // CHECK: %inst_reset = firrtl.instance_choice inst @Child alternatives @Platform
    // CHECK-SAME: (in reset: !firrtl.asyncreset)
    %inst_reset = firrtl.instance_choice inst @Child alternatives @Platform {
      @FPGA -> @FPGATarget,
    } (in reset: !firrtl.reset)
    firrtl.matchingconnect %inst_reset, %reset : !firrtl.reset
  }
}

// -----

// Reset inference should look through `unsafe_domain_cast` so that reset
// networks connected across a domain cast are inferred to a concrete type.
// CHECK-LABEL: firrtl.circuit "UnsafeDomainCast"
firrtl.circuit "UnsafeDomainCast" {
  firrtl.domain @ClockDomain
  // CHECK-LABEL: firrtl.module @UnsafeDomainCast
  firrtl.module @UnsafeDomainCast(
    in %B: !firrtl.domain<@ClockDomain()>,
    in %a: !firrtl.uint<1>
  ) {
    // CHECK: %b = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.reset
    %0 = firrtl.resetCast %a : (!firrtl.uint<1>) -> !firrtl.reset
    %1 = firrtl.unsafe_domain_cast %0 domains[%B] : !firrtl.reset domains[!firrtl.domain<@ClockDomain()>]
    firrtl.matchingconnect %b, %1 : !firrtl.reset
  }
}
