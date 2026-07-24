// RUN: circt-opt %s -verify-diagnostics --split-input-file --lower-seq-to-sv

// An asynchronous reset whose signal is defined inside a guarded region (here an
// sv.read_inout of a wire declared in the sv.ifdef) does not dominate the
// module-level async-reset initial block, so it cannot drive the reset
// initialization. Reject it rather than emit invalid IR.
sv.macro.decl @SOME_MACRO
hw.module @RegionLocalAsyncReset(in %clk: !seq.clock, in %d: i32) {
  %c0 = hw.constant 0 : i32
  sv.ifdef @SOME_MACRO {
    %w = sv.wire : !hw.inout<i1>
    %rst = sv.read_inout %w : !hw.inout<i1>
    // expected-error @below {{asynchronous reset is defined inside a guarded region and cannot drive the module-level reset initialization}}
    %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
  }
}
