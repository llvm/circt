// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-materialize-debug-info,firrtl-lower-intrinsics)))' %s | FileCheck %s

// Integration check: when MaterializeDebugInfo runs BEFORE LowerIntrinsics
// (the order firtool pins in lib/Firtool/Firtool.cpp), names covered by a
// `circt_debug_var` intrinsic must end up with a single rich `dbg.variable`
// from LowerIntrinsics -- not a duplicate basic one from MaterializeDebugInfo.

// CHECK-LABEL: firrtl.module @PipelineDedup
firrtl.circuit "PipelineDedup" {
  firrtl.module @PipelineDedup() {
    %w = firrtl.wire : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "w", typeName: none = "UInt">
      %w : (!firrtl.uint<8>) -> ()

    %u = firrtl.wire : !firrtl.uint<4>

    // The rich variable for `w` (with typeName) is the only `dbg.variable "w"`.
    // CHECK:     dbg.variable "w", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK-NOT: dbg.variable "w"
    //
    // The wire `u` has no intrinsic, so MaterializeDebugInfo emits a basic
    // `dbg.variable "u"` (no typeName, ground type passes through).
    // CHECK:     dbg.variable "u", %{{.*}} : !firrtl.uint<4>
    // CHECK-NOT: dbg.variable "u"
  }
}

// Port covered by circt_debug_var: MaterializeDebugInfo must NOT emit a basic
// dbg.variable for the port; only LowerIntrinsics emits the rich one.
//
// CHECK-LABEL: firrtl.module @PipelinePortCovered
firrtl.circuit "PipelinePortCovered" {
  firrtl.module @PipelinePortCovered(in %clk: !firrtl.clock,
                                     in %data: !firrtl.uint<16>) {
    firrtl.int.generic "circt_debug_var"
      <name: none = "data", typeName: none = "UInt">
      %data : (!firrtl.uint<16>) -> ()

    // MaterializeDebugInfo runs first and emits ports in declaration order,
    // skipping `data` (covered by the intrinsic). LowerIntrinsics then emits
    // the rich `dbg.variable "data"` at the intrinsic site.
    // CHECK:     dbg.variable "clk", %{{.*}} : !firrtl.clock
    // CHECK-NOT: dbg.variable "clk"
    // CHECK:     dbg.variable "data", %{{.*}} typeName "UInt" : !firrtl.uint<16>
    // CHECK-NOT: dbg.variable "data"
  }
}

// Two distinct SSA values with the same name in different scopes: the
// Value-based skip-set must not conflate them. Here only `w1` is covered by
// an intrinsic; `w2` (a different SSA value, same name conceptually but a
// separate wire) must still get a materialized dbg.variable.
//
// Note: firrtl.when with nested wire declarations is not supported in this IR
// form -- both arms share the same basic-block scope so a wire declared in one
// arm is visible in the other, making true name-shadowing impossible at this
// IR level. Instead we use two separate wires with distinct SSA values to
// verify that the DenseSet<Value> key (not a name key) correctly discriminates
// them. True shadowed-name coverage would require a when-aware IR pass that
// is out of scope here.
//
// CHECK-LABEL: firrtl.module @PipelineScopeDistinct
firrtl.circuit "PipelineScopeDistinct" {
  firrtl.module @PipelineScopeDistinct() {
    %w1 = firrtl.wire : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "w1", typeName: none = "UInt">
      %w1 : (!firrtl.uint<8>) -> ()

    // w2 is a *different* SSA value from w1 (even if logically same-named).
    // The Value-based skip-set only covers w1; w2 must be materialized.
    %w2 = firrtl.wire : !firrtl.uint<8>

    // LowerIntrinsics emits the rich `dbg.variable "w1"` at the intrinsic site
    // (between the two wires). MaterializeDebugInfo emits the basic `w2` after
    // its wire. The Value-keyed skip-set must not conflate the two SSA values.
    // CHECK:     dbg.variable "w1", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK-NOT: dbg.variable "w1"
    // CHECK:     dbg.variable "w2", %{{.*}} : !firrtl.uint<8>
    // CHECK-NOT: dbg.variable "w2"
  }
}

// 0-operand circt_debug_var (memory / name-only form): MaterializeDebugInfo
// must skip the wire/port/reg matched by name, deferring to LowerIntrinsics.
//
// CHECK-LABEL: firrtl.module @PipelineZeroOperand
firrtl.circuit "PipelineZeroOperand" {
  firrtl.module @PipelineZeroOperand(in %clk: !firrtl.clock,
                                     in %data: !firrtl.uint<16>) {
    // 0-operand intrinsic covering a wire by name.
    %mem = firrtl.wire : !firrtl.uint<32>
    firrtl.int.generic "circt_debug_var"
      <name: none = "mem", typeName: none = "UInt"> : () -> ()

    // 0-operand intrinsic covering a port by name.
    firrtl.int.generic "circt_debug_var"
      <name: none = "data", typeName: none = "UInt"> : () -> ()

    // `clk` has no intrinsic -> MaterializeDebugInfo emits it at block start.
    // `mem` is name-covered by a 0-operand intrinsic -> MaterializeDebugInfo
    //   skips it; LowerIntrinsics emits the rich dbg.variable (with typeName)
    //   at the intrinsic site (after the wire declaration).
    // `data` is name-covered by a 0-operand intrinsic -> same.
    // Emission order: clk (materialized at block start), then mem (intrinsic
    // site after wire), then data (intrinsic site after mem's intrinsic).
    // CHECK:     dbg.variable "clk", %{{.*}} : !firrtl.clock
    // CHECK-NOT: dbg.variable "clk"
    // CHECK:     dbg.variable "mem", %{{.*}} typeName "UInt" : !firrtl.uint<32>
    // CHECK-NOT: dbg.variable "mem"
    // CHECK:     dbg.variable "data", %{{.*}} typeName "UInt" : !firrtl.uint<16>
    // CHECK-NOT: dbg.variable "data"
  }
}

// firrtl.reg covered by 1-operand intrinsic: MaterializeDebugInfo must not
// emit a basic dbg.variable for the reg; only LowerIntrinsics emits the rich
// one. An uncovered reg must still be materialized.
//
// CHECK-LABEL: firrtl.module @PipelineRegCovered
firrtl.circuit "PipelineRegCovered" {
  firrtl.module @PipelineRegCovered(in %clk: !firrtl.clock) {
    %r_covered = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "r_covered", typeName: none = "UInt">
      %r_covered : (!firrtl.uint<8>) -> ()

    %r_plain = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<4>

    // CHECK:     dbg.variable "clk", %{{.*}} : !firrtl.clock
    // CHECK:     dbg.variable "r_covered", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK-NOT: dbg.variable "r_covered"
    // CHECK:     dbg.variable "r_plain", %{{.*}} : !firrtl.uint<4>
    // CHECK-NOT: dbg.variable "r_plain"
  }
}

// firrtl.node covered by 1-operand intrinsic: same dedup logic applies.
// An uncovered node must still be materialized.
//
// CHECK-LABEL: firrtl.module @PipelineNodeCovered
firrtl.circuit "PipelineNodeCovered" {
  firrtl.module @PipelineNodeCovered(in %in: !firrtl.uint<8>) {
    %n_covered = firrtl.node %in : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "n_covered", typeName: none = "UInt">
      %n_covered : (!firrtl.uint<8>) -> ()

    %n_plain = firrtl.node %in : !firrtl.uint<8>

    // CHECK:     dbg.variable "in", %{{.*}} : !firrtl.uint<8>
    // CHECK:     dbg.variable "n_covered", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK-NOT: dbg.variable "n_covered"
    // CHECK:     dbg.variable "n_plain", %{{.*}} : !firrtl.uint<8>
    // CHECK-NOT: dbg.variable "n_plain"
  }
}
