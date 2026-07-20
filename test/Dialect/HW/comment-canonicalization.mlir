// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: hw.module @live_hw_wire
hw.module @live_hw_wire(in %a : i1, out out : i1) {
  // CHECK: %wire = hw.wire %a {comment = "live wire"} : i1
  %wire = hw.wire %a {comment = "live wire"} : i1
  hw.output %wire : i1
}

// CHECK-LABEL: hw.module @dead_hw_wire
// CHECK-NOT: hw.wire
hw.module @dead_hw_wire(in %a : i1) {
  %wire = hw.wire %a {comment = "dead wire"} : i1
}

// CHECK-LABEL: hw.module @live_firreg
hw.module @live_firreg(in %clk : !seq.clock, out out : i1) {
  // CHECK: %reg = seq.firreg %reg clock %clk {comment = "live reg"} : i1
  %reg = seq.firreg %reg clock %clk {comment = "live reg"} : i1
  hw.output %reg : i1
}

// A self-feedback use is not an external user and a comment is not a liveness
// root.
// CHECK-LABEL: hw.module @dead_firreg
// CHECK-NOT: seq.firreg
// CHECK-NOT: dead FIR register
hw.module @dead_firreg(in %clk : !seq.clock) {
  %reg = seq.firreg %reg clock %clk {comment = "dead FIR register"} : i1
}

// CHECK-LABEL: hw.module @firreg_reset_rewrite
hw.module @firreg_reset_rewrite(in %clk : !seq.clock, in %in : i1, out out : i1) {
  %false = hw.constant false
  // CHECK: %reg = seq.firreg %in clock %clk {comment = "reset reg"} : i1
  %reg = seq.firreg %in clock %clk reset sync %false, %in {comment = "reset reg"} : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @live_sv_wire
hw.module @live_sv_wire(in %a : i1, out out : i1) {
  // CHECK: %wire = sv.wire {comment = "live SV wire"} : !hw.inout<i1>
  %wire = sv.wire {comment = "live SV wire"} : !hw.inout<i1>
  sv.assign %wire, %a : i1
  %read = sv.read_inout %wire : !hw.inout<i1>
  hw.output %read : i1
}

// CHECK-LABEL: hw.module @dead_sv_wire
// CHECK-NOT: sv.wire
// CHECK-NOT: sv.assign
hw.module @dead_sv_wire(in %a : i1) {
  %wire = sv.wire {comment = "dead SV wire"} : !hw.inout<i1>
  sv.assign %wire, %a : i1
}

// CHECK-LABEL: hw.module @dead_sv_reg
// CHECK-NOT: sv.reg
// CHECK-NOT: sv.assign
hw.module @dead_sv_reg(in %a : i1) {
  %reg = sv.reg {comment = "dead SV reg"} : !hw.inout<i1>
  sv.assign %reg, %a : i1
}
