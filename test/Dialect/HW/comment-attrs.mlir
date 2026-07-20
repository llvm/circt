// RUN: circt-opt %s --verify-roundtrip | FileCheck %s

hw.module.extern @Child(in %x : i1, out y : i1)

// CHECK-LABEL: hw.module @CommentAttrs
hw.module @CommentAttrs(in %a : i1, in %clk : !seq.clock, out out : i1) {
  // CHECK: %hw_wire = hw.wire %a {comment = "HW wire"} : i1
  %hw_wire = hw.wire %a {comment = "HW wire"} : i1

  // CHECK: %reg = seq.firreg %hw_wire clock %clk {comment = "FIR register"} : i1
  %reg = seq.firreg %hw_wire clock %clk {comment = "FIR register"} : i1

  // CHECK: %sv_wire = sv.wire {comment = "SV wire"} : !hw.inout<i1>
  %sv_wire = sv.wire {comment = "SV wire"} : !hw.inout<i1>
  sv.assign %sv_wire, %reg : i1

  // CHECK: %sv_reg = sv.reg {comment = "SV reg"} : !hw.inout<i1>
  %sv_reg = sv.reg {comment = "SV reg"} : !hw.inout<i1>
  sv.assign %sv_reg, %reg : i1

  // CHECK: %sv_logic = sv.logic {comment = "SV logic"} : !hw.inout<i1>
  %sv_logic = sv.logic {comment = "SV logic"} : !hw.inout<i1>
  sv.assign %sv_logic, %reg : i1

  // CHECK: %{{.*}} = hw.instance "child" @Child(x: %hw_wire: i1) -> (y: i1) {comment = "HW instance"}
  %child = hw.instance "child" @Child(x: %hw_wire : i1) -> (y : i1) {comment = "HW instance"}

  hw.output %child : i1
}
