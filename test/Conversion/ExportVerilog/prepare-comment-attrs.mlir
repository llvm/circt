// RUN: circt-opt %s --pass-pipeline='builtin.module(any(prepare-for-emission))' | FileCheck %s

hw.module @CommentAttrs(in %a : i1, out out : i1) {
  // CHECK: %module_wire = sv.wire {comment = "module wire"} : !hw.inout<i1>
  %module_wire = hw.wire %a {comment = "module wire"} : i1

  sv.always {
    // CHECK: %procedural_wire = sv.logic {comment = "procedural wire"} : !hw.inout<i1>
    %procedural_wire = hw.wire %a {comment = "procedural wire"} : i1
    sv.assert %procedural_wire, immediate
  }

  hw.output %module_wire : i1
}
