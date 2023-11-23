// RUN: circt-opt -firrtl-lower-layers -split-input-file %s | FileCheck %s

firrtl.circuit "Simple" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind {}
    }
  }
  firrtl.module @Simple() {
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<2>
    firrtl.layerblock @A {
      %aa = firrtl.node %a: !firrtl.uint<1>
      %c = firrtl.wire : !firrtl.uint<3>
      firrtl.layerblock @A::@B {
        %bb = firrtl.node %b: !firrtl.uint<2>
        %cc = firrtl.node %c: !firrtl.uint<3>
        firrtl.layerblock @A::@B::@C {
          %ccc = firrtl.node %cc: !firrtl.uint<3>
        }
      }
    }
  }
}

// CHECK-LABEL: firrtl.circuit "Simple"
//
// CHECK:      sv.verbatim "`include \22groups_Simple_A.sv\22\0A
// CHECK-SAME:   `include \22groups_Simple_A_B.sv\22\0A
// CHECK-SAME:   `ifndef groups_Simple_A_B_C\0A
// CHECK-SAME:   define groups_Simple_A_B_C"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A_B_C.sv", excludeFromFileList>
// CHECK:      sv.verbatim "`include \22groups_Simple_A.sv\22\0A
// CHECK-SAME:   `ifndef groups_Simple_A_B\0A
// CHECK-SAME:   define groups_Simple_A_B"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A_B.sv", excludeFromFileList>
// CHECK:      sv.verbatim "`ifndef groups_Simple_A\0A
// CHECK-SAME:   define groups_Simple_A"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A.sv", excludeFromFileList>
//
// CHECK:      firrtl.module private @Simple_A_B_C(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[cc_port:[_a-zA-Z0-9]+]]: !firrtl.uint<3>
// CHECK-NEXT:   %ccc = firrtl.node %[[cc_port]]
// CHECK-NEXT: }
//
// CHECK:      firrtl.module private @Simple_A_B(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[b_port:[_a-zA-Z0-9]+]]: !firrtl.uint<2>
// CHECK-SAME:   in %[[c_port:[_a-zA-Z0-9]+]]: !firrtl.uint<3>
// CHECK-SAME:   out %[[cc_port:[_a-zA-Z0-9]+]]: !firrtl.probe<uint<3>>
// CHECK-NEXT:   %bb = firrtl.node %[[b_port]]
// CHECK-NEXT:   %cc = firrtl.node %[[c_port]]
// CHECK-NEXT:   %0 = firrtl.ref.send %cc
// CHECK-NEXT:   firrtl.ref.define %[[cc_port]], %0
// CHECK-NEXT: }
//
// CHECK:      firrtl.module private @Simple_A(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[a_port:[_a-zA-Z0-9]+]]: !firrtl.uint<1>
// CHECK-SAME:   out %[[c_port:[_a-zA-Z0-9]+]]: !firrtl.probe<uint<3>>
// CHECK-NEXT:   %aa = firrtl.node %[[a_port]]
// CHECK:        %[[c_ref:[_a-zA-Z0-9]+]] = firrtl.ref.send %c
// CHECK-NEXT:   firrtl.ref.define %[[c_port]], %[[c_ref]]
// CHECK-NEXT: }
//
// CHECK:      firrtl.module @Simple() {
// CHECK-NOT:  firrtl.module
// CHECK-NOT:    firrtl.layerblock
// CHECK:        %[[A_B_C_cc:[_a-zA-Z0-9]+]] = firrtl.instance simple_A_B_C {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"groups_Simple_A_B_C.sv"
// CHECK-SAME:     excludeFromFileList
// CHECK-SAME:     @Simple_A_B_C(
// CHECK-NEXT:   %[[A_B_b:[_a-zA-Z0-9]+]], %[[A_B_c:[_a-zA-Z0-9]+]], %[[A_B_cc:[_a-zA-Z0-9]+]] = firrtl.instance simple_A_B {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"groups_Simple_A_B.sv", excludeFromFileList>
// CHECK-SAME:     @Simple_A_B(
// CHECK-NEXT:   %[[A_B_cc_resolve:[_a-zA-Z0-9]+]] = firrtl.ref.resolve %[[A_B_cc]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_B_C_cc]], %[[A_B_cc_resolve]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_B_b]], %b
// CHECK-NEXT:   %[[A_a:[_a-zA-Z0-9]+]], %[[A_c:[_a-zA-Z0-9]+]] = firrtl.instance simple_A {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"groups_Simple_A.sv", excludeFromFileList>
// CHECK-SAME:     @Simple_A(
// CHECK-NEXT:   %[[A_c_resolve:[_a-zA-Z0-9]+]] = firrtl.ref.resolve %[[A_c]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_B_c]], %[[A_c_resolve]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_a]], %a
// CHECK:      }
//
// CHECK-DAG:  sv.verbatim "`endif // groups_Simple_A"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A.sv", excludeFromFileList>
// CHECK-DAG:  sv.verbatim "`endif // groups_Simple_A_B"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A_B.sv", excludeFromFileList>

// -----

firrtl.circuit "ModuleNameConflict" {
  firrtl.layer @A bind {}
  firrtl.module private @ModuleNameConflict_A() {}
  firrtl.module @ModuleNameConflict() {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.instance foo @ModuleNameConflict_A()
    firrtl.layerblock @A {
      %b = firrtl.node %a : !firrtl.uint<1>
    }
  }
}

// CHECK-LABEL: firrtl.circuit "ModuleNameConflict"
//
// CHECK:       firrtl.module private @[[groupModule:[_a-zA-Z0-9]+]](in
//
// CHECK:       firrtl.module @ModuleNameConflict()
// CHECK-NOT:   firrtl.module
// CHECK:         firrtl.instance foo @ModuleNameConflict_A()
// CHECK-NEXT:    firrtl.instance {{[_a-zA-Z0-9]+}} {lowerToBind,
// CHECK-SAME:      @[[groupModule]](
