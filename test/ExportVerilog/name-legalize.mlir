// RUN: circt-opt %s -export-verilog -verify-diagnostics -o %t.mlir | FileCheck %s --strict-whitespace

// CHECK: module namechange(
// CHECK: input  [3:0] casex_0,
// CHECK: output [3:0] if_1); 
hw.module @namechange(%casex: i4) -> (if: i4) {
  // CHECK: assign if_1 = casex_0;
  hw.output %casex : i4
}

hw.module.extern @module_with_bool<bparam: i1>() -> ()

// CHECK-LABEL: module parametersNameConflict
// CHECK-NEXT:    #(parameter [41:0] p1_0 = 42'd17,
// CHECK-NEXT:      parameter [0:0]  wire_1) (
// CHECK-NEXT:    input [7:0] p1);
hw.module @parametersNameConflict<p1: i42 = 17, wire: i1>(%p1: i8) {
  %myWire = sv.wire : !hw.inout<i1>

  // CHECK: `ifdef SOMEMACRO
  sv.ifdef "SOMEMACRO" {
    // CHECK: localparam local_0 = wire_1;
    %local = sv.localparam : i1 { value = #hw.param.decl.ref<"wire">: i1 }

    // CHECK: assign myWire = wire_1;
    %0 = hw.param.value i1 = #hw.param.decl.ref<"wire">
    sv.assign %myWire, %0: i1
  }

  // "wire" param getting updated should update in this instance.
  
  // CHECK: module_with_bool #(
  // CHECK:  .bparam(wire_1)
  // CHECK: ) inst ();
  hw.instance "inst" @module_with_bool<bparam: i1 = #hw.param.decl.ref<"wire">>() -> ()

  // CHECK: module_with_bool #(
  // CHECK:  .bparam(wire ^ 1)
  // CHECK: ) inst2 ();
  hw.instance "inst2" @module_with_bool<bparam: i1 = #hw.param.expr.xor<#hw.param.verbatim<"wire">, true>>() -> ()
}

// CHECK-LABEL: module useParametersNameConflict(
hw.module @useParametersNameConflict(%xxx: i8) {
  // CHECK: parametersNameConflict #(
  // CHECK:  .p1_0(42'd27),
  // CHECK:  .wire_1(0)
  // CHECK: ) inst (
  // CHECK:  .p1 (xxx)
  // CHECK: );
  hw.instance "inst" @parametersNameConflict<p1: i42 = 27, wire: i1 = 0>(p1: %xxx: i8) -> ()

  // CHECK: `ifdef SOMEMACRO
  sv.ifdef "SOMEMACRO" {
    // CHECK: reg [3:0] xxx_0;
    %0 = sv.reg  { name = "xxx" } : !hw.inout<i4>
  }
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: module inout_0(
// CHECK:         input  inout_0,
// CHECK:         output output_1);
hw.module @inout(%inout: i1) -> (output: i1) {
// CHECK:       assign output_1 = inout_0;
  hw.output %inout : i1
}

// CHECK-LABEL: module inout_inst(	
hw.module @inout_inst(%a: i1) {
  // CHECK: inout_0 foo (
  // CHECK:   .inout_0  (a),
  // CHECK:   .output_1 (foo_output)
  // CHECK: );
  %0 = hw.instance "foo" @inout (inout: %a: i1) -> (output: i1)
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: module reg_1(
// CHECK-NEXT:    input  inout_0,
// CHECK-NEXT:    output output_1);
hw.module @reg(%inout: i1) -> (output: i1) {
  // CHECK: assign output_1 = inout_0;
  hw.output %inout : i1
}

// https://github.com/llvm/circt/issues/525
// CHECK-LABEL: module issue525(
// CHECK-NEXT:    input  [1:0] struct_0, else_1,
// CHECK-NEXT:    output [1:0] casex_2);
hw.module @issue525(%struct: i2, %else: i2) -> (casex: i2) {
  // CHECK: assign casex_2 = struct_0 + else_1;
  %2 = comb.add %struct, %else : i2
  hw.output %2 : i2
}

hw.module @B(%a: i1) -> () {
}

// CHECK-LABEL: module TestDupInstanceName(
hw.module @TestDupInstanceName(%a: i1) {
  // CHECK: B name (
  hw.instance "name" @B(a: %a: i1) -> ()
  // CHECK: B name_0 (
  hw.instance "name" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: module TestEmptyInstanceName(
hw.module @TestEmptyInstanceName(%a: i1) {
  // CHECK: B _T (
  hw.instance "" @B(a: %a: i1) -> ()
  // CHECK: B _T_0 (
  hw.instance "" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: module TestInstanceNameValueConflict(
hw.module @TestInstanceNameValueConflict(%a: i1) {
  // CHECK:  wire name;
  %name = sv.wire : !hw.inout<i1>
  // CHECK:  wire output_0;
  %output = sv.wire : !hw.inout<i1>
  // CHECK:  reg  input_1;
  %input = sv.reg : !hw.inout<i1>
  // CHECK: B name_2 (
  hw.instance "name" @B(a: %a: i1) -> ()
}

// https://github.com/llvm/circt/issues/855
// CHECK-LABEL: module nameless_reg(
hw.module @nameless_reg(%a: i1) -> () {
  // CHECK: reg [3:0] _T;
  %661 = sv.reg : !hw.inout<i4>
}

// CHECK-LABEL: module verif_renames(
hw.module @verif_renames(%cond: i1) {
  // CHECK: initial
  sv.initial {
    // CHECK:   assert_0: assert(cond);
    sv.assert "assert" %cond : i1
  }
}

// CHECK-LABEL: module verbatim_renames(
hw.module @verbatim_renames(%a: i1) {
  // CHECK: wire wire_0;

  // CHECK: // VERB Module : reg_1 inout_0
  sv.verbatim "// VERB Module : {{0}} {{1}}" {symbols = [@reg, @inout]}

  // Make sure symbol references to wires and instances get renamed correctly.
  %wire = sv.wire sym @wire1 : !hw.inout<i1>

  // CHECK: inout_0 module_1 (
  %0 = hw.instance "module" sym @struct @inout (inout: %a: i1) -> (output: i1)

  // CHECK: // VERB Instance : module_1 wire_0
  sv.verbatim "// VERB Instance : {{0}} {{1}}" {symbols = [@struct, @wire1]}

}
