// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --export-verilog | FileCheck %s --check-prefix=VERILOG

// CHECK-LABEL: hw.module @ModuleA
hw.module @ModuleA(in %a: i1, in %b: i4, out c: i5) {
  %0 = comb.concat %a, %b : i1, i4
  hw.output %0 : i5
}

// CHECK-LABEL: hw.module @ModuleB
hw.module @ModuleB(in %a: i1, in %b: i4, out c: i5) {
  %0 = comb.concat %a, %b : i1, i4
  hw.output %0 : i5
}

// CHECK-LABEL: sv.macro.decl @WHICH_MODULE
sv.macro.decl @WHICH_MODULE

// CHECK-LABEL: hw.module @Top
hw.module @Top(in %a: i1, in %b: i4, out c: i5) {
  // CHECK: %inst.c = sv.macro_instance "inst" @WHICH_MODULE [@ModuleA, @ModuleB](a: %a: i1, b: %b: i4) -> (c: i5)
  %0 = sv.macro_instance "inst" @WHICH_MODULE [@ModuleA, @ModuleB] (a: %a: i1, b: %b: i4) -> (c: i5)
  hw.output %0 : i5
}

// CHECK-LABEL: hw.module @TopWithSym
hw.module @TopWithSym(in %a: i1, in %b: i4, out c: i5) {
  // CHECK: %inst2.c = sv.macro_instance "inst2" sym @inst_sym @WHICH_MODULE [@ModuleA, @ModuleB](a: %a: i1, b: %b: i4) -> (c: i5)
  %0 = sv.macro_instance "inst2" sym @inst_sym @WHICH_MODULE [@ModuleA, @ModuleB] (a: %a: i1, b: %b: i4) -> (c: i5)
  hw.output %0 : i5
}

// VERILOG-LABEL: module ModuleA
// VERILOG-LABEL: module ModuleB
// VERILOG-LABEL: module Top
// VERILOG: `WHICH_MODULE inst (
// VERILOG-LABEL: module TopWithSym
// VERILOG: `WHICH_MODULE inst2 (

