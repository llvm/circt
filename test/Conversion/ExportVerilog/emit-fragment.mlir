// RUN: circt-opt -o=- --export-verilog %s | FileCheck %s
// RUN: circt-opt -o=- --export-split-verilog='dir-name=%t' %s
// RUN: cat %t%{fs-sep}FirstModule.sv | FileCheck %s --check-prefix=FIRST
// RUN: cat %t%{fs-sep}SecondModule.sv | FileCheck %s --check-prefix=SECOND

// CHECK:      `ifndef MacroA
// CHECK-NEXT:   `define MacroA A
// CHECK-NEXT: `endif
// CHECK-NEXT: `ifndef MacroB
// CHECK-NEXT:   `define MacroB B
// CHECK-NEXT: `endif
// CHECK-LABEL: FirstModule
// CHECK-NOT: MacroB
// CHECK:      `ifndef MacroC
// CHECK-NEXT:   `define MacroC C
// CHECK-NEXT: `endif
// CHECK-LABEL: SecondModule

// FIRST:      `ifndef MacroA
// FIRST-NEXT:   `define MacroA A
// FIRST-NEXT: `endif
// FIRST-NEXT: `ifndef MacroB
// FIRST-NEXT:   `define MacroB B
// FIRST-NEXT: `endif
// FIRST-NEXT: module FirstModule

// SECOND:      `ifndef MacroB
// SECOND-NEXT:   `define MacroB B
// SECOND-NEXT: `endif
// SECOND-NEXT: `ifndef MacroC
// SECOND-NEXT:   `define MacroC C
// SECOND-NEXT: `endif
// SECOND-NEXT: module SecondModule

sv.macro.decl @MacroA
sv.macro.decl @MacroB
sv.macro.decl @MacroC

emit.fragment @FragmentA {
  sv.ifdef @MacroA {
  } else {
    sv.macro.def @MacroA "A"
  }
}

emit.fragment @FragmentB {
  sv.ifdef @MacroB {
  } else {
    sv.macro.def @MacroB "B"
  }
}

emit.fragment @FragmentC {
  sv.ifdef @MacroC {
  } else {
    sv.macro.def @MacroC "C"
  }
}

hw.module @FirstModule(in %in : i32, out out : i32) attributes { "emit.fragments" = [@FragmentA, @FragmentB] } {
  hw.output %in : i32
}

hw.module @SecondModule(in %in : i32, out out : i32) attributes { "emit.fragments" = [@FragmentB, @FragmentC] } {
  hw.output %in : i32
}
