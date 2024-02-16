// RUN: circt-opt %s -export-split-verilog='dir-name=%t.dir'
// RUN: cat %t.dir%{fs-sep}top.sv | FileCheck %s

hw.module private @TargetA(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetB(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetDefault(in %a: i32, out b: i32) {
  hw.output %a : i32
}

// CHECK-LABEL: module top
hw.module public @top(in %a: i32, out b: i32, out d: i32) {
  // CHECK:      `ifndef __circt_choice_top_inst1
  // CHECK-NEXT: `define __circt_choice_top_inst1 TargetDefault
  // CHECK-NEXT: `endif
  // CHECK-NEXT: `__circt_choice_top_inst1 inst1 (
  // CHECK-NEXT:    .a (a),
  // CHECK-NEXT:    .b (b)
  // CHECK-NEXT: );
  %b = hw.instance_choice "inst1" sym @inst1 option "Perf" @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)

  // CHECK:      `ifndef __circt_choice_top_inst2
  // CHECK-NEXT: `define __circt_choice_top_inst2 TargetDefault
  // CHECK-NEXT: `endif
  // CHECK-NEXT: `__circt_choice_top_inst2 inst2 (
  // CHECK-NEXT:   .a (a),
  // CHECK-NEXT:   .b (_inst2_b)
  // CHECK-NEXT: );

  %c = hw.instance_choice "inst2" sym @inst2 option "Perf" @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)

  %d = comb.add %c, %a : i32

  hw.output %b, %d : i32, i32
}
