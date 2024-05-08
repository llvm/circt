// RUN: circt-opt %s -hw-lower-instance-choices --split-input-file -verify-diagnostics | FileCheck %s

// CHECK: sv.macro.decl @__circt_choice_top_inst1
// CHECK: sv.macro.decl @__circt_choice_top_inst2

hw.module private @TargetA(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetB(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetDefault(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module public @top(in %a: i32, out b: i32, out d: i32) {
  // CHECK:               sv.ifdef @__circt_choice_top_inst1 {
  // CHECK-NEXT:          } else {
  // CHECK-NEXT{LITERAL}:   sv.macro.def @__circt_choice_top_inst1 "{{0}}"([@TargetDefault])
  // CHECK-NEXT:          }
  // CHECK-NEXT:          hw.instance_choice "inst1"
  // CHECK-SAME:          {hw.choiceTarget = @__circt_choice_top_inst1}
  %b = hw.instance_choice "inst1" sym @inst1 option "Perf" @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)

  // CHECK:               sv.ifdef @__circt_choice_top_inst2 {
  // CHECK-NEXT:          } else {
  // CHECK-NEXT{LITERAL}:   sv.macro.def @__circt_choice_top_inst2 "{{0}}"([@TargetB])
  // CHECK-NEXT:          }
  // CHECK-NEXT:          hw.instance_choice "inst2"
  // CHECK-SAME:          {hw.choiceTarget = @__circt_choice_top_inst2}
  %c = hw.instance_choice "inst2" sym @inst2 option "Perf" @TargetB or @TargetA if "A" or @TargetDefault if "B"(a: %a: i32) -> (b: i32)

  %d = comb.add %c, %a : i32

  hw.output %b, %d : i32, i32
}
