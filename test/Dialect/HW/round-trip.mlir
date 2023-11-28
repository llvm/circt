// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module private @TargetA(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetB(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module private @TargetDefault(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module public @top(in %a: i32) {
  // CHECK: hw.instance_choice "inst1" sym @inst1 @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)
  hw.instance_choice "inst1" sym @inst1 @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)
  // CHECK: hw.instance_choice "inst2" @TargetDefault(a: %a: i32) -> (b: i32)
  hw.instance_choice "inst2" @TargetDefault(a: %a: i32) -> (b: i32)
}

