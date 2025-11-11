// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-inliner{inline-single-use=false inline-small=false inline-empty=false inline-no-outputs=false})' | FileCheck %s --check-prefix=NONE
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-inliner{inline-single-use=true inline-small=false inline-empty=false inline-no-outputs=false})' | FileCheck %s --check-prefix=SINGLE
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-inliner{inline-single-use=false inline-small=true inline-empty=false inline-no-outputs=false})' | FileCheck %s --check-prefix=SMALL
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-inliner{small-threshold=3 inline-single-use=false})' | FileCheck %s --check-prefix=THRESHOLD
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-inliner{inline-with-state=true})' | FileCheck %s --check-prefix=STATE

// Test that all inlining heuristics can be controlled via command line options

// NONE-LABEL: hw.module @TestSingleUse
// SINGLE-LABEL: hw.module @TestSingleUse
hw.module @TestSingleUse(in %x: i4, out y: i4) {
  // NONE-NEXT: hw.instance "small" @SmallModule
  // SINGLE-NEXT: %[[V0:.+]] = comb.add %x, %x
  // SINGLE-NEXT: hw.output %[[V0]]
  %0 = hw.instance "small" @SmallModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}

hw.module private @SmallModule(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  hw.output %0 : i4
}

// SMALL-LABEL: hw.module @TestSmall
hw.module @TestSmall(in %x: i4, out y: i4) {
  // SMALL-NEXT: %[[V0:.+]] = comb.add %x, %x
  // SMALL-NEXT: hw.output %[[V0]]
  %0 = hw.instance "tiny" @TinyModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}

hw.module private @TinyModule(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  hw.output %0 : i4
}

// THRESHOLD-LABEL: hw.module @TestThreshold
hw.module @TestThreshold(in %x: i4, out y: i4) {
  // THRESHOLD-NEXT: hw.instance "medium" @MediumModule
  %0 = hw.instance "medium" @MediumModule(a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}

// This module has 5 operations (4 comb ops + 1 hw.output)
// With threshold=3, it should NOT be inlined
hw.module private @MediumModule(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = comb.mul %0, %a : i4
  %2 = comb.xor %1, %a : i4
  %3 = comb.or %2, %a : i4
  hw.output %3 : i4
}

// STATE-LABEL: hw.module @TestState
hw.module @TestState(in %clk: !seq.clock, in %x: i4, out y: i4) {
  // STATE-NEXT: %[[REG:.+]] = seq.firreg %x clock %clk
  // STATE-NEXT: hw.output %[[REG]]
  %0 = hw.instance "reg" @RegModule(clk: %clk: !seq.clock, a: %x: i4) -> (b: i4)
  hw.output %0 : i4
}

hw.module private @RegModule(in %clk: !seq.clock, in %a: i4, out b: i4) {
  %0 = seq.firreg %a clock %clk : i4
  hw.output %0 : i4
}

