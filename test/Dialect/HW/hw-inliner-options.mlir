// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-flatten-modules{hw-inline-all=false hw-inline-single-use=false hw-inline-small=false hw-inline-empty=false hw-inline-no-outputs=false})' | FileCheck %s --check-prefix=NONE
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-flatten-modules{hw-inline-all=false hw-inline-single-use=true hw-inline-small=false hw-inline-empty=false hw-inline-no-outputs=false})' | FileCheck %s --check-prefix=SINGLE
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-flatten-modules{hw-inline-all=false hw-inline-single-use=false hw-inline-small=true hw-inline-empty=false hw-inline-no-outputs=false})' | FileCheck %s --check-prefix=SMALL
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-flatten-modules{hw-inline-all=false hw-small-threshold=3 hw-inline-single-use=false})' | FileCheck %s --check-prefix=THRESHOLD
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw-flatten-modules{hw-inline-with-state=true})' | FileCheck %s --check-prefix=STATE

// Test that all inlining heuristics can be controlled via command line options

// NONE-LABEL: hw.module @TestSingleUse
// SINGLE-LABEL: hw.module @TestSingleUse
hw.module @TestSingleUse(in %x: i4, out y: i4) {
  // NONE-NEXT: hw.instance "large" @LargeModule
  // SINGLE-NEXT: %[[V0:.+]] = comb.add %x, %x
  // SINGLE-NEXT: %[[V1:.+]] = comb.and %[[V0]], %[[V0]]
  // SINGLE-NEXT: %[[V2:.+]] = comb.or %[[V1]], %[[V1]]
  // SINGLE-NEXT: %[[V3:.+]] = comb.xor %[[V2]], %[[V2]]
  // SINGLE-NEXT: %[[V4:.+]] = comb.mul %[[V3]], %[[V3]]
  // SINGLE-NEXT: %[[V5:.+]] = comb.add %[[V4]], %[[V4]]
  // SINGLE-NEXT: %[[V6:.+]] = comb.sub %[[V5]], %[[V5]]
  // SINGLE-NEXT: %[[V7:.+]] = comb.add %x, %x
  // SINGLE-NEXT: hw.output %[[V7]]
  %0 = hw.instance "large" @LargeModule(a: %x: i4) -> (b: i4)
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
  %0 = hw.instance "small" @SmallModule(a: %x: i4) -> (b: i4)
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

// This module has 9 operations (8 comb ops + 1 hw.output), exceeding the default threshold of 8
// It should NOT be inlined based on size alone, but WILL be inlined if single-use is enabled
hw.module private @LargeModule(in %a: i4, out b: i4) {
  %0 = comb.add %a, %a : i4
  %1 = comb.and %0, %0 : i4
  %2 = comb.or %1, %1 : i4
  %3 = comb.xor %2, %2 : i4
  %4 = comb.mul %3, %3 : i4
  %5 = comb.add %4, %4 : i4
  %6 = comb.sub %5, %5 : i4
  %7 = comb.add %a, %a : i4
  hw.output %7 : i4
}

