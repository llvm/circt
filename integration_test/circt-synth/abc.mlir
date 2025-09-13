// REQUIRES: yosys
// RUN: circt-synth %s -abc-path %yosys-abc -abc-commands "balance" -top balanceTest | FileCheck %s
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-abc-runner{abc-path=%yosys-abc abc-commands="balance"}))' | FileCheck %s

// Test that aig commands work
hw.module @balanceTest(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out: i1) {
  // Create an unbalanced chain: ((a & b) & c) & d
  // Balance should restructure this to reduce depth
  // CHECK-LABEL: hw.module @balanceTest(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out out : i1) {
  // CHECK-NEXT:    %[[AND_INV_0:.+]] = synth.aig.and_inv %b, %a : i1
  // CHECK-NEXT:    %[[AND_INV_1:.+]] = synth.aig.and_inv %d, %c : i1
  // CHECK-NEXT:    %[[AND_INV_2:.+]] = synth.aig.and_inv %[[AND_INV_1]], %[[AND_INV_0]] : i1
  // CHECK-NEXT:    hw.output %[[AND_INV_2]] : i1
  // CHECK-NEXT:  }
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  %2 = synth.aig.and_inv %1, %d : i1
  hw.output %2 : i1
}
