// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-print-longest-path-analysis, hw.module(comb-balance-mux{mux-chain-threshold=4}))' -o %t.mlir | FileCheck %s --check-prefix=DEPTH_BEFORE
// RUN: circt-opt %t.mlir --pass-pipeline='builtin.module(synth-print-longest-path-analysis)' | FileCheck %s --check-prefix=DEPTH_AFTER
// RUN: circt-lec %t.mlir %s -c1=priority_mux_18_depth -c2=priority_mux_18_depth --shared-libs=%libz3 | FileCheck %s --check-prefix=MUX18_LEC
// Check that balancing muxes reduces the longest path in a priority mux from O(n) to O(log n).
// DEPTH_BEFORE-LABEL: priority_mux_18_depth
// DEPTH_BEFORE: Maximum path delay: 17
// DEPTH_AFTER-LABEL: priority_mux_18_depth
// DEPTH_AFTER: Maximum path delay: 6
// MUX18_LEC: c1 == c2
hw.module @priority_mux_18_depth(in %cond0: i1, in %cond1: i1, in %cond2: i1, in %cond3: i1, in %cond4: i1, in %cond5: i1, in %cond6: i1, in %cond7: i1, in %cond8: i1, in %cond9: i1, in %cond10: i1, in %cond11: i1, in %cond12: i1, in %cond13: i1, in %cond14: i1, in %cond15: i1, in %cond16: i1, in %cond17: i1, out true_output: i5, out false_side: i5) {
  %c0_i5 = hw.constant 0 : i5
  %c1_i5 = hw.constant 1 : i5
  %c2_i5 = hw.constant 2 : i5
  %c3_i5 = hw.constant 3 : i5
  %c4_i5 = hw.constant 4 : i5
  %c5_i5 = hw.constant 5 : i5
  %c6_i5 = hw.constant 6 : i5
  %c7_i5 = hw.constant 7 : i5
  %c8_i5 = hw.constant 8 : i5
  %c9_i5 = hw.constant 9 : i5
  %c10_i5 = hw.constant 10 : i5
  %c11_i5 = hw.constant 11 : i5
  %c12_i5 = hw.constant 12 : i5
  %c13_i5 = hw.constant 13 : i5
  %c14_i5 = hw.constant 14 : i5
  %c15_i5 = hw.constant 15 : i5
  %c16_i5 = hw.constant 16 : i5
  %c17_i5 = hw.constant 17 : i5

  %mux17_t = comb.mux %cond16, %c17_i5, %c0_i5 : i5
  %mux16_t = comb.mux %cond15, %mux17_t, %c16_i5 : i5
  %mux15_t = comb.mux %cond14, %mux16_t, %c15_i5 : i5
  %mux14_t = comb.mux %cond13, %mux15_t, %c14_i5 : i5
  %mux13_t = comb.mux %cond12, %mux14_t, %c13_i5 : i5
  %mux12_t = comb.mux %cond11, %mux13_t, %c12_i5 : i5
  %mux11_t = comb.mux %cond10, %mux12_t, %c11_i5 : i5
  %mux10_t = comb.mux %cond9, %mux11_t, %c10_i5 : i5
  %mux9_t = comb.mux %cond8, %mux10_t, %c9_i5 : i5
  %mux8_t = comb.mux %cond7, %mux9_t, %c8_i5 : i5
  %mux7_t = comb.mux %cond6, %mux8_t, %c7_i5 : i5
  %mux6_t = comb.mux %cond5, %mux7_t, %c6_i5 : i5
  %mux5_t = comb.mux %cond4, %mux6_t, %c5_i5 : i5
  %mux4_t = comb.mux %cond3, %mux5_t, %c4_i5 : i5
  %mux3_t = comb.mux %cond2, %mux4_t, %c3_i5 : i5
  %mux2_t = comb.mux %cond1, %mux3_t, %c2_i5 : i5
  %mux1_t = comb.mux %cond0, %mux2_t, %c1_i5 : i5

  %mux17_f = comb.mux %cond16, %c0_i5, %c17_i5 : i5
  %mux16_f = comb.mux %cond15, %c16_i5, %mux17_f : i5
  %mux15_f = comb.mux %cond14, %c15_i5, %mux16_f : i5
  %mux14_f = comb.mux %cond13, %c14_i5, %mux15_f : i5
  %mux13_f = comb.mux %cond12, %c13_i5, %mux14_f : i5
  %mux12_f = comb.mux %cond11, %c12_i5, %mux13_f : i5
  %mux11_f = comb.mux %cond10, %c11_i5, %mux12_f : i5
  %mux10_f = comb.mux %cond9, %c10_i5, %mux11_f : i5
  %mux9_f = comb.mux %cond8, %c9_i5, %mux10_f : i5
  %mux8_f = comb.mux %cond7, %c8_i5, %mux9_f : i5
  %mux7_f = comb.mux %cond6, %c7_i5, %mux8_f : i5
  %mux6_f = comb.mux %cond5, %c6_i5, %mux7_f : i5
  %mux5_f = comb.mux %cond4, %c5_i5, %mux6_f : i5
  %mux4_f = comb.mux %cond3, %c4_i5, %mux5_f : i5
  %mux3_f = comb.mux %cond2, %c3_i5, %mux4_f : i5
  %mux2_f = comb.mux %cond1, %c2_i5, %mux3_f : i5
  %mux1_f = comb.mux %cond0, %c1_i5, %mux2_f : i5

  hw.output %mux1_t, %mux1_f : i5, i5
}

// RUN: circt-lec %t.mlir %s -c1=index_to_balanced_mux -c2=index_to_balanced_mux --shared-libs=%libz3 | FileCheck %s --check-prefix=INDEX_TO_BALANCED_MUX_LEC
// DEPTH_BEFORE-LABEL: Longest Path Analysis result for "index_to_balanced_mux"
// DEPTH_BEFORE: Maximum path delay: 11
// DEPTH_AFTER-LABEL: Longest Path Analysis result for "index_to_balanced_mux"
// DEPTH_AFTER: Maximum path delay: 3
// INDEX_TO_BALANCED_MUX_LEC: c1 == c2
hw.module @index_to_balanced_mux(in %index: i3, out result: i8) {
  // Values to select from based on index
  %a = hw.constant 10 : i8
  %b = hw.constant 20 : i8
  %c = hw.constant 30 : i8
  %d = hw.constant 40 : i8
  %e = hw.constant 50 : i8
  %f = hw.constant 60 : i8
  %g = hw.constant 70 : i8
  %default = hw.constant 0 : i8

  // Index comparison constants
  %c0 = hw.constant 0 : i3
  %c1 = hw.constant 1 : i3
  %c2 = hw.constant 2 : i3
  %c3 = hw.constant 3 : i3
  %c4 = hw.constant 4 : i3
  %c5 = hw.constant 5 : i3
  %c6 = hw.constant 6 : i3

  %eq0 = comb.icmp eq %index, %c0 : i3
  %eq1 = comb.icmp eq %index, %c1 : i3
  %eq2 = comb.icmp eq %index, %c2 : i3
  %eq3 = comb.icmp eq %index, %c3 : i3
  %eq4 = comb.icmp eq %index, %c4 : i3
  %eq5 = comb.icmp eq %index, %c5 : i3
  %eq6 = comb.icmp eq %index, %c6 : i3

  %mux6 = comb.mux %eq6, %g, %default : i8
  %mux5 = comb.mux %eq5, %f, %mux6 : i8
  %mux4 = comb.mux %eq4, %e, %mux5 : i8
  %mux3 = comb.mux %eq3, %d, %mux4 : i8
  %mux2 = comb.mux %eq2, %c, %mux3 : i8
  %mux1 = comb.mux %eq1, %b, %mux2 : i8
  %result = comb.mux %eq0, %a, %mux1 : i8

  hw.output %result : i8
}

