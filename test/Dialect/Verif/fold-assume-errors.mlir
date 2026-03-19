// RUN: circt-opt %s --pass-pipeline='builtin.module(any(fold-assume))'  --split-input-file --verify-diagnostics 

hw.module @ManyAssumes(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin uge %a, %c0_i42 : i42
  %2 = comb.icmp bin uge %b, %c0_i42 : i42
  verif.assume %1 : i1
  // expected-error @below {{Multiple verif.assume found in the current block! Run `--combine-assert-like` before running --fold-assume.}}
  verif.assume %2 : i1
  hw.output %0 : i42
}

// -----

hw.module @ManyAsserts(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add bin %a, %b : i42
  %c0_i42 = hw.constant 0 : i42
  %1 = comb.icmp bin uge %a, %c0_i42 : i42
  %2 = comb.icmp bin uge %b, %c0_i42 : i42
  verif.assert %1 : i1
  // expected-error @below {{Multiple verif.assert found in the current block! Run `--combine-assert-like` before running --fold-assume.}}
  verif.assert %2 : i1
  hw.output %0 : i42
}
