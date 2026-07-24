// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --cse | FileCheck %s --check-prefix=CSE

// `llhd.resample` pins a live sample of a captured value to the block that
// takes it. Its read effect must keep CSE from merging markers (and the pure
// chains rooted at them) across sibling check blocks: merging would let a
// later resume block observe an earlier wake's sample.

// CHECK-LABEL: hw.module @Roundtrip
hw.module @Roundtrip(in %a : i8) {
  // CHECK: llhd.process
  llhd.process {
    // CHECK: llhd.resample %a : i8
    %0 = llhd.resample %a : i8
    llhd.halt
  }
  hw.output
}

// CSE-LABEL: hw.module @KeepAcrossBlocks
hw.module @KeepAcrossBlocks(in %clk : i1) {
  %true = hw.constant true
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2
  ^bb2:
    // CSE: ^bb2:
    // CSE: [[S1:%.+]] = llhd.resample %clk : i1
    // CSE: [[X1:%.+]] = comb.xor bin [[S1]], %true
    // CSE: cf.cond_br [[X1]],
    %0 = llhd.resample %clk : i1
    %1 = comb.xor bin %0, %true : i1
    cf.cond_br %1, ^bb3, ^bb1
  ^bb3:
    llhd.wait ^bb4
  ^bb4:
    // The marker in this block must NOT be merged with ^bb2's, and the xor
    // chain must stay rooted at the block-local marker.
    // CSE: ^bb4:
    // CSE: [[S2:%.+]] = llhd.resample %clk : i1
    // CSE: [[X2:%.+]] = comb.xor bin [[S2]], %true
    // CSE: cf.cond_br [[X2]],
    %2 = llhd.resample %clk : i1
    %3 = comb.xor bin %2, %true : i1
    cf.cond_br %3, ^bb5, ^bb3
  ^bb5:
    llhd.halt
  }
  hw.output
}
