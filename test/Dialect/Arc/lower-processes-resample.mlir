// RUN: circt-opt --arc-lower-processes %s | FileCheck %s

// `llhd.resample` markers protect the live "after" samples of consecutive
// edge-sensitive event waits from cross-check-block CSE (the polarity term of
// two negedge waits on the same captured value is otherwise identical and
// merges into the first check block, where it is persisted across the
// suspension). Once this pass has threaded the coroutine arguments through
// the CFG -- binding each resume block's uses to that block's re-entry
// values -- the markers have done their job and must fold to their operand,
// leaving each check block's trigger chain rooted at that block's own
// argument.

// CHECK-LABEL: arc.coroutine.define @TwoNegedgeWaits.llhd.process
// CHECK-SAME: ([[CLK_ENTRY:%.+]]: i1, [[NOW:%.+]]: i64)
// CHECK-NOT: llhd.resample
hw.module @TwoNegedgeWaits(in %clk: i1) {
  %true = hw.constant true
  llhd.process {
    cf.br ^bb1(%clk : i1)
  ^bb1(%before1: i1):
    llhd.wait (%clk : i1), ^bb2(%clk : i1)
  // First check block: the xor must use this block's own clk binding.
  // CHECK: ^[[CHECK1:bb[0-9]+]]([[CLK1:%.+]]: i1, {{%.+}}: i64, [[BEFORE1:%.+]]: i1):
  // CHECK: [[X1:%.+]] = comb.xor bin [[CLK1]], %true
  // CHECK: [[T1:%.+]] = comb.and bin [[BEFORE1]], [[X1]]
  // CHECK: cf.cond_br [[T1]],
  ^bb2(%b1: i1):
    %0 = llhd.resample %clk : i1
    %1 = comb.xor bin %0, %true : i1
    %2 = comb.and bin %b1, %1 : i1
    cf.cond_br %2, ^bb3, ^bb1(%clk : i1)
  ^bb3:
    llhd.wait (%clk : i1), ^bb4(%clk : i1)
  // Second check block: after folding, the xor must be rebuilt from THIS
  // block's clk argument, not reuse the first block's term.
  // CHECK: ^[[CHECK2:bb[0-9]+]]([[CLK2:%.+]]: i1, {{%.+}}: i64, [[BEFORE2:%.+]]: i1):
  // CHECK: [[X2:%.+]] = comb.xor bin [[CLK2]], %true
  // CHECK: [[T2:%.+]] = comb.and bin [[BEFORE2]], [[X2]]
  // CHECK: cf.cond_br [[T2]],
  ^bb4(%b2: i1):
    %3 = llhd.resample %clk : i1
    %4 = comb.xor bin %3, %true : i1
    %5 = comb.and bin %b2, %4 : i1
    cf.cond_br %5, ^bb5, ^bb3
  ^bb5:
    llhd.halt
  }
  hw.output
}
