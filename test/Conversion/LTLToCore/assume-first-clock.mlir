// RUN: circt-opt %s --lower-ltl-to-core='assume-first-clock' | FileCheck %s

// CHECK: hw.module @clock_arg(in [[A:%.+]] : i32, in [[CLK:%.+]] : !seq.clock)
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[SHIFTREG:%.+]] = seq.shiftreg[2] [[A]], [[CLK]], [[TRUE]] : i32

hw.module @clock_arg(in %a: i32, in %clk: !seq.clock) {
  ltl.past %a, 2 : i32
}

// CHECK: hw.module @to_clock(in [[A:%.+]] : i32, in [[CLK:%.+]] : i1)
// CHECK: [[TO_CLK:%.+]] = seq.to_clock [[CLK]]
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[SHIFTREG:%.+]] = seq.shiftreg[2] [[A]], [[TO_CLK]], [[TRUE]] : i32

hw.module @to_clock(in %a: i32, in %clk: i1) {
  %0 = seq.to_clock %clk
  ltl.past %a, 2 : i32
}

// Ensure !seq.clock arguments are considered over to_clocks
// CHECK: hw.module @both(in [[A:%.+]] : i32, in [[B:%.+]] : i1, in [[CLK:%.+]] : !seq.clock)
// CHECK: [[TO_CLK:%.+]] = seq.to_clock [[B]]
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[SHIFTREG:%.+]] = seq.shiftreg[2] [[A]], [[CLK]], [[TRUE]] : i32

hw.module @both(in %a: i32, in %b: i1, in %clk: !seq.clock) {
  %0 = seq.to_clock %b
  ltl.past %a, 2 : i32
}
