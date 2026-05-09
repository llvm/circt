// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Check subset of output to make sure posedge calculation works with different clock positions
// CHECK-LABEL:  func.func @test_bmc_clock_not_first() -> i1 {
// CHECK:        [[LOOP:%.+]] = func.call @bmc_loop([[ARG2:%.+]]) : (!smt.bv<1>) -> !smt.bv<1>
// CHECK:        [[OLDCLOCKLOW:%.+]] = smt.bv.not [[ARG2]]
// CHECK:        [[BVPOSEDGE:%.+]] = smt.bv.and [[OLDCLOCKLOW]], [[LOOP]]

func.func @test_bmc_clock_not_first() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [unit]
  init {
    %clk = seq.const_clock low
    verif.yield %clk : !seq.clock
  }
  loop {
    ^bb0(%clk: !seq.clock):
    %from_clock = seq.from_clock %clk
    %c-1_i1 = hw.constant -1 : i1
    %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
    %newclk = seq.to_clock %neg_clock
    verif.yield %newclk: !seq.clock
  }
  circuit {
  ^bb0(%arg0: i32, %clk: !seq.clock, %state0: i32):
    // dummy property
    %true = hw.constant true
    verif.assert %true : i1
    %c-1_i32 = hw.constant -1 : i32
    %0 = comb.add %arg0, %state0 : i32
    // %state0 is the result of a seq.compreg taking %0 as input
    %2 = comb.xor %state0, %c-1_i32 : i32
    verif.yield %2, %0 : i32, i32
  }
  func.return %bmc : i1
}
