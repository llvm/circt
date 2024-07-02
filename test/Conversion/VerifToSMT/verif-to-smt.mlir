// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @test_lec
// CHECK-SAME:  ([[ARG0:%.+]]: !smt.bv<1>)
func.func @test_lec(%arg0: !smt.bv<1>) -> (i1, i1, i1) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
  // CHECK: [[C0:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
  // CHECK: [[V0:%.+]] = smt.eq %arg0, [[C0]] : !smt.bv<1>
  // CHECK: [[V1:%.+]] = smt.not [[V0]]
  // CHECK: smt.assert [[V1]]
  verif.assert %0 : i1

  // CHECK: [[EQ:%.+]] = smt.solver() : () -> i1
  // CHECK: [[IN0:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK: [[V0:%.+]] = builtin.unrealized_conversion_cast [[IN0]] : !smt.bv<32> to i32
  // CHECK: [[IN1:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK: [[V1:%.+]] = builtin.unrealized_conversion_cast [[IN1]] : !smt.bv<32> to i32
  // CHECK: [[V2:%.+]]:2 = "some_op"([[V0]], [[V1]]) : (i32, i32) -> (i32, i32)
  // CHECK: [[V3:%.+]] = builtin.unrealized_conversion_cast [[V2]]#0 : i32 to !smt.bv<32>
  // CHECK: [[V4:%.+]] = smt.distinct [[IN0]], [[V3]] : !smt.bv<32>
  // CHECK: [[V5:%.+]] = builtin.unrealized_conversion_cast [[V2]]#1 : i32 to !smt.bv<32>
  // CHECK: [[V6:%.+]] = smt.distinct [[IN1]], [[V5]] : !smt.bv<32>
  // CHECK: [[V7:%.+]] = smt.or [[V4]], [[V6]]
  // CHECK: smt.assert [[V7]]
  // CHECK: [[FALSE:%.+]] = arith.constant false
  // CHECK: [[TRUE:%.+]] = arith.constant true
  // CHECK: [[V8:%.+]] = smt.check
  // CHECK: smt.yield [[FALSE]]
  // CHECK: smt.yield [[FALSE]]
  // CHECK: smt.yield [[TRUE]]
  // CHECK: smt.yield [[V8]] :
  %1 = verif.lec first {
  ^bb0(%arg1: i32, %arg2: i32):
    verif.yield %arg1, %arg2 : i32, i32
  } second {
  ^bb0(%arg1: i32, %arg2: i32):
    %2, %3 = "some_op"(%arg1, %arg2) : (i32, i32) -> (i32, i32)
    verif.yield %2, %3 : i32, i32
  }

  // CHECK: [[EQ2:%.+]] = smt.solver() : () -> i1
  // CHECK: [[V9:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK: [[V10:%.+]] = smt.distinct [[V9]], [[V9]] : !smt.bv<32>
  // CHECK: smt.assert [[V10]]
  %2 = verif.lec first {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  } second {
  ^bb0(%arg1: i32):
    verif.yield %arg1 : i32
  }

  %3 = verif.lec first {
  ^bb0(%arg1: i32):
    verif.yield
  } second {
  ^bb0(%arg1: i32):
    verif.yield
  }

  // CHECK: return [[EQ]], [[EQ2]], %true
  return %1, %2, %3 : i1, i1, i1
}

func.func @test_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 1
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk, %c0_i1 : !seq.clock, i1
  }
  loop {
    ^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32, %stateArg: i1):
    %from_clock = seq.from_clock %clk
    %c-1_i1 = hw.constant -1 : i1
    %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
    %newStateArg = comb.xor %stateArg, %c-1_i1 : i1
    %newclk = seq.to_clock %neg_clock
    verif.yield %newclk, %newStateArg : !seq.clock, i1
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32):
    %c-1_i32 = hw.constant -1 : i32
    %0 = comb.add %arg0, %state0 : i32
    // %state0 is the result of a seq.compreg taking %0 as input
    %2 = comb.xor %state0, %c-1_i32 : i32
    %false = hw.constant false
    // Arbitrary assertion so op verifies
    verif.assert %false : i1
    verif.yield %2, %0 : i32, i32
  }
  func.return %bmc : i1
}
