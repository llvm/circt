// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK: func.func @lower_assert([[ARG0:%.+]]: i1)
// CHECK:   [[CAST:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : i1 to !smt.bv<1>
// CHECK:   [[Cn1_BV:%.+]] = smt.bv.constant #smt.bv<-1>
// CHECK:   [[EQ:%.+]] = smt.eq [[CAST]], [[Cn1_BV]]
// CHECK:   [[NEQ:%.+]] = smt.not [[EQ]]
// CHECK:   smt.assert [[NEQ]]
// CHECK:   return

func.func @lower_assert(%arg0: i1) {
  verif.assert %arg0 : i1
  return
}

// CHECK: func.func @lower_assume([[ARG0:%.+]]: i1)
// CHECK:   [[CAST:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : i1 to !smt.bv<1>
// CHECK:   [[Cn1_BV:%.+]] = smt.bv.constant #smt.bv<-1>
// CHECK:   [[EQ:%.+]] = smt.eq [[CAST]], [[Cn1_BV]]
// CHECK:   smt.assert [[EQ]]
// CHECK:   return

func.func @lower_assume(%arg0: i1) {
  verif.assume %arg0 : i1
  return
}

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
  // CHECK-DAG: [[IN0:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-DAG: [[IN1:%.+]] = smt.declare_fun : !smt.bv<32>
  // CHECK-DAG: [[V0:%.+]] = builtin.unrealized_conversion_cast [[IN0]] : !smt.bv<32> to i32
  // CHECK-DAG: [[V1:%.+]] = builtin.unrealized_conversion_cast [[IN1]] : !smt.bv<32> to i32
  // CHECK-DAG: [[V2:%.+]]:2 = "some_op"([[V0]], [[V1]]) : (i32, i32) -> (i32, i32)
  // CHECK-DAG: [[V3:%.+]] = builtin.unrealized_conversion_cast [[V2]]#0 : i32 to !smt.bv<32>
  // CHECK-DAG: [[V4:%.+]] = smt.distinct [[IN0]], [[V3]] : !smt.bv<32>
  // CHECK-DAG: [[V5:%.+]] = builtin.unrealized_conversion_cast [[V2]]#1 : i32 to !smt.bv<32>
  // CHECK-DAG: [[V6:%.+]] = smt.distinct [[IN1]], [[V5]] : !smt.bv<32>
  // CHECK-DAG: [[V7:%.+]] = smt.or [[V4]], [[V6]]
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

// CHECK-LABEL:  func.func @test_bmc() -> i1 {
// CHECK:    [[BMC:%.+]] = smt.solver
// CHECK:      [[INIT:%.+]]:2 = func.call @bmc_init()
// CHECK:      smt.push 1
// CHECK:      [[F0:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:      [[F1:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:      [[C42_BV32:%.+]] = smt.bv.constant #smt.bv<42> : !smt.bv<32>
// CHECK:      [[ARRAYFUN:%.+]] = smt.declare_fun : !smt.array<[!smt.bv<1> -> !smt.bv<32>]>
// CHECK:      [[C0_I32:%.+]] = arith.constant 0 : i32
// CHECK:      [[C1_I32:%.+]] = arith.constant 1 : i32
// CHECK:      [[C10_I32:%.+]] = arith.constant 10 : i32
// CHECK:      [[FALSE:%.+]] = arith.constant false
// CHECK:      [[TRUE:%.+]] = arith.constant true
// CHECK:      [[FOR:%.+]]:7 = scf.for [[ARG0:%.+]] = [[C0_I32]] to [[C10_I32]] step [[C1_I32]] iter_args([[ARG1:%.+]] = [[INIT]]#0, [[ARG2:%.+]] = [[F0]], [[ARG3:%.+]] = [[F1]], [[ARG4:%.+]] = [[C42_BV32]], [[ARG5:%.+]] = [[ARRAYFUN]], [[ARG6:%.+]] = [[INIT]]#1, [[ARG7:%.+]] = [[FALSE]])
// CHECK:        smt.pop 1
// CHECK:        smt.push 1
// CHECK:        [[CIRCUIT:%.+]]:4 = func.call @bmc_circuit([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]])
// CHECK:        [[SMTCHECK:%.+]] = smt.check sat {
// CHECK:          smt.yield [[TRUE]]
// CHECK:        } unknown {
// CHECK:          smt.yield [[TRUE]]
// CHECK:        } unsat {
// CHECK:          smt.yield [[FALSE]]
// CHECK:        }
// CHECK:        [[ORI:%.+]] = arith.ori [[SMTCHECK]], [[ARG7]]
// CHECK:        [[LOOP:%.+]]:2 = func.call @bmc_loop([[ARG1]], [[ARG6]])
// CHECK:        [[F2:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:        [[OLDCLOCKLOW:%.+]] = smt.bv.not [[ARG1]]
// CHECK:        [[BVPOSEDGE:%.+]] = smt.bv.and [[OLDCLOCKLOW]], [[LOOP]]#0
// CHECK:        [[BVTRUE:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK:        [[BOOLPOSEDGE:%.+]] = smt.eq [[BVPOSEDGE]], [[BVTRUE]]
// CHECK:        [[NEWREG1:%.+]] = smt.ite [[BOOLPOSEDGE]], [[CIRCUIT]]#1, [[ARG3]]
// CHECK:        [[NEWREG2:%.+]] = smt.ite [[BOOLPOSEDGE]], [[CIRCUIT]]#2, [[ARG4]]
// CHECK:        [[NEWREG3:%.+]] = smt.ite [[BOOLPOSEDGE]], [[CIRCUIT]]#3, [[ARG5]]
// CHECK:        scf.yield [[LOOP]]#0, [[F2]], [[NEWREG1]], [[NEWREG2]], [[NEWREG3]], [[LOOP]]#1, [[ORI]]
// CHECK:      }
// CHECK:      [[XORI:%.+]] = arith.xori [[FOR]]#6, [[TRUE]]
// CHECK:      smt.yield [[XORI]]
// CHECK:    }
// CHECK:    return [[BMC]]
// CHECK:  }
// CHECK-LABEL:  func.func @bmc_init() -> (!smt.bv<1>, !smt.bv<1>) {
// CHECK:    [[FALSE:%.+]] = hw.constant false
// CHECK:    [[TOCLOCK:%.+]] = seq.to_clock %false
// CHECK:    [[C0:%.+]] = builtin.unrealized_conversion_cast [[TOCLOCK]] : !seq.clock to !smt.bv<1>
// CHECK:    [[C1:%.+]] = builtin.unrealized_conversion_cast [[FALSE]] : i1 to !smt.bv<1>
// CHECK:    return [[C0]], [[C1]]
// CHECK:  }
// CHECK:  func.func @bmc_loop([[ARGO:%.+]]: !smt.bv<1>, [[ARG1:%.+]]: !smt.bv<1>)
// CHECK:    [[C2:%.+]] = builtin.unrealized_conversion_cast [[ARG1]] : !smt.bv<1> to i1
// CHECK:    [[C3:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : !smt.bv<1> to !seq.clock
// CHECK:    [[FROMCLOCK:%.+]] = seq.from_clock [[C3]]
// CHECK:    [[TRUE]] = hw.constant true
// CHECK:    [[NCLOCK:%.+]] = comb.xor [[FROMCLOCK]], [[TRUE]]
// CHECK:    [[NARG:%.+]] = comb.xor [[C2]], [[TRUE]]
// CHECK:    [[TOCLOCK:%.+]] = seq.to_clock [[NCLOCK]]
// CHECK:    [[C4:%.+]] = builtin.unrealized_conversion_cast [[TOCLOCK]] : !seq.clock to !smt.bv<1>
// CHECK:    [[C5:%.+]] = builtin.unrealized_conversion_cast [[NARG]] : i1 to !smt.bv<1>
// CHECK:    return [[C4]], [[C5]]
// CHECK:  }
// CHECK:  func.func @bmc_circuit([[ARGO:%.+]]: !smt.bv<1>, [[ARG1:%.+]]: !smt.bv<32>, [[ARG2:%.+]]: !smt.bv<32>, [[ARG3:%.+]]: !smt.bv<32>, [[ARG4:%.+]]: !smt.array<[!smt.bv<1> -> !smt.bv<32>]>)
// CHECK:    [[C6:%.+]] = builtin.unrealized_conversion_cast [[ARG2]] : !smt.bv<32> to i32
// CHECK:    [[C7:%.+]] = builtin.unrealized_conversion_cast [[ARG1]] : !smt.bv<32> to i32
// CHECK:    [[CN1_I32:%.+]] = hw.constant -1 : i32
// CHECK:    [[ADD:%.+]] = comb.add [[C7]], [[C6]]
// CHECK:    [[XOR:%.+]] = comb.xor [[C6]], [[CN1_I32]]
// CHECK:    [[C9:%.+]] = builtin.unrealized_conversion_cast [[XOR]] : i32 to !smt.bv<32>
// CHECK:    [[C10:%.+]] = builtin.unrealized_conversion_cast [[ADD]] : i32 to !smt.bv<32>
// CHECK:    return [[C9]], [[C10]], [[ARG3]], [[ARG4]]
// CHECK:  }

// RUN: circt-opt %s --convert-verif-to-smt="rising-clocks-only=true" --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=CHECK1
// CHECK1-LABEL:  func.func @test_bmc() -> i1 {
// CHECK1:        [[CIRCUIT:%.+]]:4 = func.call @bmc_circuit(
// CHECK1:        [[SMTCHECK:%.+]] = smt.check
// CHECK1:        [[ORI:%.+]] = arith.ori [[SMTCHECK]], {{%.*}}
// CHECK1:        [[LOOP:%.+]]:2 = func.call @bmc_loop({{%.*}}, {{%.*}})
// CHECK1:        [[F:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK1:        scf.yield [[LOOP]]#0, [[F]], [[CIRCUIT]]#1, [[CIRCUIT]]#2, [[CIRCUIT]]#3, [[LOOP]]#1, [[ORI]]

func.func @test_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 3 initial_values [unit, 42, unit]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk, %c0_i1 : !seq.clock, i1
  }
  loop {
    ^bb0(%clk: !seq.clock, %stateArg: i1):
    %from_clock = seq.from_clock %clk
    %c-1_i1 = hw.constant -1 : i1
    %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
    %newStateArg = comb.xor %stateArg, %c-1_i1 : i1
    %newclk = seq.to_clock %neg_clock
    verif.yield %newclk, %newStateArg : !seq.clock, i1
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32, %state1: i32, %state2: !hw.array<2xi32>):
    %c-1_i32 = hw.constant -1 : i32
    %0 = comb.add %arg0, %state0 : i32
    // %state0 is the result of a seq.compreg taking %0 as input
    %2 = comb.xor %state0, %c-1_i32 : i32
    verif.yield %2, %0, %state1, %state2 : i32, i32, i32, !hw.array<2xi32>
  }
  func.return %bmc : i1
}
