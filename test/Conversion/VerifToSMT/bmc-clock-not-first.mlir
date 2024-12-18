// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:  func.func @test_bmc_clock_not_first() -> i1 {
// CHECK:    [[BMC:%.+]] = smt.solver
// CHECK:      [[INIT:%.+]] = func.call @bmc_init()
// CHECK:      smt.push 1
// CHECK:      [[F0:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:      [[F1:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:      [[C0_I32:%.+]] = arith.constant 0 : i32
// CHECK:      [[C1_I32:%.+]] = arith.constant 1 : i32
// CHECK:      [[C10_I32:%.+]] = arith.constant 10 : i32
// CHECK:      [[FALSE:%.+]] = arith.constant false
// CHECK:      [[TRUE:%.+]] = arith.constant true
// CHECK:      [[FOR:%.+]]:4 = scf.for [[ARG0:%.+]] = [[C0_I32]] to [[C10_I32]] step [[C1_I32]] iter_args([[ARG1:%.+]] = [[F0]], [[ARG2:%.+]] = [[INIT]], [[ARG3:%.+]] = [[F1]], [[ARG4:%.+]] = [[FALSE]])
// CHECK:        smt.pop 1
// CHECK:        smt.push 1
// CHECK:        [[CIRCUIT:%.+]]:2 = func.call @bmc_circuit([[ARG1]], [[ARG2]], [[ARG3]])
// CHECK:        [[SMTCHECK:%.+]] = smt.check sat {
// CHECK:          smt.yield [[TRUE]]
// CHECK:        } unknown {
// CHECK:          smt.yield [[TRUE]]
// CHECK:        } unsat {
// CHECK:          smt.yield [[FALSE]]
// CHECK:        }
// CHECK:        [[ORI:%.+]] = arith.ori [[SMTCHECK]], [[ARG4]]
// CHECK:        [[LOOP:%.+]] = func.call @bmc_loop([[ARG2]])
// CHECK:        [[F2:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK:        [[OLDCLOCKLOW:%.+]] = smt.bv.not [[ARG2]]
// CHECK:        [[BVPOSEDGE:%.+]] = smt.bv.and [[OLDCLOCKLOW]], [[LOOP]]
// CHECK:        [[BVTRUE:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK:        [[BOOLPOSEDGE:%.+]] = smt.eq [[BVPOSEDGE]], [[BVTRUE]]
// CHECK:        [[NEWREG1:%.+]] = smt.ite [[BOOLPOSEDGE]], [[CIRCUIT]]#1, [[ARG3]]
// CHECK:        scf.yield [[F2]], [[LOOP]], [[NEWREG1]], [[ORI]]
// CHECK:      }
// CHECK:      [[XORI:%.+]] = arith.xori [[FOR]]#3, [[TRUE]]
// CHECK:      smt.yield [[XORI]]
// CHECK:    }
// CHECK:    return [[BMC]]
// CHECK:  }
// CHECK-LABEL:  func.func @bmc_init() -> !smt.bv<1> {
// CHECK:    [[INITCLK:%.+]] = seq.const_clock low
// CHECK:    [[C0:%.+]] = builtin.unrealized_conversion_cast [[INITCLK]] : !seq.clock to !smt.bv<1>
// CHECK:    return [[C0]]
// CHECK:  }
// CHECK:  func.func @bmc_loop([[ARGO:%.+]]: !smt.bv<1>)
// CHECK:    [[C1:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : !smt.bv<1> to !seq.clock
// CHECK:    [[FROMCLOCK:%.+]] = seq.from_clock [[C1]]
// CHECK:    [[TRUE]] = hw.constant true
// CHECK:    [[NCLOCK:%.+]] = comb.xor [[FROMCLOCK]], [[TRUE]]
// CHECK:    [[TOCLOCK:%.+]] = seq.to_clock [[NCLOCK]]
// CHECK:    [[C4:%.+]] = builtin.unrealized_conversion_cast [[TOCLOCK]] : !seq.clock to !smt.bv<1>
// CHECK:    return [[C4]]
// CHECK:  }
// CHECK:  func.func @bmc_circuit([[ARG0:%.+]]: !smt.bv<32>, [[ARG1:%.+]]: !smt.bv<1>, [[ARG2:%.+]]: !smt.bv<32>)
// CHECK:    [[C6:%.+]] = builtin.unrealized_conversion_cast [[ARG2]] : !smt.bv<32> to i32
// CHECK:    [[C7:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : !smt.bv<32> to i32
// CHECK:    [[CN1_I32:%.+]] = hw.constant -1 : i32
// CHECK:    [[ADD:%.+]] = comb.add [[C7]], [[C6]]
// CHECK:    [[XOR:%.+]] = comb.xor [[C6]], [[CN1_I32]]
// CHECK:    [[C9:%.+]] = builtin.unrealized_conversion_cast [[XOR]] : i32 to !smt.bv<32>
// CHECK:    [[C10:%.+]] = builtin.unrealized_conversion_cast [[ADD]] : i32 to !smt.bv<32>
// CHECK:    return [[C9]], [[C10]]
// CHECK:  }

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
    %c-1_i32 = hw.constant -1 : i32
    %0 = comb.add %arg0, %state0 : i32
    // %state0 is the result of a seq.compreg taking %0 as input
    %2 = comb.xor %state0, %c-1_i32 : i32
    verif.yield %2, %0 : i32, i32
  }
  func.return %bmc : i1
}
