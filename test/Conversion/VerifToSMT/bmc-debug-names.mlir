// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_bmc_debug_names() -> i1 {
// CHECK: [[INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[REG_DECL:%.+]] = smt.declare_fun "state_q" : !smt.bv<8>
// CHECK: scf.for {{%.+}} iter_args([[CLK_ARG:%.+]] = {{%.+}}, [[INPUT_ARG:%.+]] = [[INPUT_DECL]], [[REG_ARG:%.+]] = [[REG_DECL]], {{%.+}})
// CHECK: [[CIRCUIT_RESULT:%.+]] = func.call @bmc_circuit([[CLK_ARG]], [[INPUT_ARG]], [[REG_ARG]]) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NOT: smt.declare_fun "input_1"
// CHECK-NOT: smt.declare_fun "reg_0"
// CHECK: [[NEXT_INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[NEXT_REG:%.+]] = smt.ite {{%.+}}, [[CIRCUIT_RESULT]], [[REG_ARG]] : !smt.bv<8>
// CHECK: scf.yield {{%.+}}, [[NEXT_INPUT_DECL]], [[NEXT_REG]], {{%.+}} : !smt.bv<1>, !smt.bv<8>, !smt.bv<8>, i1

func.func @test_bmc_debug_names() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 1 initial_values [unit]
  init {
    %clk = seq.const_clock low
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i8, %state0: i8):
    %scope = dbg.scope "Top", "Top"
    dbg.variable "data_in", %arg0 scope %scope : i8
    dbg.variable "state_q", %state0 scope %scope : i8
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : i8
  }
  func.return %bmc : i1
}
