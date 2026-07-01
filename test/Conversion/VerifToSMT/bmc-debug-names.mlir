// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_bmc_debug_names() -> i1 {
// CHECK: [[INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[REG_DECL:%.+]] = smt.declare_fun "state_q" : !smt.bv<8>
// CHECK: dbg.variable "data_in", [[INPUT_DECL]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[REG_DECL]] : !smt.bv<8>
// CHECK: scf.for [[STEP:%.+]] = {{%.+}} iter_args([[CLK_ARG:%.+]] = {{%.+}}, [[INPUT_ARG:%.+]] = [[INPUT_DECL]], [[REG_ARG:%.+]] = [[REG_DECL]], {{%.+}})
// CHECK: dbg.variable "data_in", [[INPUT_ARG]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[REG_ARG]] : !smt.bv<8>
// CHECK: dbg.trace [[STEP]], "data_in", [[INPUT_ARG]] : i32, !smt.bv<8>
// CHECK: dbg.trace [[STEP]], "state_q", [[REG_ARG]] : i32, !smt.bv<8>
// CHECK: [[CIRCUIT_RESULT:%.+]] = func.call @bmc_circuit{{(_[0-9]+)?}}([[CLK_ARG]], [[INPUT_ARG]], [[REG_ARG]]) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NOT: smt.declare_fun "input_1"
// CHECK-NOT: smt.declare_fun "reg_0"
// CHECK: [[NEXT_INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[NEXT_REG:%.+]] = smt.ite {{%.+}}, [[CIRCUIT_RESULT]], [[REG_ARG]] : !smt.bv<8>
// CHECK: dbg.variable "data_in", [[NEXT_INPUT_DECL]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[NEXT_REG]] : !smt.bv<8>
// CHECK: scf.yield {{%.+}}, [[NEXT_INPUT_DECL]], [[NEXT_REG]], {{%.+}} : !smt.bv<1>, !smt.bv<8>, !smt.bv<8>, i1
// CHECK-LABEL: func.func @test_bmc_debug_names_extra_loop_state() -> i1 {
// CHECK: [[INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[REG_DECL:%.+]] = smt.declare_fun "state_q" : !smt.bv<8>
// CHECK: [[REG_CONST:%.+]] = smt.bv.constant #smt.bv<7> : !smt.bv<8>
// CHECK: dbg.variable "data_in", [[INPUT_DECL]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[REG_DECL]] : !smt.bv<8>
// CHECK: dbg.variable "state_r", [[REG_CONST]] : !smt.bv<8>
// CHECK: scf.for {{%.+}} iter_args([[CLK_ARG:%.+]] = {{%.+}}, [[INPUT_ARG:%.+]] = [[INPUT_DECL]], [[REG0_ARG:%.+]] = [[REG_DECL]], [[REG1_ARG:%.+]] = [[REG_CONST]], {{%.+}}, {{%.+}})
// CHECK: dbg.variable "data_in", [[INPUT_ARG]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[REG0_ARG]] : !smt.bv<8>
// CHECK: dbg.variable "state_r", [[REG1_ARG]] : !smt.bv<8>
// CHECK: [[CIRCUIT_RESULT0:%.+]]:2 = func.call @bmc_circuit{{(_[0-9]+)?}}([[CLK_ARG]], [[INPUT_ARG]], [[REG0_ARG]], [[REG1_ARG]]) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<8>, !smt.bv<8>) -> (!smt.bv<8>, !smt.bv<8>)
// CHECK: [[NEXT_INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[NEXT_REG0:%.+]] = smt.ite {{%.+}}, [[CIRCUIT_RESULT0]]#0, [[REG0_ARG]] : !smt.bv<8>
// CHECK: [[NEXT_REG1:%.+]] = smt.ite {{%.+}}, [[CIRCUIT_RESULT0]]#1, [[REG1_ARG]] : !smt.bv<8>
// CHECK: dbg.variable "data_in", [[NEXT_INPUT_DECL]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[NEXT_REG0]] : !smt.bv<8>
// CHECK: dbg.variable "state_r", [[NEXT_REG1]] : !smt.bv<8>
// CHECK: scf.yield {{%.+}}, [[NEXT_INPUT_DECL]], [[NEXT_REG0]], [[NEXT_REG1]], {{%.+}}, {{%.+}} : !smt.bv<1>, !smt.bv<8>, !smt.bv<8>, !smt.bv<8>, !smt.bv<1>, i1
// CHECK-LABEL: func.func @test_bmc_debug_names_const_init() -> i1 {
// CHECK: [[INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[REG_CONST:%.+]] = smt.bv.constant #smt.bv<42> : !smt.bv<8>
// CHECK: dbg.variable "data_in", [[INPUT_DECL]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[REG_CONST]] : !smt.bv<8>
// CHECK: scf.for {{%.+}} iter_args([[CLK_ARG:%.+]] = {{%.+}}, [[INPUT_ARG:%.+]] = [[INPUT_DECL]], [[REG_ARG:%.+]] = [[REG_CONST]], {{%.+}})
// CHECK: dbg.variable "data_in", [[INPUT_ARG]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[REG_ARG]] : !smt.bv<8>
// CHECK: [[CIRCUIT_RESULT:%.+]] = func.call @bmc_circuit{{(_[0-9]+)?}}([[CLK_ARG]], [[INPUT_ARG]], [[REG_ARG]]) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK: [[NEXT_INPUT_DECL:%.+]] = smt.declare_fun "data_in" : !smt.bv<8>
// CHECK: [[NEXT_REG:%.+]] = smt.ite {{%.+}}, [[CIRCUIT_RESULT]], [[REG_ARG]] : !smt.bv<8>
// CHECK: dbg.variable "data_in", [[NEXT_INPUT_DECL]] : !smt.bv<8>
// CHECK: dbg.variable "state_q", [[NEXT_REG]] : !smt.bv<8>
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

func.func @test_bmc_debug_names_extra_loop_state() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 2 initial_values [unit, 7 : i8]
  init {
    %clk = seq.const_clock low
    %false = hw.constant false
    verif.yield %clk, %false : !seq.clock, i1
  }
  loop {
  ^bb0(%clk: !seq.clock, %shadow: i1):
    %true = hw.constant true
    %next_shadow = comb.xor %shadow, %true : i1
    verif.yield %clk, %next_shadow : !seq.clock, i1
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i8, %state0: i8, %state1: i8):
    %scope = dbg.scope "Top", "Top"
    dbg.variable "data_in", %arg0 scope %scope : i8
    dbg.variable "state_q", %state0 scope %scope : i8
    dbg.variable "state_r", %state1 scope %scope : i8
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %state0, %arg0 : i8, i8
  }
  func.return %bmc : i1
}

func.func @test_bmc_debug_names_const_init() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 1 initial_values [42 : i8]
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
