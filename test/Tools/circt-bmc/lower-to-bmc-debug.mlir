// RUN: circt-opt --lower-to-bmc="top-module=Simple bound=1" %s | FileCheck %s --check-prefix=SIMPLE
// RUN: circt-opt --lower-to-bmc="top-module=Passthrough bound=2" %s | FileCheck %s --check-prefix=PASSTHROUGH
// RUN: circt-bmc %s --module=Sequential -b=2 --emit-mlir | FileCheck %s --check-prefix=END-TO-END
// RUN: circt-bmc %s --module=Aggregate -b=2 --emit-mlir | FileCheck %s --check-prefix=AGGREGATE
// RUN: circt-bmc %s --module=Sequential -b=2 --emit-llvm | FileCheck %s --check-prefix=LLVM
// RUN: circt-bmc %s --module=Aggregate -b=2 --emit-llvm | FileCheck %s --check-prefix=AGGREGATE-LLVM

// SIMPLE: ^bb0([[IN:%.+]]: i8):
// SIMPLE: [[SCOPE:%.+]] = dbg.scope "Simple", "Simple"
// SIMPLE-DAG: dbg.variable "in", [[IN]] scope [[SCOPE]] : i8
// SIMPLE-DAG: dbg.variable "out", [[IN]] scope [[SCOPE]] : i8

hw.module @Simple(in %in: i8, out out: i8) attributes {num_regs = 0 : i32, initial_values = []} {
  hw.output %in : i8
}

// PASSTHROUGH: ^bb0([[A:%.+]]: i4, [[B:%.+]]: i4):
// PASSTHROUGH: [[SUM:%.+]] = comb.add [[A]], [[B]] : i4
// PASSTHROUGH: [[SCOPE:%.+]] = dbg.scope "Passthrough", "Passthrough"
// PASSTHROUGH-DAG: dbg.variable "a", [[A]] scope [[SCOPE]] : i4
// PASSTHROUGH-DAG: dbg.variable "b", [[B]] scope [[SCOPE]] : i4
// PASSTHROUGH-DAG: dbg.variable "sum", [[SUM]] scope [[SCOPE]] : i4

hw.module @Passthrough(in %a: i4, in %b: i4, out sum: i4) attributes {num_regs = 0 : i32, initial_values = []} {
  %0 = comb.add %a, %b : i4
  hw.output %0 : i4
}

// Verify that the complete circt-bmc lowering pipeline carries the names of
// externalized circuit values to per-cycle BMC traces.
// END-TO-END-LABEL: func.func @Sequential()
// END-TO-END: [[INPUT_INIT:%.+]] = smt.declare_fun "state_q_next" : !smt.bv<8>
// END-TO-END: [[STATE_INIT:%.+]] = smt.declare_fun "out" : !smt.bv<8>
// END-TO-END: scf.for [[STEP:%.+]] = {{%.+}} iter_args({{.*}}[[INPUT:%.+]] = [[INPUT_INIT]], [[STATE:%.+]] = [[STATE_INIT]]{{.*}})
// END-TO-END: verif.bmc.trace [[STEP]], "state_q_next", [[INPUT]] : i32, !smt.bv<8>
// END-TO-END: verif.bmc.trace [[STEP]], "out", [[STATE]] : i32, !smt.bv<8>
// END-TO-END: [[NEXT_INPUT:%.+]] = smt.declare_fun "state_q_next" : !smt.bv<8>
// END-TO-END: scf.yield {{.*}}[[NEXT_INPUT]]{{.*}} :

// LLVM-LABEL: define void @Sequential(ptr

hw.module @Sequential(in %clk: !seq.clock, in %in: i8, in %state_q: i8,
                     out out: i8, out state_q_next: i8)
    attributes {num_regs = 1 : i32, initial_values = [unit]} {
  %valid = comb.icmp eq %state_q, %in : i8
  verif.assert %valid : i1
  hw.output %state_q, %in : i8, i8
}

// Ensure aggregate state is tracked as an SMT array instead of being silently
// omitted from the trace.
// AGGREGATE-LABEL: func.func @Aggregate()
// AGGREGATE: [[STATE_INIT:%.+]] = smt.declare_fun "res_state" : !smt.array<[!smt.bv<1> -> !smt.bv<32>]>
// AGGREGATE: scf.for [[STEP:%.+]] = {{%.+}} iter_args({{.*}}[[STATE:%.+]] = [[STATE_INIT]]{{.*}})
// AGGREGATE: verif.bmc.trace [[STEP]], "res_state", [[STATE]] : i32, !smt.array<[!smt.bv<1> -> !smt.bv<32>]>
// AGGREGATE-LLVM-LABEL: define void @Aggregate(ptr
// AGGREGATE-LLVM-NOT: call void @circt_bmc_record_trace

hw.module @Aggregate(in %clk: !seq.clock) {
  %zero = hw.constant 0 : i32
  %array = hw.array_create %zero, %zero : i32
  %res = seq.compreg %array, %clk : !hw.array<2xi32>
  %index = hw.constant 0 : i1
  %element = hw.array_get %res[%index] : !hw.array<2xi32>, i1
  %valid = comb.icmp eq %zero, %element : i32
  verif.assert %valid : i1
}
