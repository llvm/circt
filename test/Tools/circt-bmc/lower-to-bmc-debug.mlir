// RUN: circt-opt --lower-to-bmc="top-module=Simple bound=1" %s | FileCheck %s --check-prefix=SIMPLE
// RUN: circt-opt --lower-to-bmc="top-module=Passthrough bound=2" %s | FileCheck %s --check-prefix=PASSTHROUGH

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
