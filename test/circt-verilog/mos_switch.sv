// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// NMOS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @nmos_test(
// CHECK: [[Z_NMOS:%.+]] = moore.constant bZ : l1
// CHECK: [[ZERO_NMOS:%.+]] = moore.constant 0 : l1
// CHECK: [[IN_NMOS:%.+]] = moore.extract %in from 2 : l4 -> l1
// CHECK: [[COND_NMOS:%.+]] = moore.case_eq %control, [[ZERO_NMOS]] : l1
// CHECK: moore.conditional [[COND_NMOS]] : i1 -> l1
// CHECK: moore.yield [[Z_NMOS]] : l1
// CHECK: moore.yield [[IN_NMOS]] : l1

module nmos_test(
  input wire [3:0] in,
  input wire control,
  output wire out
);

  nmos (out, in[2], control);

endmodule


//===----------------------------------------------------------------------===//
// PMOS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @pmos_test(
// CHECK: [[Z_PMOS:%.+]] = moore.constant bZ : l1
// CHECK: [[ONE_PMOS:%.+]] = moore.constant 1 : l1
// CHECK: [[IN_PMOS:%.+]] = moore.extract %in from 2 : l4 -> l1
// CHECK: [[COND_PMOS:%.+]] = moore.case_eq %control, [[ONE_PMOS]] : l1
// CHECK: moore.conditional [[COND_PMOS]] : i1 -> l1
// CHECK: moore.yield [[Z_PMOS]] : l1
// CHECK: moore.yield [[IN_PMOS]] : l1

module pmos_test(
  input wire [3:0] in,
  input wire control,
  output wire out
);

  pmos (out, in[2], control);

endmodule


//===----------------------------------------------------------------------===//
// CMOS
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @cmos_test(
// CHECK: [[Z_CMOS:%.+]] = moore.constant bZ : l1
// CHECK: [[ZERO_CMOS:%.+]] = moore.constant 0 : l1
// CHECK: [[ONE_CMOS:%.+]] = moore.constant 1 : l1
// CHECK: [[IN_CMOS:%.+]] = moore.extract %in from 2 : l4 -> l1
// CHECK: [[N_ENABLED_CMOS:%.+]] = moore.eq %n_control, [[ONE_CMOS]] : l1 -> l1
// CHECK: [[P_ENABLED_CMOS:%.+]] = moore.eq %p_control, [[ZERO_CMOS]] : l1 -> l1
// CHECK: [[COND_CMOS:%.+]] = moore.and [[N_ENABLED_CMOS]], [[P_ENABLED_CMOS]] : l1
// CHECK: moore.conditional [[COND_CMOS]] : l1 -> l1
// CHECK: moore.yield [[IN_CMOS]] : l1
// CHECK: moore.yield [[Z_CMOS]] : l1

module cmos_test(
  input wire [3:0] in,
  input wire n_control,
  input wire p_control,
  output wire out
);

  cmos (out, in[2], n_control, p_control);

endmodule


//===----------------------------------------------------------------------===//
// RNMOS (Resistive NMOS)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @rnmos_test(
// CHECK: [[Z_RNMOS:%.+]] = moore.constant bZ : l1
// CHECK: [[ZERO_RNMOS:%.+]] = moore.constant 0 : l1
// CHECK: [[IN_RNMOS:%.+]] = moore.extract %in from 2 : l4 -> l1
// CHECK: [[COND_RNMOS:%.+]] = moore.case_eq %control, [[ZERO_RNMOS]] : l1
// CHECK: moore.conditional [[COND_RNMOS]] : i1 -> l1
// CHECK: moore.yield [[Z_RNMOS]] : l1
// CHECK: moore.yield [[IN_RNMOS]] : l1

module rnmos_test(
  input wire [3:0] in,
  input wire control,
  output wire out
);

  rnmos (out, in[2], control);

endmodule


//===----------------------------------------------------------------------===//
// RPMOS (Resistive PMOS)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @rpmos_test(
// CHECK: [[Z_RPMOS:%.+]] = moore.constant bZ : l1
// CHECK: [[ONE_RPMOS:%.+]] = moore.constant 1 : l1
// CHECK: [[IN_RPMOS:%.+]] = moore.extract %in from 2 : l4 -> l1
// CHECK: [[COND_RPMOS:%.+]] = moore.case_eq %control, [[ONE_RPMOS]] : l1
// CHECK: moore.conditional [[COND_RPMOS]] : i1 -> l1
// CHECK: moore.yield [[Z_RPMOS]] : l1
// CHECK: moore.yield [[IN_RPMOS]] : l1

module rpmos_test(
  input wire [3:0] in,
  input wire control,
  output wire out
);

  rpmos (out, in[2], control);

endmodule


//===----------------------------------------------------------------------===//
// RCMOS (Resistive CMOS)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @rcmos_test(
// CHECK: [[Z_RCMOS:%.+]] = moore.constant bZ : l1
// CHECK: [[ZERO_RCMOS:%.+]] = moore.constant 0 : l1
// CHECK: [[ONE_RCMOS:%.+]] = moore.constant 1 : l1
// CHECK: [[IN_RCMOS:%.+]] = moore.extract %in from 2 : l4 -> l1
// CHECK: [[N_ENABLED_RCMOS:%.+]] = moore.eq %n_control, [[ONE_RCMOS]] : l1 -> l1
// CHECK: [[P_ENABLED_RCMOS:%.+]] = moore.eq %p_control, [[ZERO_RCMOS]] : l1 -> l1
// CHECK: [[COND_RCMOS:%.+]] = moore.and [[N_ENABLED_RCMOS]], [[P_ENABLED_RCMOS]] : l1
// CHECK: moore.conditional [[COND_RCMOS]] : l1 -> l1
// CHECK: moore.yield [[IN_RCMOS]] : l1
// CHECK: moore.yield [[Z_RCMOS]] : l1

module rcmos_test(
  input wire [3:0] in,
  input wire n_control,
  input wire p_control,
  output wire out
);

  rcmos (out, in[2], n_control, p_control);

endmodule
