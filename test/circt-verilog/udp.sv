// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: hw.module @udp_mux(
// CHECK-DAG: [[NOT_SEL:%.+]] = comb.xor {{.*}}%sel{{.*}}
// CHECK-DAG: [[TERM1:%.+]] = comb.and {{.*}}%a{{.*}}
// CHECK-DAG: [[TERM2:%.+]] = comb.and {{.*}}%b{{.*}}
// CHECK-DAG: [[OUT:%.+]] = comb.or {{.*}}
// CHECK: hw.output [[OUT]]
primitive udp_mux(out, sel, a, b);
  output out;
  input sel, a, b;
  table
    0  1  ? : 1 ;
    1  ?  1 : 1 ;
    0  0  ? : 0 ;
    1  ?  0 : 0 ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @TestCombUdp(
module TestCombUdp(input logic sel, input logic a, input logic b, output logic out);
  // CHECK: [[U_MUX:%.+]] = hw.instance "u_mux" @udp_mux(sel: %sel: i1, a: %a: i1, b: %b: i1) -> (out: i1)
  // CHECK: hw.output [[U_MUX]]
  udp_mux u_mux (out, sel, a, b);
endmodule

// CHECK-LABEL: hw.module @TestCombUdpDelay(
module TestCombUdpDelay(input logic sel, input logic a, input logic b, output logic out);
  // CHECK-DAG: [[TIME:%.+]] = llhd.constant_time <5000000fs, 0d, 0e>
  // CHECK-DAG: [[SIG:%.+]] = llhd.sig : <i1>
  // CHECK-DAG: [[U_MUX_DELAY:%.+]] = hw.instance "u_mux_delay" @udp_mux(sel: %sel: i1, a: %a: i1, b: %b: i1) -> (out: i1)
  // CHECK-DAG: llhd.drv [[SIG]], [[U_MUX_DELAY]] after [[TIME]] : i1
  // CHECK-DAG: [[PRB:%.+]] = llhd.prb [[SIG]] : i1
  // CHECK: hw.output [[PRB]]
  udp_mux #5 u_mux_delay (out, sel, a, b);
endmodule

// CHECK-LABEL: hw.module @udp_dff(
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff(q, clk, d);
  output reg q;
  input clk, d;
  table
    (01) 0 : ? : 0 ;
    (01) 1 : ? : 1 ;
    (10) ? : ? : - ;
    ?   (??): ? : - ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @udp_dff_neg(
// CHECK: [[CLK_INV:%.+]] = seq.clock_inv %clk
// CHECK: [[REG:%.+]] = seq.firreg %d clock [[CLK_INV]] preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_neg(q, clk, d);
  output reg q;
  input clk, d;
  table
    (10) 0 : ? : 0 ;
    (10) 1 : ? : 1 ;
    (01) ? : ? : - ;
    ?   (??): ? : - ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @udp_dff_en(
// CHECK: [[MUX:%.+]] = comb.mux bin %en, %d, [[REG:%.+]] : i1
// CHECK: [[REG]] = seq.firreg [[MUX]] clock %clk preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_en(q, clk, en, d);
  output reg q;
  input clk, en, d;
  table
    (01) 1 0 : ? : 0 ;
    (01) 1 1 : ? : 1 ;
    (01) 0 ? : ? : - ;
    (10) ? ? : ? : - ;
    ?    ? (??): ? : - ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @udp_dual_edge_dff(
// CHECK: [[SIG:%.+]] = llhd.sig
// CHECK: [[PROCESS:%.+]]:2 = llhd.process -> i1, i1 {
// CHECK: ^[[LOOP_BB:.+]](
// CHECK:   llhd.wait yield {{.*}}, (%clk : i1), ^[[CHECK_BB:.+]](%clk : i1)
// CHECK: ^[[CHECK_BB]]([[CLK_VAL:%.+]]: i1):
// CHECK:   [[NOT_OLD:%.+]] = comb.xor bin [[CLK_VAL]]
// CHECK:   [[POS:%.+]] = comb.and bin [[NOT_OLD]], %clk
// CHECK:   [[NOT_NEW:%.+]] = comb.xor bin %clk
// CHECK:   [[NEG:%.+]] = comb.and bin [[CLK_VAL]], [[NOT_NEW]]
// CHECK:   [[EDGE:%.+]] = comb.or bin [[POS]], [[NEG]]
// CHECK:   cf.cond_br [[EDGE]], ^[[LOOP_BB]](%d, {{.*}}), ^[[LOOP_BB]](
// CHECK: }
// CHECK: llhd.drv [[SIG]], [[PROCESS]]#0 after {{.*}} if [[PROCESS]]#1 : i1
primitive udp_dual_edge_dff(q, clk, d);
  output reg q;
  input clk, d;
  table
    (01) 0 : ? : 0 ;
    (01) 1 : ? : 1 ;
    (10) 0 : ? : 0 ;
    (10) 1 : ? : 1 ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @udp_dual_edge_mux(
// CHECK-DAG: [[TERM1:%.+]] = comb.and %clk, %da : i1
// CHECK-DAG: [[NOT_CLK:%.+]] = comb.xor %clk, %true : i1
// CHECK-DAG: [[TERM2:%.+]] = comb.and [[NOT_CLK]], %db : i1
// CHECK-DAG: [[VAL:%.+]] = comb.or [[TERM1]], [[TERM2]] : i1
// CHECK: llhd.process
primitive udp_dual_edge_mux(q, clk, da, db);
  output reg q;
  input clk, da, db;
  table
    (01) 0 ? : ? : 0 ;
    (01) 1 ? : ? : 1 ;
    (10) ? 0 : ? : 0 ;
    (10) ? 1 : ? : 1 ;
  endtable
endprimitive

// Verify that async rows in a sequential UDP are skipped.
// CHECK-LABEL: hw.module @udp_dff_skip_async(
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_skip_async(q, clk, d);
  output reg q;
  input clk, d;
  table
    1 0 : ? : 0 ;
    (01) 0 : ? : 0 ;
    (01) 1 : ? : 1 ;
    (10) ? : ? : - ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @TestPosedgeDFF(
module TestPosedgeDFF(input logic clk, input logic d, output logic q);
  // CHECK: [[U_DFF:%.+]] = hw.instance "u_dff" @udp_dff(clk: %clk: !seq.clock, d: %d: i1) -> (q: i1)
  // CHECK: hw.output [[U_DFF]]
  udp_dff u_dff (q, clk, d);
endmodule

// CHECK-LABEL: hw.module @TestNegedgeDFF(
module TestNegedgeDFF(input logic clk, input logic d, output logic q);
  // CHECK: [[U_DFF:%.+]] = hw.instance "u_dff" @udp_dff_neg(clk: %clk: !seq.clock, d: %d: i1) -> (q: i1)
  // CHECK: hw.output [[U_DFF]]
  udp_dff_neg u_dff (q, clk, d);
endmodule

// CHECK-LABEL: hw.module @TestEnabledDFF(
module TestEnabledDFF(input logic clk, input logic en, input logic d, output logic q);
  // CHECK: [[U_DFF:%.+]] = hw.instance "u_dff" @udp_dff_en(clk: %clk: !seq.clock, en: %en: i1, d: %d: i1) -> (q: i1)
  // CHECK: hw.output [[U_DFF]]
  udp_dff_en u_dff (q, clk, en, d);
endmodule

// CHECK-LABEL: hw.module @TestDualEdgeDFF(
module TestDualEdgeDFF(input logic clk, input logic d, output logic q);
  // CHECK: [[U_DET:%.+]] = hw.instance "u_det" @udp_dual_edge_dff(clk: %clk: i1, d: %d: i1) -> (q: i1)
  // CHECK: hw.output [[U_DET]]
  udp_dual_edge_dff u_det (q, clk, d);
endmodule

// CHECK-LABEL: hw.module @TestDualEdgeMux(
module TestDualEdgeMux(input logic clk, input logic da, input logic db, output logic q);
  // CHECK: [[U_ASYM:%.+]] = hw.instance "u_asym" @udp_dual_edge_mux(clk: %clk: i1, da: %da: i1, db: %db: i1) -> (q: i1)
  // CHECK: hw.output [[U_ASYM]]
  udp_dual_edge_mux u_asym (q, clk, da, db);
endmodule

// CHECK-LABEL: hw.module @TestSkipAsyncRowDFF(
module TestSkipAsyncRowDFF(input logic clk, input logic d, output logic q);
  // CHECK: [[U_DFF:%.+]] = hw.instance "u_dff" @udp_dff_skip_async(clk: %clk: !seq.clock, d: %d: i1) -> (q: i1)
  // CHECK: hw.output [[U_DFF]]
  udp_dff_skip_async u_dff (q, clk, d);
endmodule
