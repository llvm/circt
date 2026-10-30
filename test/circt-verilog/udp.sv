// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Combinational UDP
//===----------------------------------------------------------------------===//

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
  // CHECK-DAG: [[DELAYED:%.+]] = llhd.delay [[U_MUX_DELAY:%.+]] by <5000000fs, 0d> : i1
  // CHECK-DAG: [[U_MUX_DELAY]] = hw.instance "u_mux_delay" @udp_mux(sel: %sel: i1, a: %a: i1, b: %b: i1) -> (out: i1)
  // CHECK: hw.output [[DELAYED]]
  udp_mux #5 u_mux_delay (out, sel, a, b);
endmodule

//===----------------------------------------------------------------------===//
// Sequential UDP Primitives
//===----------------------------------------------------------------------===//

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

// Verify that benign hold rows and inactive clock edges in an edge-triggered
// UDP are safely skipped.
// CHECK-LABEL: hw.module @udp_dff_hold(
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_hold(q, clk, d);
  output reg q;
  input clk, d;
  table
    (01) 0 : ? : 0 ;
    (01) 1 : ? : 1 ;
    (10) ? : ? : - ; // Skipped: inactive clock edge (output is '-')
    ?    0 : 0 : 0 ; // Skipped: steady-state hold (state == output)
    ?    1 : 1 : 1 ; // Skipped: steady-state hold (state == output)
  endtable
endprimitive

// CHECK-LABEL: hw.module @udp_dual_edge_dff(out q : i1, in %clk : !seq.clock, in %d : i1)
// CHECK-DAG: [[CLK_BOOL:%.+]] = seq.from_clock %clk
// CHECK-DAG: [[Q_POS:%.+]] = seq.firreg %d clock %clk preset 0 : i1
// CHECK-DAG: [[CLK_INV:%.+]] = seq.clock_inv %clk
// CHECK-DAG: [[Q_NEG:%.+]] = seq.firreg %d clock [[CLK_INV]] preset 0 : i1
// CHECK-DAG: [[TERM_POS:%.+]] = comb.and [[CLK_BOOL]], [[Q_POS]] : i1
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[NOT_CLK:%.+]] = comb.xor [[CLK_BOOL]], [[TRUE]] : i1
// CHECK-DAG: [[TERM_NEG:%.+]] = comb.and [[NOT_CLK]], [[Q_NEG]] : i1
// CHECK: [[Q_OUT:%.+]] = comb.or [[TERM_POS]], [[TERM_NEG]] : i1
// CHECK: hw.output [[Q_OUT]] : i1
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

// CHECK-LABEL: hw.module @udp_dual_edge_mux(out q : i1, in %clk : !seq.clock, in %da : i1, in %db : i1)
// CHECK-DAG: [[CLK_BOOL:%.+]] = seq.from_clock %clk
// CHECK-DAG: [[Q_POS:%.+]] = seq.firreg %da clock %clk preset 0 : i1
// CHECK-DAG: [[CLK_INV:%.+]] = seq.clock_inv %clk
// CHECK-DAG: [[Q_NEG:%.+]] = seq.firreg %db clock [[CLK_INV]] preset 0 : i1
// CHECK-DAG: [[TERM_POS:%.+]] = comb.and [[CLK_BOOL]], [[Q_POS]] : i1
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[NOT_CLK:%.+]] = comb.xor [[CLK_BOOL]], [[TRUE]] : i1
// CHECK-DAG: [[TERM_NEG:%.+]] = comb.and [[NOT_CLK]], [[Q_NEG]] : i1
// CHECK: [[Q_OUT:%.+]] = comb.or [[TERM_POS]], [[TERM_NEG]] : i1
// CHECK: hw.output [[Q_OUT]] : i1
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

// CHECK-LABEL: hw.module @udp_dff_asym_cond(out q : i1, in %clk : !seq.clock, in %clr_n : i1, in %pre_n : i1, in %d : i1, in %n : i1)
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[SET_TERM:%.+]] = comb.and %clr_n, %d : i1
// CHECK-DAG: [[OR_HOLD:%.+]] = comb.or [[SET_TERM]], [[REG:%.+]] : i1
// CHECK-DAG: [[NOT_D:%.+]] = comb.xor %d, [[TRUE]] : i1
// CHECK-DAG: [[CLR_TERM:%.+]] = comb.and %pre_n, [[NOT_D]] : i1
// CHECK-DAG: [[NOT_CLR:%.+]] = comb.xor [[CLR_TERM]], [[TRUE]] : i1
// CHECK-DAG: [[NEXT:%.+]] = comb.and [[NOT_CLR]], [[OR_HOLD]] : i1
// CHECK: [[REG]] = seq.firreg [[NEXT]] clock %clk preset 0 : i1
// CHECK: hw.output [[REG]] : i1
primitive udp_dff_asym_cond(q, clk, clr_n, pre_n, d, n);
  output reg q;
  input clk, clr_n, pre_n, d, n;
  table
    (01) ? 1 0 ? : ? : 0 ;
    (01) 1 ? 1 ? : ? : 1 ;
  endtable
endprimitive


// CHECK-LABEL: hw.module @udp_dff_async_rst(
// CHECK-DAG: [[FALSE:%.+]] = hw.constant false
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk reset async %rst, [[FALSE]] preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_async_rst(q, clk, rst, d);
  output reg q;
  input clk, rst, d;
  table
    ?    1 ? : ? : 0 ;
    (01) 0 0 : ? : 0 ;
    (01) 0 1 : ? : 1 ;
    (10) ? ? : ? : - ;
    ?   (10) ? : ? : - ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @udp_dff_async_sr(
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[NOT_RST:%.+]] = comb.xor %rst, [[TRUE]] : i1
// CHECK-DAG: [[SET_TERM:%.+]] = comb.and [[NOT_RST]], %set : i1
// CHECK-DAG: [[OR_D:%.+]] = comb.or [[SET_TERM]], %d : i1
// CHECK-DAG: [[NEXT:%.+]] = comb.and [[NOT_RST]], [[OR_D]] : i1
// CHECK: [[REG:%.+]] = seq.firreg [[NEXT]] clock %clk preset 0 : i1
// CHECK-DAG: [[OR_REG:%.+]] = comb.or [[SET_TERM]], [[REG]] : i1
// CHECK-DAG: [[OUT:%.+]] = comb.and [[NOT_RST]], [[OR_REG]] : i1
// CHECK: hw.output [[OUT]]
primitive udp_dff_async_sr(q, clk, rst, set, d);
  output reg q;
  input clk, rst, set, d;
  table
    ?    1 ? ? : ? : 0 ;
    ?    0 1 ? : ? : 1 ;
    (01) 0 0 0 : ? : 0 ;
    (01) 0 0 1 : ? : 1 ;
    (10) ? ? ? : ? : - ;
  endtable
endprimitive

// CHECK-LABEL: hw.module @udp_dff_async_clear_preset(out q : i1, in %clk : !seq.clock, in %clr_n : i1, in %pre_n : i1, in %d : i1)
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[NOT_PRE_N:%.+]] = comb.xor %pre_n, [[TRUE]] : i1
// CHECK-DAG: [[SET_TERM:%.+]] = comb.and %clr_n, [[NOT_PRE_N]] : i1
// CHECK-DAG: [[OR_D:%.+]] = comb.or [[SET_TERM]], %d : i1
// CHECK-DAG: [[NEXT:%.+]] = comb.and %clr_n, [[OR_D]] : i1
// CHECK: [[REG:%.+]] = seq.firreg [[NEXT]] clock %clk preset 0 : i1
// CHECK-DAG: [[OR_REG:%.+]] = comb.or [[SET_TERM]], [[REG]] : i1
// CHECK-DAG: [[OUT:%.+]] = comb.and %clr_n, [[OR_REG]] : i1
// CHECK: hw.output [[OUT]] : i1
primitive udp_dff_async_clear_preset(q, clk, clr_n, pre_n, d);
  output reg q;
  input clk, clr_n, pre_n, d;
  table
    ?    0 ? ? : ? : 0 ;
    ?    1 0 ? : ? : 1 ;
    (01) 1 1 0 : ? : 0 ;
    (01) 1 1 1 : ? : 1 ;
    (10) ? ? ? : ? : - ; // Skipped: inactive clock edge (output is '-')
    0    1 1 ? : 0 : 0 ; // Skipped: steady-clock hold row (state == output)
    0    1 1 ? : 1 : 1 ; // Skipped: steady-clock hold row (state == output)
  endtable
endprimitive


//===----------------------------------------------------------------------===//
// Sequential UDP Instantiations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @TestSeqUdp(
module TestSeqUdp(input logic clk, input logic d, output logic q);
  // CHECK: [[U_DFF:%.+]] = hw.instance "u_dff" @udp_dff(clk: %clk: !seq.clock, d: %d: i1) -> (q: i1)
  // CHECK: hw.output [[U_DFF]]
  udp_dff u_dff (q, clk, d);
endmodule
