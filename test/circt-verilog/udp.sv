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

// Test standard positive-edge triggered D flip-flop lowering.
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

// Test negative-edge triggered D flip-flop lowering.
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

// Test clock enable gating extraction on synchronous inputs.
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

// Test dual-edge triggered flip-flop lowering to posedge and negedge registers with clock mux.
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
    (0x) 0 : 1 : 0 ; // Skipped: NotTwoState edge (must not set usesState)
  endtable
endprimitive

// Test dual-edge multiplexer with distinct data inputs per active edge.
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

// Test asymmetric next-state boolean logic with don't-care condition synthesis.
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

// Test single asynchronous active-high reset lowering.
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

// Test dual asynchronous set/reset priority lowering.
// CHECK-LABEL: hw.module @udp_dff_async_sr(out q : i1, in %clk : !seq.clock, in %rst : i1, in %set : i1, in %d : i1)
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[NOT_RST:%.+]] = comb.xor %rst, [[TRUE]] : i1
// CHECK-DAG: [[SET_TERM:%.+]] = comb.and [[NOT_RST]], %set : i1
// CHECK-DAG: [[ACOND:%.+]] = comb.or %rst, [[SET_TERM]] : i1
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk reset async [[ACOND]], [[SET_TERM]] preset 0 : i1
// CHECK: hw.output [[REG]] : i1
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

// Test active-low clear and preset lowering.
// CHECK-LABEL: hw.module @udp_dff_async_clear_preset(out q : i1, in %clk : !seq.clock, in %clr_n : i1, in %pre_n : i1, in %d : i1)
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[NOT_CLR_N:%.+]] = comb.xor %clr_n, [[TRUE]] : i1
// CHECK-DAG: [[NOT_PRE_N:%.+]] = comb.xor %pre_n, [[TRUE]] : i1
// CHECK-DAG: [[SET_TERM:%.+]] = comb.and %clr_n, [[NOT_PRE_N]] : i1
// CHECK-DAG: [[ACOND:%.+]] = comb.or [[NOT_CLR_N]], [[SET_TERM]] : i1
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk reset async [[ACOND]], [[SET_TERM]] preset 0 : i1
// CHECK: hw.output [[REG]] : i1
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

// Verify that active-low asynchronous reset recovers seq.firreg reset async end-to-end.
// CHECK-LABEL: hw.module @udp_dff_async_rst_n(
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[FALSE:%.+]] = hw.constant false
// CHECK-DAG: [[NOT_RST:%.+]] = comb.xor %rst_n, [[TRUE]] : i1
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk reset async [[NOT_RST]], [[FALSE]] preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_async_rst_n(q, clk, rst_n, d);
  output reg q;
  input clk, rst_n, d;
  table
    ?    0 ? : ? : 0 ;
    (01) 1 0 : ? : 0 ;
    (01) 1 1 : ? : 1 ;
    (10) ? ? : ? : - ;
  endtable
endprimitive

// Verify that initial state specification (initial q = 1) lowers to seq.firreg preset 1 end-to-end.
// CHECK-LABEL: hw.module @udp_dff_init(
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk preset 1 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_init(q, clk, d);
  output reg q;
  input clk, d;
  initial q = 1;
  table
    (01) 0 : ? : 0 ;
    (01) 1 : ? : 1 ;
    (10) ? : ? : - ;
  endtable
endprimitive

// Verify that dual-edge flip-flop with initial q = 1 initializes both edge registers.
// CHECK-LABEL: hw.module @udp_dual_edge_init(
// CHECK-DAG: [[Q_POS:%.+]] = seq.firreg %d clock %clk preset 1 : i1
// CHECK-DAG: [[Q_NEG:%.+]] = seq.firreg %d clock {{%.+}} preset 1 : i1
// CHECK: hw.output
primitive udp_dual_edge_init(q, clk, d);
  output reg q;
  input clk, d;
  initial q = 1;
  table
    (01) 0 : ? : 0 ;
    (01) 1 : ? : 1 ;
    (10) 0 : ? : 0 ;
    (10) 1 : ? : 1 ;
  endtable
endprimitive

// Verify that dual-edge flip-flop with clock enable hold rows lowers to cross-register muxes end-to-end.
// CHECK-LABEL: hw.module @udp_dual_edge_en(out q : i1, in %clk : !seq.clock, in %en : i1, in %d : i1)
// CHECK-DAG: [[CLK_BOOL:%.+]] = seq.from_clock %clk
// CHECK-DAG: [[MUX_POS:%.+]] = comb.mux %en, %d, [[Q_NEG:%.+]] : i1
// CHECK-DAG: [[Q_POS:%.+]] = seq.firreg [[MUX_POS]] clock %clk preset 0 : i1
// CHECK-DAG: [[MUX_NEG:%.+]] = comb.mux %en, %d, [[Q_POS]] : i1
// CHECK-DAG: [[CLK_INV:%.+]] = seq.clock_inv %clk
// CHECK-DAG: [[Q_NEG]] = seq.firreg [[MUX_NEG]] clock [[CLK_INV]] preset 0 : i1
// CHECK-DAG: [[TERM_POS:%.+]] = comb.and [[CLK_BOOL]], [[Q_POS]] : i1
// CHECK-DAG: [[TRUE:%.+]] = hw.constant true
// CHECK-DAG: [[NOT_CLK:%.+]] = comb.xor [[CLK_BOOL]], [[TRUE]] : i1
// CHECK-DAG: [[TERM_NEG:%.+]] = comb.and [[NOT_CLK]], [[Q_NEG]] : i1
// CHECK: [[Q_OUT:%.+]] = comb.or [[TERM_POS]], [[TERM_NEG]] : i1
// CHECK: hw.output [[Q_OUT]] : i1
primitive udp_dual_edge_en(q, clk, en, d);
  output reg q;
  input clk, en, d;
  table
    (01) 1 0 : ? : 0 ;
    (01) 1 1 : ? : 1 ;
    (10) 1 0 : ? : 0 ;
    (10) 1 1 : ? : 1 ;
    (01) 0 ? : ? : - ;
    (10) 0 ? : ? : - ;
  endtable
endprimitive

// Verify that composite reset condition recovers seq.firreg reset async end-to-end.
// CHECK-LABEL: hw.module @udp_dff_or_rst(
// CHECK-DAG: [[COMB_RST:%.+]] = comb.or %rst1, %rst2 : i1
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk reset async [[COMB_RST]], {{%.+}} preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_dff_or_rst(q, clk, rst1, rst2, d);
  output reg q;
  input clk, rst1, rst2, d;
  table
    ?    1 ? ? : ? : 0 ;
    ?    ? 1 ? : ? : 0 ;
    (01) 0 0 0 : ? : 0 ;
    (01) 0 0 1 : ? : 1 ;
    (10) ? ? ? : ? : - ;
  endtable
endprimitive

// Verify that sequential UDP with internal state dependency (JK flip-flop) lowers to seq.firreg with feedback end-to-end.
// CHECK-LABEL: hw.module @udp_jkff(
// CHECK: [[REG:%.+]] = seq.firreg {{%.+}} clock %clk preset 0 : i1
// CHECK: hw.output [[REG]]
primitive udp_jkff(q, clk, j, k);
  output reg q;
  input clk, j, k;
  table
    (01) 0 0 : ? : - ;
    (01) 0 1 : ? : 0 ;
    (01) 1 0 : ? : 1 ;
    (01) 1 1 : 0 : 1 ;
    (01) 1 1 : 1 : 0 ;
    (10) ? ? : ? : - ;
  endtable
endprimitive

// Verify that level-sensitive latch lowers to llhd.sig and llhd.assign if %en end-to-end.
// CHECK-LABEL: hw.module @udp_latch(
// CHECK: [[SIG:%.+]] = llhd.sig {{%.+}} : <i1>
// CHECK: llhd.assign [[SIG]], {{%.+}} if {{%.+}} : i1
// CHECK: hw.output
primitive udp_latch(q, en, d);
  output reg q;
  input en, d;
  table
    1 0 : ? : 0 ;
    1 1 : ? : 1 ;
    0 ? : ? : - ;
  endtable
endprimitive

// Verify that cell-library async reset enumerated per clock level lowers cleanly end-to-end.
// CHECK-LABEL: hw.module @udp_dff_split_clk_rst(out q : i1, in %clk : !seq.clock, in %rst : i1, in %d : i1)
// CHECK-DAG: [[FALSE:%.+]] = hw.constant false
// CHECK: [[REG:%.+]] = seq.firreg %d clock %clk reset async %rst, [[FALSE]] preset 0 : i1
// CHECK: hw.output [[REG]] : i1
primitive udp_dff_split_clk_rst(q, clk, rst, d);
  output reg q;
  input clk, rst, d;
  table
    0 1 ? : ? : 0 ;
    1 1 ? : ? : 0 ;
    (01) 0 0 : ? : 0 ;
    (01) 0 1 : ? : 1 ;
    (10) ? ? : ? : - ;
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
