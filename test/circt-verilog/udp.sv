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
  // CHECK-DAG: [[DELAYED:%.+]] = llhd.delay [[U_MUX_DELAY:%.+]] by <5000000fs, 0d> : i1
  // CHECK-DAG: [[U_MUX_DELAY]] = hw.instance "u_mux_delay" @udp_mux(sel: %sel: i1, a: %a: i1, b: %b: i1) -> (out: i1)
  // CHECK: hw.output [[DELAYED]]
  udp_mux #5 u_mux_delay (out, sel, a, b);
endmodule


//===----------------------------------------------------------------------===//
// Sequential UDP Primitives
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Sequential UDP Instantiations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @TestSeqUdp(
module TestSeqUdp(input logic en, input logic d, output logic q);
  // CHECK: [[U_LATCH:%.+]] = hw.instance "u_latch" @udp_latch(en: %en: i1, d: %d: i1) -> (q: i1)
  // CHECK: hw.output [[U_LATCH]]
  udp_latch u_latch (q, en, d);
endmodule
