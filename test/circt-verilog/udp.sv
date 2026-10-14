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

