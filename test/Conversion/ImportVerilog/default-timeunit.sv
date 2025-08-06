// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: @DefaultTimeUnit1ns1ns
module DefaultTimeUnit1ns1ns;
  // CHECK: moore.constant_time 12000000 fs
  time a0 = 12ns;
  // CHECK: moore.constant_time 2000000 fs
  time a1 = 2.345ns;
  // CHECK: moore.constant_time 3000000 fs
  time a2 = 3456ps;
  // CHECK: moore.constant_time 45000000 fs
  realtime b0 = 45ns;
  // CHECK: moore.constant_time 6000000 fs
  realtime b1 = 5.678ns;
  // CHECK: moore.constant_time 7000000 fs
  realtime b2 = 6789ps;
endmodule

