// RUN: not circt-verilog --ir-moore %s 2>&1 | FileCheck %s --check-prefix=NOALLOW
// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// NOALLOW: identifier 'a' used before its declaration

// CHECK-LABEL: moore.module @ForwardNamedGenerateStorage(
module ForwardNamedGenerateStorage(output int out);
  if (1) begin : g
    // CHECK-DAG: %g.a = moore.variable : <i32>
    // CHECK-DAG: %g.b = moore.variable {{.*}} : <i32>
    // CHECK: moore.procedure initial {
    // CHECK:   moore.read %g.b : <i32>
    // CHECK:   moore.blocking_assign %g.a, {{.*}} : i32
    // CHECK: }
    initial a = b;

    int a;
    int b = 7;

    // CHECK: moore.read %g.a : <i32>
    // CHECK: moore.output {{.*}} : !moore.i32
    assign out = a;
  end
endmodule

// CHECK-LABEL: moore.module @ForwardGenerateArrayVariables(
module ForwardGenerateArrayVariables(output logic [1:0] out);
  for (genvar i = 0; i < 2; ++i) begin : lane
    // CHECK-DAG: %lane_0.bit_out = moore.variable : <i1>
    // CHECK-DAG: %lane_1.bit_out = moore.variable : <i1>
    // CHECK: moore.blocking_assign %lane_0.bit_out
    // CHECK: moore.assign {{.*}} : l1
    // CHECK: moore.blocking_assign %lane_1.bit_out
    // CHECK: moore.assign {{.*}} : l1
    initial bit_out = bit'(i);

    bit bit_out;
    assign out[i] = bit_out;
  end
endmodule
