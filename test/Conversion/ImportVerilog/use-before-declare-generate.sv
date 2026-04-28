// RUN: not circt-verilog --ir-moore %s 2>&1 | FileCheck %s --check-prefix=NOALLOW
// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// NOALLOW: identifier 'a' used before its declaration

// CHECK-LABEL: moore.module @ForwardNamedGenerateProcedural(
module ForwardNamedGenerateProcedural(output int out);
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

// CHECK-LABEL: moore.module @ForwardNamedGenerateInitializer(
module ForwardNamedGenerateInitializer(output int out);
  if (1) begin : g
    // CHECK-DAG: %g.a = moore.variable {{.*}} : <i32>
    // CHECK-DAG: %g.b = moore.variable {{.*}} : <i32>
    // CHECK: moore.read %g.b : <i32>
    // CHECK: moore.output {{.*}} : !moore.i32
    assign out = a;

    int a = b;
    int b = 11;
  end
endmodule

// CHECK-LABEL: moore.module @ForwardNestedGenerate(
module ForwardNestedGenerate(output int out);
  if (1) begin : outer
    if (1) begin : inner
      // CHECK-DAG: %outer.inner.lhs = moore.variable : <i32>
      // CHECK-DAG: %outer.inner.rhs = moore.variable {{.*}} : <i32>
      // CHECK: moore.procedure initial {
      // CHECK:   moore.read %outer.inner.rhs : <i32>
      // CHECK:   moore.blocking_assign %outer.inner.lhs, {{.*}} : i32
      // CHECK: }
      initial lhs = rhs;

      int lhs;
      int rhs = 13;

      // CHECK: moore.read %outer.inner.lhs : <i32>
      assign out = lhs;
    end
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

// CHECK-LABEL: moore.module @ForwardGenerateNetAssignments(
module ForwardGenerateNetAssignments(output logic out);
  if (1) begin : g
    // CHECK: %g.tmp = moore.assigned_variable %g.late : l1
    // CHECK: %g.late = moore.assigned_variable
    // CHECK: moore.output %g.tmp : !moore.l1
    assign out = tmp;

    wire tmp = late;
    wire late = 1'b1;
  end
endmodule

// CHECK-LABEL: moore.module @ForwardCaseGenerate(
module ForwardCaseGenerate(output int out);
  parameter int Sel = 1;
  case (Sel)
    0: begin : arm0
      assign out = a;
      int a = 3;
    end
    default: begin : arm1
      // CHECK-DAG: %arm1.a = moore.variable : <i32>
      // CHECK-DAG: %arm1.b = moore.variable {{.*}} : <i32>
      // CHECK: moore.procedure initial {
      // CHECK:   moore.read %arm1.b : <i32>
      // CHECK:   moore.blocking_assign %arm1.a, {{.*}} : i32
      // CHECK: }
      initial a = b;

      int a;
      int b = 17;
      assign out = a;
    end
  endcase
endmodule

// CHECK-LABEL: moore.module @ForwardNestedLoopGenerate(
module ForwardNestedLoopGenerate(output logic [3:0] out);
  for (genvar i = 0; i < 2; ++i) begin : row
    for (genvar j = 0; j < 2; ++j) begin : col
      // CHECK-DAG: %row_0.col_0.bit_out = moore.variable : <i1>
      // CHECK-DAG: %row_0.col_1.bit_out = moore.variable : <i1>
      // CHECK-DAG: %row_1.col_0.bit_out = moore.variable : <i1>
      // CHECK-DAG: %row_1.col_1.bit_out = moore.variable : <i1>
      // CHECK: moore.blocking_assign %row_0.col_0.bit_out
      // CHECK: moore.blocking_assign %row_0.col_1.bit_out
      // CHECK: moore.blocking_assign %row_1.col_0.bit_out
      // CHECK: moore.blocking_assign %row_1.col_1.bit_out
      initial bit_out = bit'(i ^ j);

      bit bit_out;
      assign out[i * 2 + j] = bit_out;
    end
  end
endmodule
