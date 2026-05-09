// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

interface output_if
    (input logic clk);
  logic [7 : 0] data;
endinterface

// CHECK-LABEL: moore.module @top
// CHECK: [[VIFS:%.*]] = moore.variable : <uarray<2 x ustruct<
// CHECK-SAME: clk:
// CHECK-SAME: data:
// CHECK: [[TMP:%.*]] = moore.variable : <ustruct<
// CHECK-SAME: clk:
// CHECK-SAME: data:
module top;
  logic clk0, clk1;
  output_if out0(clk0);
  output_if out1(clk1);

  virtual output_if vifs[2];
  virtual output_if tmp;
  logic [7 : 0] s;

  initial begin
    // CHECK: [[VIFS0_REF:%.*]] = moore.extract_ref [[VIFS]] from 1
    // CHECK: [[OUT0:%.*]] = moore.struct_create %clk0, %out0_data
    // CHECK: moore.blocking_assign [[VIFS0_REF]], [[OUT0]]
    vifs[0] = out0;
    // CHECK: [[VIFS1_REF:%.*]] = moore.extract_ref [[VIFS]] from 0
    // CHECK: [[OUT1:%.*]] = moore.struct_create %clk1, %out1_data
    // CHECK: moore.blocking_assign [[VIFS1_REF]], [[OUT1]]
    vifs[1] = out1;

    // CHECK: [[VIFS_VAL:%.*]] = moore.read [[VIFS]] : <uarray<2 x ustruct<
    // CHECK: [[VIF1:%.*]] = moore.extract [[VIFS_VAL]] from 0
    // CHECK: moore.blocking_assign [[TMP]], [[VIF1]]
    tmp = vifs[1];

    // CHECK: [[TMP_R:%.*]] = moore.read [[TMP]]
    // CHECK: [[DATA_REF:%.*]] = moore.struct_extract [[TMP_R]], "data"
    // CHECK: [[LIT:%.*]] = moore.constant 34 : l8
    // CHECK: moore.blocking_assign [[DATA_REF]], [[LIT]]
    tmp.data = 8'h22;

    // CHECK: [[TMP_R2:%.*]] = moore.read [[TMP]]
    // CHECK: [[DATA_REF2:%.*]] = moore.struct_extract [[TMP_R2]], "data"
    // CHECK: [[DATA_VAL:%.*]] = moore.read [[DATA_REF2]]
    s = tmp.data;
  end
endmodule
