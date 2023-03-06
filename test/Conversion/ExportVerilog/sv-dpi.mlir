// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK:      import "DPI-C" function void test(
// CHECK-NEXT:   input  [31:0] arg0,
// CHECK-NEXT:   input  [63:0] arg1,
// CHECK-NEXT:   output [4:0] res0,
// CHECK-NEXT:   output [5:0] res1
// CHECK-NEXT: );
sv.dpi.import @test(%arg0: i32, %arg1: i64) -> (res0: i5, res1: i6)

hw.module @top(%clk: i1, %arg1: i64) -> () {
  %arg0 = hw.constant 0 : i32
  %fd = hw.constant 0x80000002 : i32
  // CHECK: reg [4:0] [[TMP_0:.+]];
  // CHECK: reg [5:0] [[TMP_1:.+]];
  sv.alwaysff(posedge %clk) {
    // CHECK: test(32'h0, arg1, [[TMP_0]], [[TMP_1]]);
    // CHECK: $fwrite(32'h80000002, "%d %d\n", [[TMP_0]], [[TMP_1]]);
    %res0, %res1 = sv.dpi.call @test(%arg0, %arg1) : (i32, i64) -> (i5, i6)
    sv.fwrite %fd, "%d %d\n"(%res0, %res1) : i5, i6
  }
}
