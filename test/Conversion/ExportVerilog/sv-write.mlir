// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

hw.module @top(in %clock : i1) {
  sv.alwaysff(posedge %clock) {
    %c0 = hw.constant 42 : i32
    %c1 = hw.constant 0x80000001 : i32
    %c2 = hw.constant 0x80000002 : i32

    // CHECK:      always_ff @(posedge clock) begin
    // CHECK-NEXT:   $write("stdout");
    sv.write "stdout"

    // CHECK-NEXT:   $write("%d", 32'h2A);
    sv.write "%d"(%c0) : i32

    // CHECK-NEXT:   $write("%d %d", 32'h80000001, 32'h80000002);
    sv.write "%d %d"(%c1, %c2) : i32, i32

    // CHECK-NEXT: end
  }
}
