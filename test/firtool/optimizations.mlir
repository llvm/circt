// RUN: firtool %s -verilog | FileCheck %s
// Issue 2393: Check that if statements created by canonicalizer are merged.
// CHECK-LABEL: module Issue2393(
hw.module @Issue2393(%clock: i1, %c: i1, %data: i2) {
  %r1 = sv.reg : !hw.inout<i2>
  %1 = sv.read_inout %r1 : !hw.inout<i2>
  %r2 = sv.reg : !hw.inout<i2>
  %2 = sv.read_inout %r2 : !hw.inout<i2>
  %mux1 = comb.mux %c, %data, %1 : i2
  %mux2 = comb.mux %c, %data, %2 : i2
  sv.always posedge %clock {
    sv.passign %r1, %mux1 : i2
    sv.passign %r2, %mux2 : i2
  }
  // CHECK:     always @(posedge clock) begin
  // CHECK-NEXT:   if (c) begin
  // CHECK-NEXT:     r1 <= data;
  // CHECK-NEXT:     r2 <= data;
  // CHECK-NEXT:   end
  // CHECK-NEXT: end
}