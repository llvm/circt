// RUN: not circt-verilog --ir-moore %s 2>&1 | FileCheck %s --check-prefix=NOALLOW
// RUN: circt-verilog --ir-moore --allow-use-before-declare %s | FileCheck %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// NOALLOW: identifier 'data' used before its declaration

module Storage;
  int value;
endmodule

module Leaf(input int in, output int value);
  assign value = in;
endmodule

module ConstLeaf(output int value);
  assign value = 5;
endmodule

// CHECK-LABEL: moore.module private @Storage(
// CHECK-SAME: out value : !moore.ref<i32>
// CHECK-LABEL: moore.module private @Leaf(
// CHECK-SAME: in %in : !moore.i32
// CHECK-LABEL: moore.module private @ConstLeaf(
// CHECK-LABEL: moore.module @ForwardHierarchicalInstanceWrite(
module ForwardHierarchicalInstanceWrite(output int out);
  // CHECK-DAG: %data = moore.variable {{.*}} : <i32>
  // CHECK-DAG: %u.value = moore.instance "u" @Storage
  // CHECK: moore.procedure initial {
  // CHECK:   moore.read %data : <i32>
  // CHECK:   moore.blocking_assign %u.value, {{.*}} : i32
  // CHECK: }
  initial u.value = data;

  // CHECK: moore.read %u.value : <i32>
  assign out = u.value;

  Storage u();
  int data = 29;
endmodule

// CHECK-LABEL: moore.module @ForwardHierarchicalInstanceInput(
module ForwardHierarchicalInstanceInput(output int out);
  // CHECK-DAG: %src = moore.variable {{.*}} : <i32>
  // CHECK: moore.read %src : <i32>
  // CHECK: moore.instance "u" @Leaf
  // CHECK-SAME: in:
  // CHECK: moore.procedure initial {
  // CHECK:   moore.read %u.value_0 : <i32>
  // CHECK:   moore.blocking_assign %out, {{.*}} : i32
  // CHECK: }
  initial out = u.value;

  Leaf u(.in(src));
  int src = 31;
endmodule

// CHECK-LABEL: moore.module @ForwardGenerateInstance(
module ForwardGenerateInstance(output int out);
  if (1) begin : g
    // CHECK: %g.u.value, %g.u.value_0 = moore.instance "g.u" @ConstLeaf
    // CHECK: moore.procedure initial {
    // CHECK:   moore.read %g.u.value_0 : <i32>
    // CHECK:   moore.blocking_assign %out, {{.*}} : i32
    // CHECK: }
    initial out = u.value;

    ConstLeaf u();
  end
endmodule
