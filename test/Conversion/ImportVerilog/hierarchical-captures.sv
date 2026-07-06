// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module private @SiblingLeaf(out val : !moore.ref<i32>)
module SiblingLeaf;
  int val = 42;
endmodule

// CHECK-LABEL: moore.module @SiblingCaptures()
module SiblingCaptures;
  // CHECK-DAG: %u1.val = moore.instance "u1" @SiblingLeaf() -> (val: !moore.ref<i32>)
  // CHECK-DAG: %u2.val = moore.instance "u2" @SiblingLeaf() -> (val: !moore.ref<i32>)
  SiblingLeaf u1();
  SiblingLeaf u2();

  function automatic int getU1();
    return u1.val;
  endfunction

  function automatic int getU2();
    return u2.val;
  endfunction

  // CHECK: func.call @getU1(%u1.val)
  // CHECK: func.call @getU2(%u2.val)
  initial $display("%0d %0d", getU1(), getU2());
endmodule

// CHECK-LABEL: func.func private @getU1(%arg0: !moore.ref<i32>) -> !moore.i32
// CHECK: moore.read %arg0
// CHECK-LABEL: func.func private @getU2(%arg0: !moore.ref<i32>) -> !moore.i32
// CHECK: moore.read %arg0

// CHECK-LABEL: moore.module private @DeepLeaf(out val : !moore.ref<i32>)
module DeepLeaf;
  int val = 7;
endmodule

// CHECK-LABEL: moore.module private @DeepMid(out u.val : !moore.ref<i32>)
module DeepMid;
  // CHECK: %u.val = moore.instance "u" @DeepLeaf() -> (val: !moore.ref<i32>)
  // CHECK: moore.output %u.val : !moore.ref<i32>
  DeepLeaf u();
endmodule

// CHECK-LABEL: moore.module @DeepCaptures()
module DeepCaptures;
  // CHECK-DAG: %m1.u.val = moore.instance "m1" @DeepMid() -> (u.val: !moore.ref<i32>)
  // CHECK-DAG: %m2.u.val = moore.instance "m2" @DeepMid() -> (u.val: !moore.ref<i32>)
  DeepMid m1();
  DeepMid m2();

  function automatic int getM2();
    return m2.u.val;
  endfunction

  function automatic int getM1();
    return m1.u.val;
  endfunction

  // CHECK: func.call @getM2(%m2.u.val)
  // CHECK: func.call @getM1(%m1.u.val)
  initial $display("%0d %0d", getM2(), getM1());
endmodule

// CHECK-LABEL: func.func private @getM2(%arg0: !moore.ref<i32>) -> !moore.i32
// CHECK-LABEL: func.func private @getM1(%arg0: !moore.ref<i32>) -> !moore.i32

// CHECK-LABEL: moore.module private @NestedLeaf(out val : !moore.ref<i32>)
module NestedLeaf;
  int val = 5;
endmodule

// CHECK-LABEL: moore.module private @NestedMid(out u.val : !moore.ref<i32>)
module NestedMid;
  NestedLeaf u();
endmodule

// CHECK-LABEL: moore.module @NestedSecondOnly()
module NestedSecondOnly;
  // CHECK-DAG: %m1.u.val = moore.instance "m1" @NestedMid() -> (u.val: !moore.ref<i32>)
  // CHECK-DAG: %m2.u.val = moore.instance "m2" @NestedMid() -> (u.val: !moore.ref<i32>)
  NestedMid m1();
  NestedMid m2();
  // CHECK: moore.read %m2.u.val
  initial $display("%0d", m2.u.val);
endmodule
