// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @Foo(in %subA.subB.y : !moore.ref<i32>, in %subA.subB.x : !moore.ref<i32>)
module Foo;
  // CHECK: %0 = moore.read %subA.subB.y : <i32>
  // CHECK: %s = moore.variable %0 : <i32>
  int s = subA.subB.y;
  // CHECK: %r = moore.variable : <i32>
  int r;
  // CHECK: %1 = moore.read %r : <i32>
  // CHECK: moore.assign %subA.subB.x, %1 : i32
  assign subA.subB.x = r;
  // CHECK: moore.instance "subA" @SubA(subB.y: %subA.subB.y: !moore.ref<i32>, subB.x: %subA.subB.x: !moore.ref<i32>) -> ()
  SubA subA();
endmodule

// CHECK-LABEL: moore.module private @SubA(in %subB.y : !moore.ref<i32>, in %subB.x : !moore.ref<i32>)
module SubA;
  // CHECK: %a = moore.variable : <i32>
  int a;
  // CHECK: moore.instance "subB" @SubB(y: %subB.y: !moore.ref<i32>, x: %subB.x: !moore.ref<i32>) -> ()
  SubB subB();
  // CHECK: %0 = moore.read %subB.y : <i32>
  // CHECK: moore.assign %a, %0 : i32
  assign a = subB.y;
endmodule

// CHECK-LABEL: moore.module private @SubB(in %y : !moore.ref<i32>, in %x : !moore.ref<i32>)
module SubB;
  // CHECK-NOT: %x_0 = moore.variable
  // CHECK-NOT: %y_1 = moore.variable
  int x, y;
  // CHECK: moore.output
endmodule

// CHECK-LABEL: moore.module @Bar(in %subC1.subD.z : !moore.ref<i32>, in %subC2.subD.z : !moore.ref<i32>)
module Bar;
  // CHECK: moore.instance "subC1" @SubC(subD.z: %subC1.subD.z: !moore.ref<i32>) -> ()
  SubC subC1();
  // CHECK: moore.instance "subC2" @SubC(subD.z: %subC2.subD.z: !moore.ref<i32>) -> ()
  SubC subC2();
  // CHECK: %0 = moore.read %subC1.subD.z : <i32>
  // CHECK: %u = moore.variable %0 : <i32>
  int u = subC1.subD.z;
  // CHECK: %1 = moore.read %u : <i32>
  // CHECK: moore.assign %subC2.subD.z, %1 : i32
  assign subC2.subD.z = u;
endmodule

// CHECK-LABEL: moore.module private @SubC(in %subD.z : !moore.ref<i32>)
module SubC;
  // CHECK: moore.instance "subD" @SubD(z: %subD.z: !moore.ref<i32>) -> ()
  SubD subD();
endmodule

// CHECK-LABEL: moore.module private @SubD(in %z : !moore.ref<i32>)
module SubD;
  // CHECK-NOT: %z_0 = moore.variable
  int z;
  // CHECK: %w = moore.variable : <i32>
  int w;
  // CHECK: %0 = moore.read %z : <i32>
  // CHECK: moore.assign %w, %0 : i32
  assign SubD.w = SubD.z;
endmodule
