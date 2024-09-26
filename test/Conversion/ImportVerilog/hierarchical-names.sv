// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @Foo()
module Foo;
  // CHECK: %subA.subB.y, %subA.subB.x = moore.instance "subA" @SubA() -> (subB.y: !moore.ref<i32>, subB.x: !moore.ref<i32>)
  SubA subA();
  // CHECK: [[RD_SA_SB_Y:%.+]] = moore.read %subA.subB.y : <i32>
  int s = subA.subB.y;
  int r;
  // CHECK: [[RD_R:%.+]] = moore.read %r : <i32>
  // CHECK: moore.assign %subA.subB.x, [[RD_R]] : i32
  assign subA.subB.x = r;
endmodule

// CHECK-LABEL: moore.module private @SubA(out subB.y : !moore.ref<i32>, out subB.x : !moore.ref<i32>)
module SubA;
  int a;
  // CHECK: %subB.y, %subB.x = moore.instance "subB" @SubB() -> (y: !moore.ref<i32>, x: !moore.ref<i32>)
  SubB subB();
  // CHECK: [[RD_SB_Y:%.+]] = moore.read %subB.y : <i32>
  // CHECK: moore.assign %a, [[RD_SB_Y]] : i32
  assign a = subB.y;
endmodule

// CHECK-LABEL: moore.module private @SubB(out y : !moore.ref<i32>, out x : !moore.ref<i32>)
module SubB;
  int x, y;
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

// CHECK-LABEL: moore.module private @SubC(out subD.z : !moore.ref<i32>)
module SubC;
  // CHECK: %subD.z = moore.instance "subD" @SubD() -> (z: !moore.ref<i32>)
  SubD subD();
endmodule

// CHECK-LABEL: moore.module private @SubD(out z : !moore.ref<i32>)
module SubD;
  int z;
  int w;
  // CHECK: [[RD_Z:%.+]] = moore.read %z : <i32>
  // CHECK: moore.assign %w, [[RD_Z]] : i32
  assign SubD.w = SubD.z;
endmodule
