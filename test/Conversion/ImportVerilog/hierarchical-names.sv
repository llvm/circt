// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @Foo()
module Foo;
  int r;
  // CHECK: %subA.subB.y, %subA.subB.x = moore.instance "subA" @SubA(Foo.r: %r: !moore.ref<i32>) -> (subB.y: !moore.ref<i32>, subB.x: !moore.ref<i32>)
  SubA subA();
  // CHECK: [[RD_SA_SB_Y:%.+]] = moore.read %subA.subB.y : <i32>
  int s = subA.subB.y;
  // CHECK: [[RD_R:%.+]] = moore.read %r : <i32>
  // CHECK: moore.assign %subA.subB.x, [[RD_R]] : i32
  assign subA.subB.x = r;
endmodule

// CHECK-LABEL: moore.module private @SubA(in %Foo.r : !moore.ref<i32>, out subB.y : !moore.ref<i32>, out subB.x : !moore.ref<i32>)
module SubA;
  int a;
  // CHECK: %subB.y, %subB.x = moore.instance "subB" @SubB(Foo.r: %Foo.r: !moore.ref<i32>, subA.a: %a: !moore.ref<i32>) -> (y: !moore.ref<i32>, x: !moore.ref<i32>)
  SubB subB();
  // CHECK: [[RD_SB_Y:%.+]] = moore.read %subB.y : <i32>
  // CHECK: moore.assign %a, [[RD_SB_Y]] : i32
  assign a = subB.y;
  // CHECK: moore.output %subB.y, %subB.x : !moore.ref<i32>, !moore.ref<i32>
endmodule

// CHECK-LABEL: moore.module private @SubB(in %Foo.r : !moore.ref<i32>, in %subA.a : !moore.ref<i32>, out y : !moore.ref<i32>, out x : !moore.ref<i32>)
module SubB;
  int x, y, z;
  // CHECK: [[RD_FOO_R:%.+]] = moore.read %Foo.r : <i32>
  // CHECK: moore.assign %y, [[RD_FOO_R]] : i32
  // CHECK: [[RD_SA_A:%.+]] = moore.read %subA.a : <i32>
  // CHECK: moore.assign %z, [[RD_SA_A]] : i32
  assign y = Foo.r;
  assign z = Foo.subA.a;
  // CHECK: moore.output %y, %x : !moore.ref<i32>, !moore.ref<i32>
endmodule

// -----

// CHECK-LABEL: moore.module @Bar(in %a : !moore.l1, in %b : !moore.l1, out c : !moore.l1)
module Bar(input a, b,
           output c);
  // CHECK: %subC1.c, %subC1.subD.z = moore.instance "subC1" @SubC(a: %0: !moore.l1, b: %1: !moore.l1) -> (c: !moore.l1, subD.z: !moore.ref<i32>)
  SubC subC1(a, b, c);
  // CHECK: %subC2.c, %subC2.subD.z = moore.instance "subC2" @SubC(a: %2: !moore.l1, b: %3: !moore.l1) -> (c: !moore.l1, subD.z: !moore.ref<i32>)
  SubC subC2(a, b, c);
  // CHECK: [[RD_SC1_SD_Z:%.+]] = moore.read %subC1.subD.z : <i32>
  // CHECK: moore.variable [[RD_SC1_SD_Z]] : <i32>
  int u = subC1.subD.z;
endmodule

// CHECK-LABEL: moore.module private @SubC(in %a : !moore.l1, in %b : !moore.l1, out c : !moore.l1, out subD.z : !moore.ref<i32>)
module SubC(input a, b,
            output c);
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
