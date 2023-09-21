// RUN: circt-opt -mlir-print-debuginfo -mlir-print-local-scope -pass-pipeline='builtin.module(firrtl.circuit(firrtl-dedup))' %s | FileCheck %s

firrtl.circuit "Test" {
// CHECK-LABEL: @Dedup0()
firrtl.module private @Dedup0() {
  // CHECK: %w = firrtl.wire  : !firrtl.uint<1> loc(fused["foo", "bar"])
  %w = firrtl.wire : !firrtl.uint<1> loc("foo")
} loc("dedup0")
// CHECK: loc(fused["dedup0", "dedup1"])
// CHECK-NOT: @Dedup1()
firrtl.module private @Dedup1() {
  %w = firrtl.wire : !firrtl.uint<1> loc("bar")
} loc("dedup1")
firrtl.module @Test() {
  firrtl.instance dedup0 @Dedup0()
  firrtl.instance dedup1 @Dedup1()
}
}

// CHECK-LABEL: "PortLocations"
firrtl.circuit "PortLocations" {
// CHECK: firrtl.module private @PortLocs0(in %in: !firrtl.uint<1> loc(fused["1", "2"]))
firrtl.module private @PortLocs0(in %in : !firrtl.uint<1> loc("1")) { }
firrtl.module private @PortLocs1(in %in : !firrtl.uint<1> loc("2")) { }
firrtl.module @PortLocations() {
  firrtl.instance portLocs0 @PortLocs0(in in : !firrtl.uint<1>)
  firrtl.instance portLocs1 @PortLocs1(in in : !firrtl.uint<1>)
}
}

// Check that locations are limited.
// CHECK-LABEL: firrtl.circuit "LimitLoc"
firrtl.circuit "LimitLoc" {
  // CHECK: firrtl.module private @Simple0()
  // CHECK-NEXT: loc(fused["A.fir":0:1, "A.fir":1:1, "A.fir":2:1, "A.fir":3:1, "A.fir":4:1, "A.fir":5:1, "A.fir":6:1, "A.fir":7:1])
  firrtl.module private @Simple0() { } loc("A.fir":0:1)
  // CHECK-NOT: @Simple1
  firrtl.module private @Simple1() { } loc("A.fir":1:1)
  // CHECK-NOT: @Simple2
  firrtl.module private @Simple2() { } loc("A.fir":2:1)
  // CHECK-NOT: @Simple3
  firrtl.module private @Simple3() { } loc("A.fir":3:1)
  // CHECK-NOT: @Simple4
  firrtl.module private @Simple4() { } loc("A.fir":4:1)
  // CHECK-NOT: @Simple5
  firrtl.module private @Simple5() { } loc("A.fir":5:1)
  // CHECK-NOT: @Simple6
  firrtl.module private @Simple6() { } loc("A.fir":6:1)
  // CHECK-NOT: @Simple7
  firrtl.module private @Simple7() { } loc("A.fir":7:1)
  // CHECK-NOT: @Simple8
  firrtl.module private @Simple8() { } loc("A.fir":8:1)
  // CHECK-NOT: @Simple9
  firrtl.module private @Simple9() { } loc("A.fir":9:1)
  firrtl.module @LimitLoc() {
    firrtl.instance simple0 @Simple0()
    firrtl.instance simple1 @Simple1()
    firrtl.instance simple2 @Simple2()
    firrtl.instance simple3 @Simple3()
    firrtl.instance simple4 @Simple4()
    firrtl.instance simple5 @Simple5()
    firrtl.instance simple6 @Simple6()
    firrtl.instance simple7 @Simple7()
    firrtl.instance simple8 @Simple8()
    firrtl.instance simple9 @Simple9()
  }
}
