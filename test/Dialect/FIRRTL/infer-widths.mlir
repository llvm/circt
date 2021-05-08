// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-infer-widths)' --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: @InferConstant
  // CHECK-SAME: %out0: !firrtl.flip<uint<42>>
  // CHECK-SAME: %out1: !firrtl.flip<sint<42>>
  firrtl.module @InferConstant(%out0: !firrtl.flip<uint>, %out1: !firrtl.flip<sint>) {
    %0 = firrtl.constant(1 : ui42) : !firrtl.uint
    %1 = firrtl.constant(2 : si42) : !firrtl.sint
    firrtl.connect %out0, %0 : !firrtl.flip<uint>, !firrtl.uint
    firrtl.connect %out1, %1 : !firrtl.flip<sint>, !firrtl.sint
  }

  // CHECK-LABEL: @InferOutput
  // CHECK-SAME: %out: !firrtl.flip<uint<2>>
  firrtl.module @InferOutput(%in: !firrtl.uint<2>, %out: !firrtl.flip<uint>) {
    firrtl.connect %out, %in : !firrtl.flip<uint>, !firrtl.uint<2>
  }

  // CHECK-LABEL: @AddSubOp
  firrtl.module @AddSubOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.add {{.*}} -> !firrtl.uint<4>
    // CHECK: %3 = firrtl.sub {{.*}} -> !firrtl.uint<5>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.add %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = firrtl.sub %0, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant(2 : ui3) : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @MulDivRemOp
  firrtl.module @MulDivRemOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.sint<2>
    // CHECK: %3 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.mul {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = firrtl.div {{.*}} -> !firrtl.uint<3>
    // CHECK: %6 = firrtl.div {{.*}} -> !firrtl.sint<4>
    // CHECK: %7 = firrtl.rem {{.*}} -> !firrtl.uint<2>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.sint
    %3 = firrtl.wire : !firrtl.sint
    %4 = firrtl.mul %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = firrtl.div %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %6 = firrtl.div %3, %2 : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
    %7 = firrtl.rem %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant(2 : ui3) : !firrtl.uint<3>
    %c1_si2 = firrtl.constant(1 : si2) : !firrtl.sint<2>
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    firrtl.connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorOp
  firrtl.module @AndOrXorOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.and {{.*}} -> !firrtl.uint<3>
    // CHECK: %3 = firrtl.or {{.*}} -> !firrtl.uint<3>
    // CHECK: %4 = firrtl.xor {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.and %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = firrtl.or %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %4 = firrtl.xor %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant(2 : ui3) : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @ComparisonOp
  firrtl.module @ComparisonOp(%a: !firrtl.uint<2>, %b: !firrtl.uint<3>) {
    // CHECK: %6 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %7 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %8 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %9 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %10 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %11 = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.leq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %1 = firrtl.lt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %2 = firrtl.geq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %3 = firrtl.gt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %4 = firrtl.eq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %5 = firrtl.neq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %6 = firrtl.wire : !firrtl.uint
    %7 = firrtl.wire : !firrtl.uint
    %8 = firrtl.wire : !firrtl.uint
    %9 = firrtl.wire : !firrtl.uint
    %10 = firrtl.wire : !firrtl.uint
    %11 = firrtl.wire : !firrtl.uint
    firrtl.connect %6, %0 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %7, %1 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %8, %2 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %9, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %10, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %11, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @MuxOp
  firrtl.module @MuxOp(%a: !firrtl.uint<1>) {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.mux{{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.mux(%a, %0, %1) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant(2 : ui3) : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  firrtl.module @Foo() {}
}
