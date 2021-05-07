// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-infer-widths)' --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: @InferConstant
  // CHECK-SAME: out %out0: !firrtl.uint<42>
  // CHECK-SAME: out %out1: !firrtl.sint<42>
  firrtl.module @InferConstant(out %out0: !firrtl.uint, out %out1: !firrtl.sint) {
    %0 = firrtl.constant(1 : ui42) : !firrtl.uint
    %1 = firrtl.constant(2 : si42) : !firrtl.sint
    firrtl.connect %out0, %0 : !firrtl.uint, !firrtl.uint
    firrtl.connect %out1, %1 : !firrtl.sint, !firrtl.sint
  }

  // CHECK-LABEL: @InferOutput
  // CHECK-SAME: out %out: !firrtl.uint<2>
  firrtl.module @InferOutput(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    firrtl.connect %out, %in : !firrtl.uint, !firrtl.uint<2>
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
  firrtl.module @ComparisonOp(in %a: !firrtl.uint<2>, in %b: !firrtl.uint<3>) {
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

  // CHECK-LABEL: @CatDynShiftOp
  firrtl.module @CatDynShiftOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.sint<2>
    // CHECK: %3 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = firrtl.cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %6 = firrtl.dshl {{.*}} -> !firrtl.uint<10>
    // CHECK: %7 = firrtl.dshl {{.*}} -> !firrtl.sint<10>
    // CHECK: %8 = firrtl.dshlw {{.*}} -> !firrtl.uint<3>
    // CHECK: %9 = firrtl.dshlw {{.*}} -> !firrtl.sint<3>
    // CHECK: %10 = firrtl.dshr {{.*}} -> !firrtl.uint<3>
    // CHECK: %11 = firrtl.dshr {{.*}} -> !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.sint
    %3 = firrtl.wire : !firrtl.sint
    %4 = firrtl.cat %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = firrtl.cat %2, %3 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
    %6 = firrtl.dshl %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %7 = firrtl.dshl %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %8 = firrtl.dshlw %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %9 = firrtl.dshlw %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %10 = firrtl.dshr %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %11 = firrtl.dshr %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant(2 : ui3) : !firrtl.uint<3>
    %c1_si2 = firrtl.constant(1 : si2) : !firrtl.sint<2>
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    firrtl.connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CastOp
  firrtl.module @CastOp() {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.asSInt {{.*}} -> !firrtl.sint<2>
    // CHECK: %5 = firrtl.asUInt {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.wire : !firrtl.clock
    %3 = firrtl.wire : !firrtl.asyncreset
    %4 = firrtl.asSInt %0 : (!firrtl.uint) -> !firrtl.sint
    %5 = firrtl.asUInt %1 : (!firrtl.sint) -> !firrtl.uint
    %6 = firrtl.asUInt %2 : (!firrtl.clock) -> !firrtl.uint<1>
    %7 = firrtl.asUInt %3 : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %8 = firrtl.asClock %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
    %9 = firrtl.asAsyncReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CvtOp
  firrtl.module @CvtOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.cvt {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = firrtl.cvt {{.*}} -> !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.cvt %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = firrtl.cvt %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NegOp
  firrtl.module @NegOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.neg {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = firrtl.neg {{.*}} -> !firrtl.sint<4>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.neg %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = firrtl.neg %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NotOp
  firrtl.module @NotOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.not {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.not {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.not %0 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.not %1 : (!firrtl.sint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorReductionOp
  firrtl.module @AndOrXorReductionOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %3 = firrtl.andr {{.*}} -> !firrtl.uint<1>
    // CHECK: %4 = firrtl.orr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.xorr {{.*}} -> !firrtl.uint<1>
    %c0_ui16 = firrtl.constant(0 : ui16) : !firrtl.uint<16>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.andr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %4 = firrtl.orr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %5 = firrtl.xorr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %1, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %2, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @BitsHeadTailPadOp
  firrtl.module @BitsHeadTailPadOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %3 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %8 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %9 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %10 = firrtl.pad {{.*}} -> !firrtl.uint<42>
    // CHECK: %11 = firrtl.pad {{.*}} -> !firrtl.sint<42>
    // CHECK: %12 = firrtl.pad {{.*}} -> !firrtl.uint<99>
    // CHECK: %13 = firrtl.pad {{.*}} -> !firrtl.sint<99>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.wire : !firrtl.uint

    %4 = firrtl.bits %ui 3 to 1 : (!firrtl.uint) -> !firrtl.uint<3>
    %5 = firrtl.bits %si 3 to 1 : (!firrtl.sint) -> !firrtl.uint<3>
    %6 = firrtl.head %ui, 5 : (!firrtl.uint) -> !firrtl.uint<5>
    %7 = firrtl.head %si, 5 : (!firrtl.sint) -> !firrtl.uint<5>
    %8 = firrtl.tail %ui, 30 : (!firrtl.uint) -> !firrtl.uint
    %9 = firrtl.tail %si, 30 : (!firrtl.sint) -> !firrtl.uint
    %10 = firrtl.pad %ui, 13 : (!firrtl.uint) -> !firrtl.uint
    %11 = firrtl.pad %si, 13 : (!firrtl.sint) -> !firrtl.sint
    %12 = firrtl.pad %ui, 99 : (!firrtl.uint) -> !firrtl.uint
    %13 = firrtl.pad %si, 99 : (!firrtl.sint) -> !firrtl.sint

    firrtl.connect %0, %4 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %6 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %3, %7 : !firrtl.uint, !firrtl.uint<5>

    %c0_ui42 = firrtl.constant(0 : ui42) : !firrtl.uint<42>
    %c0_si42 = firrtl.constant(0 : si42) : !firrtl.sint<42>
    firrtl.connect %ui, %c0_ui42 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %si, %c0_si42 : !firrtl.sint, !firrtl.sint<42>
  }

  // CHECK-LABEL: @MuxOp
  firrtl.module @MuxOp(in %a: !firrtl.uint<1>) {
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

  // CHECK-LABEL: @ShlShrOp
  firrtl.module @ShlShrOp() {
    // CHECK: %0 = firrtl.shl {{.*}} -> !firrtl.uint<8>
    // CHECK: %1 = firrtl.shl {{.*}} -> !firrtl.sint<8>
    // CHECK: %2 = firrtl.shr {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.shr {{.*}} -> !firrtl.sint<2>
    // CHECK: %4 = firrtl.shr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.shr {{.*}} -> !firrtl.sint<1>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint

    %0 = firrtl.shl %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %1 = firrtl.shl %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %2 = firrtl.shr %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.shr %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %4 = firrtl.shr %ui, 9 : (!firrtl.uint) -> !firrtl.uint
    %5 = firrtl.shr %si, 9 : (!firrtl.sint) -> !firrtl.sint

    %c0_ui5 = firrtl.constant(0 : ui5) : !firrtl.uint<5>
    %c0_si5 = firrtl.constant(0 : si5) : !firrtl.sint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %si, %c0_si5 : !firrtl.sint, !firrtl.sint<5>
  }

  // CHECK-LABEL: @PassiveCastOp
  firrtl.module @PassiveCastOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %1 = firrtl.asNonPassive {{.*}} : !firrtl.flip<uint<5>>
    // CHECK: %2 = firrtl.asPassive {{.*}} : !firrtl.flip<uint<5>>
    %ui = firrtl.wire : !firrtl.uint
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.asNonPassive %ui : !firrtl.flip<uint>
    %2 = firrtl.asPassive %1 : !firrtl.flip<uint>
    firrtl.connect %0, %2 : !firrtl.uint, !firrtl.uint
    %c0_ui5 = firrtl.constant(0 : ui5) : !firrtl.uint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
  }

  firrtl.module @Foo() {}
}
