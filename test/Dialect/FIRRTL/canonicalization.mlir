// RUN: circt-opt -canonicalize='top-down=true region-simplify=aggressive' %s | FileCheck %s

firrtl.circuit "Casts" {

firrtl.layer @A bind {}

// CHECK-LABEL: firrtl.module @Casts
firrtl.module @Casts(in %ui1 : !firrtl.uint<1>, in %si1 : !firrtl.sint<1>,
    in %clock : !firrtl.clock, in %asyncreset : !firrtl.asyncreset,
    in %inreset : !firrtl.reset, out %outreset : !firrtl.reset,
    out %out_ui1 : !firrtl.uint<1>, out %out_si1 : !firrtl.sint<1>,
    out %out_clock : !firrtl.clock, out %out_asyncreset : !firrtl.asyncreset,
    out %out2_si1 : !firrtl.sint<1>, out %out2_ui1 : !firrtl.uint<1>) {

  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c1_si1 = firrtl.constant 1 : !firrtl.sint<1>
  %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
  %invalid_si1 = firrtl.invalidvalue : !firrtl.sint<1>
  %invalid_clock = firrtl.invalidvalue : !firrtl.clock
  %invalid_asyncreset = firrtl.invalidvalue : !firrtl.asyncreset

  // No effect
  // CHECK: firrtl.matchingconnect %out_ui1, %ui1 : !firrtl.uint<1>
  %0 = firrtl.asUInt %ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out_ui1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %out_si1, %si1 : !firrtl.sint<1>
  %1 = firrtl.asSInt %si1 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  firrtl.connect %out_si1, %1 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK: firrtl.matchingconnect %out_clock, %clock : !firrtl.clock
  %2 = firrtl.asClock %clock : (!firrtl.clock) -> !firrtl.clock
  firrtl.connect %out_clock, %2 : !firrtl.clock, !firrtl.clock
  // CHECK: firrtl.matchingconnect %out_asyncreset, %asyncreset : !firrtl.asyncreset
  %3 = firrtl.asAsyncReset %asyncreset : (!firrtl.asyncreset) -> !firrtl.asyncreset
  firrtl.connect %out_asyncreset, %3 : !firrtl.asyncreset, !firrtl.asyncreset

  // Constant fold.
  // CHECK: firrtl.matchingconnect %out_ui1, %c1_ui1 : !firrtl.uint<1>
  %4 = firrtl.asUInt %c1_si1 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  firrtl.connect %out_ui1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %out_si1, %c-1_si1 : !firrtl.sint<1>
  %5 = firrtl.asSInt %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.sint<1>
  firrtl.connect %out_si1, %5 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK: firrtl.matchingconnect %out_clock, %c1_clock : !firrtl.clock
  %6 = firrtl.asClock %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
  firrtl.connect %out_clock, %6 : !firrtl.clock, !firrtl.clock
  // CHECK: firrtl.matchingconnect %out_asyncreset, %c1_asyncreset : !firrtl.asyncreset
  %7 = firrtl.asAsyncReset %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  firrtl.connect %out_asyncreset, %7 : !firrtl.asyncreset, !firrtl.asyncreset
  // CHECK: firrtl.matchingconnect %outreset, %inreset : !firrtl.reset
  %8 = firrtl.resetCast %inreset : (!firrtl.reset) -> !firrtl.reset
  firrtl.matchingconnect %outreset, %8 : !firrtl.reset

  // Transparent
  // CHECK: firrtl.matchingconnect %out2_si1, %si1
  %9 = firrtl.asUInt %si1 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  %10 = firrtl.asSInt %9 : (!firrtl.uint<1>) -> !firrtl.sint<1>
  firrtl.matchingconnect %out2_si1, %10 : !firrtl.sint<1>
  // CHECK: firrtl.matchingconnect %out2_ui1, %ui1
  %11 = firrtl.asSInt %ui1 : (!firrtl.uint<1>) -> !firrtl.sint<1>
  %12 = firrtl.asUInt %11 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out2_ui1, %12 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %out2_si1, %si1
  %13 = firrtl.cvt %si1 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  firrtl.matchingconnect %out2_si1, %13 : !firrtl.sint<1>
}

// CHECK-LABEL: firrtl.module @Div
firrtl.module @Div(in %a: !firrtl.uint<4>,
                   out %b: !firrtl.uint<4>,
                   in %c: !firrtl.sint<4>,
                   out %d: !firrtl.sint<5>,
                   in %e: !firrtl.uint,
                   out %f: !firrtl.uint,
                   in %g: !firrtl.sint,
                   out %h: !firrtl.sint,
                   out %i: !firrtl.uint<4>) {

  // CHECK-DAG: [[ONE_i4:%.+]] = firrtl.constant 1 : !firrtl.uint<4>
  // CHECK-DAG: [[ONE_s5:%.+]] = firrtl.constant 1 : !firrtl.sint<5>
  // CHECK-DAG: [[ONE_i2:%.+]] = firrtl.constant 1 : !firrtl.uint
  // CHECK-DAG: [[ONE_s2:%.+]] = firrtl.constant 1 : !firrtl.sint

  // Check that 'div(a, a) -> 1' works for known UInt widths.
  // CHECK: firrtl.matchingconnect %b, [[ONE_i4]]
  %0 = firrtl.div %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %b, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // Check that 'div(c, c) -> 1' works for known SInt widths.
  // CHECK: firrtl.matchingconnect %d, [[ONE_s5]] : !firrtl.sint<5>
  %1 = firrtl.div %c, %c : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.sint<5>
  firrtl.connect %d, %1 : !firrtl.sint<5>, !firrtl.sint<5>

  // Check that 'div(e, e) -> 1' works for unknown UInt widths.
  // CHECK: firrtl.connect %f, [[ONE_i2]]
  %2 = firrtl.div %e, %e : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
  firrtl.connect %f, %2 : !firrtl.uint, !firrtl.uint

  // Check that 'div(g, g) -> 1' works for unknown SInt widths.
  // CHECK: firrtl.connect %h, [[ONE_s2]]
  %3 = firrtl.div %g, %g : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
  firrtl.connect %h, %3 : !firrtl.sint, !firrtl.sint

  // Check that 'div(a, 1) -> a' for known UInt widths.
  // CHECK: firrtl.matchingconnect %b, %a
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %4 = firrtl.div %a, %c1_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %b, %4 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %i, %c5_ui4
  %c1_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %5 = firrtl.div %c1_ui4, %c3_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %i, %5 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @And
firrtl.module @And(in %in: !firrtl.uint<4>,
                   in %in6: !firrtl.uint<6>,
                   in %sin: !firrtl.sint<4>,
                   in %zin1: !firrtl.uint<0>,
                   in %zin2: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %out6: !firrtl.uint<6>,
                   out %out5: !firrtl.uint<5>,
                   out %outz: !firrtl.uint<0>) {
  // CHECK: firrtl.matchingconnect %out, %c1_ui4
  %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %c3_si5 = firrtl.constant 3 : !firrtl.sint<5>

  %0 = firrtl.and %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %in
  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %1 = firrtl.and %in, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.and %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  %inv_2 = firrtl.and %c1_ui0, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %inv_2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %in
  %3 = firrtl.and %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  // CHECK: firrtl.matchingconnect %outz, %c0_ui0
  %zw = firrtl.and %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  firrtl.matchingconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type inputs - the constant is zero extended, not sign extended, so it
  // cannot be folded!

  // Narrows, then folds away
  // CHECK: %0 = firrtl.bits %in 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  // CHECK-NEXT: %1 = firrtl.pad %0, 4 : (!firrtl.uint<2>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %out, %1
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %4 = firrtl.and %in, %c3_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %out, %4 : !firrtl.uint<4>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.matchingconnect %out, %c1_ui4
  %c1_si4 = firrtl.constant 1 : !firrtl.sint<4>
  %5 = firrtl.and %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %5 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[AND:.+]] = firrtl.asUInt %sin
  // CHECK-NEXT: firrtl.matchingconnect %out, %[[AND]]
  %6 = firrtl.and %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  %c0_si2 = firrtl.constant 0 : !firrtl.sint<2>
  %7 = firrtl.and %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out, %7 : !firrtl.uint<4>

  // CHECK: %[[trunc:.*]] = firrtl.bits %in6
  // CHECK: %[[ANDPAD:.*]] = firrtl.and %[[trunc]], %in
  // CHECK: %[[POST:.*]] = firrtl.pad %[[ANDPAD]]
  // CHECK: firrtl.matchingconnect %out6, %[[POST]]
  %8 = firrtl.pad %in, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %9 = firrtl.and %in6, %8  : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
  firrtl.matchingconnect %out6, %9 : !firrtl.uint<6>

  // CHECK: %[[AND:.*]] = firrtl.and %in, %c3_ui4
  // CHECK: firrtl.pad %[[AND]], 5
  %10 = firrtl.cvt %in : (!firrtl.uint<4>) -> !firrtl.sint<5>
  %11 = firrtl.and %10, %c3_si5 : (!firrtl.sint<5>, !firrtl.sint<5>) -> !firrtl.uint<5>
  firrtl.matchingconnect %out5, %11 : !firrtl.uint<5>
}

// AndOfAsSInt patterns should not crash on non-IntType inputs
// CHECK-LABEL: firrtl.module @AndOfAsSIntNoCrash
firrtl.module @AndOfAsSIntNoCrash(
  in %clock: !firrtl.clock,
  in %asyncreset: !firrtl.asyncreset,
  in %sint: !firrtl.sint<4>,
  in %y: !firrtl.sint<1>,
  out %out1: !firrtl.uint<1>,
  out %out2: !firrtl.uint<1>,
  out %out3: !firrtl.uint<4>
) {
  // AndOfAsSIntL pattern should not crash on clock type
  // CHECK: %[[ASSINT1:.+]] = firrtl.asSInt %clock
  // CHECK: firrtl.and %[[ASSINT1]], %y
  %0 = firrtl.asSInt %clock : (!firrtl.clock) -> !firrtl.sint<1>
  %1 = firrtl.and %0, %y : (!firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %1 : !firrtl.uint<1>

  // AndOfAsSIntR pattern should not crash on asyncreset type
  // CHECK: %[[ASSINT2:.+]] = firrtl.asSInt %asyncreset
  // CHECK: firrtl.and %y, %[[ASSINT2]]
  %2 = firrtl.asSInt %asyncreset : (!firrtl.asyncreset) -> !firrtl.sint<1>
  %3 = firrtl.and %y, %2 : (!firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out2, %3 : !firrtl.uint<1>

  // AndOfAsSIntL/R patterns should match on SInt.
  // CHECK: %[[ASUINT:.+]] = firrtl.asUInt %sint
  // CHECK: firrtl.matchingconnect %out3, %[[ASUINT]]
  %4 = firrtl.asSInt %sint : (!firrtl.sint<4>) -> !firrtl.sint<4>
  %5 = firrtl.and %4, %sint : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out3, %5 : !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Or
firrtl.module @Or(in %in: !firrtl.uint<4>,
                  in %in6: !firrtl.uint<6>,
                  in %sin: !firrtl.sint<4>,
                  in %zin1: !firrtl.uint<0>,
                  in %zin2: !firrtl.uint<0>,
                  out %out: !firrtl.uint<4>,
                  out %out6: !firrtl.uint<6>,
                  out %outz: !firrtl.uint<0>) {
  // CHECK: firrtl.matchingconnect %out, %c7_ui4
  %c4_ui4 = firrtl.constant 4 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.or %c3_ui4, %c4_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c15_ui4
  %c1_ui15 = firrtl.constant 15 : !firrtl.uint<4>
  %1 = firrtl.or %in, %c1_ui15 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %in
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.or %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %in
  %inv_2 = firrtl.or %c1_ui0, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %inv_2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %in
  %3 = firrtl.or %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  // CHECK: firrtl.matchingconnect %outz, %c0_ui0
  %zw = firrtl.or %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  firrtl.matchingconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type input and outputs.

  // CHECK: firrtl.matchingconnect %out, %c1_ui4
  %c1_si4 = firrtl.constant 1 : !firrtl.sint<4>
  %5 = firrtl.or %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %5 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: [[OR:%.+]] = firrtl.asUInt %sin
  // CHECK-NEXT: firrtl.matchingconnect %out, [[OR]]
  %6 = firrtl.or %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c15_ui4
  %c0_si2 = firrtl.constant -1 : !firrtl.sint<2>
  %7 = firrtl.or %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  firrtl.connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[trunc:.*]] = firrtl.bits %in6
  // CHECK: %[[trunc2:.*]] = firrtl.bits %in6
  // CHECK: %[[OR:.*]] = firrtl.or %[[trunc2]], %in
  // CHECK: %[[CAT:.*]] = firrtl.cat %[[trunc]], %[[OR]]
  // CHECK: firrtl.matchingconnect %out6, %[[CAT]]
  %8 = firrtl.pad %in, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %9 = firrtl.or %in6, %8  : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
  firrtl.connect %out6, %9 : !firrtl.uint<6>, !firrtl.uint<6>

}

// CHECK-LABEL: firrtl.module @Xor
firrtl.module @Xor(in %in: !firrtl.uint<4>,
                   in %in6: !firrtl.uint<6>,
                   in %sin: !firrtl.sint<4>,
                   in %zin1: !firrtl.uint<0>,
                   in %zin2: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %out6: !firrtl.uint<6>,
                   out %outz: !firrtl.uint<0>) {
  // CHECK: firrtl.matchingconnect %out, %c2_ui4
  %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.xor %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %in
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.xor %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  %3 = firrtl.xor %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  // CHECK: firrtl.matchingconnect %outz, %c0_ui0
  %zw = firrtl.xor %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  firrtl.matchingconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type input and outputs.

  // CHECK: firrtl.matchingconnect %out, %c0_ui4
  %6 = firrtl.xor %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[aui:.*]] = firrtl.asUInt %sin
  // CHECK: firrtl.matchingconnect %out, %[[aui]]
  %c0_si2 = firrtl.constant 0 : !firrtl.sint<2>
  %7 = firrtl.xor %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  firrtl.connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[trunc:.*]] = firrtl.bits %in6
  // CHECK: %[[trunc2:.*]] = firrtl.bits %in6
  // CHECK: %[[XOR:.*]] = firrtl.xor %[[trunc2]], %in
  // CHECK: %[[CAT:.*]] = firrtl.cat %[[trunc]], %[[XOR]]
  // CHECK: firrtl.matchingconnect %out6, %[[CAT]]
  %8 = firrtl.pad %in, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %9 = firrtl.xor %in6, %8  : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
  firrtl.connect %out6, %9 : !firrtl.uint<6>, !firrtl.uint<6>

}

// CHECK-LABEL: firrtl.module @Not
firrtl.module @Not(in %in: !firrtl.uint<4>,
                   in %sin: !firrtl.sint<4>,
                   out %out1: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>,
                   out %outs: !firrtl.uint<4>) {
  %0 = firrtl.not %in : (!firrtl.uint<4>) -> !firrtl.uint<4>
  %1 = firrtl.not %0 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>
  %2 = firrtl.not %sin : (!firrtl.sint<4>) -> !firrtl.uint<4>
  %3 = firrtl.not %2 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %outs, %3 : !firrtl.uint<4>, !firrtl.uint<4>
  // CHECK: firrtl.matchingconnect %outu, %in
  // CHECK: %[[cast:.*]] = firrtl.asUInt %sin
  // CHECK: firrtl.matchingconnect %outs, %[[cast]]

  %c5_ui4 = firrtl.constant 5 : !firrtl.uint<4>
  %5 = firrtl.eq %in, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %6 = firrtl.not %5 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %6 : !firrtl.uint<1>
  // CHECK: firrtl.neq
  // CHECK-NEXT: firrtl.matchingconnect

  %7 = firrtl.neq %in, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %8 = firrtl.not %7 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %8 : !firrtl.uint<1>
  // CHECK: firrtl.eq
  // CHECK-NEXT: firrtl.matchingconnect

  %9 = firrtl.lt %in, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %10 = firrtl.not %9 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %10 : !firrtl.uint<1>
  // CHECK: firrtl.geq
  // CHECK-NEXT: firrtl.matchingconnect

  %11 = firrtl.leq %in, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %12 = firrtl.not %11 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %12 : !firrtl.uint<1>
  // CHECK: firrtl.gt
  // CHECK-NEXT: firrtl.matchingconnect

  %13 = firrtl.gt %in, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %14 = firrtl.not %13 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %14 : !firrtl.uint<1>
  // CHECK: firrtl.leq
  // CHECK-NEXT: firrtl.matchingconnect

  %15 = firrtl.geq %in, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %16 = firrtl.not %15 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %16 : !firrtl.uint<1>
  // CHECK: firrtl.lt
  // CHECK-NEXT: firrtl.matchingconnect

}

// CHECK-LABEL: firrtl.module @EQ
firrtl.module @EQ(in %in1: !firrtl.uint<1>,
                  in %in4: !firrtl.uint<4>,
                  out %out: !firrtl.uint<1>) {
  // CHECK: firrtl.matchingconnect %out, %in1
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %0 = firrtl.eq %in1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // Issue #368: https://github.com/llvm/circt/issues/368
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %1 = firrtl.eq %in1, %c3_ui2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<1>
  firrtl.connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.eq %in1, %c3_ui2
  // CHECK-NEXT: firrtl.matchingconnect

  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %2 = firrtl.eq %in1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.not %in1
  // CHECK-NEXT: firrtl.matchingconnect

  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %3 = firrtl.eq %in4, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.andr %in4
  // CHECK-NEXT: firrtl.matchingconnect

  %4 = firrtl.eq %in4, %c0_ui1 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: [[ORR:%.+]] = firrtl.orr %in4
  // CHECK-NEXT: firrtl.not [[ORR]]
  // CHECK-NEXT: firrtl.matchingconnect

  %c5_ui4 = firrtl.constant 5 : !firrtl.uint<4>
  %5 = firrtl.neq %in4, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %6 = firrtl.eq %5, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out, %6 : !firrtl.uint<1>
  // CHECK: firrtl.eq
  // CHECK-NEXT: firrtl.matchingconnect

}

// CHECK-LABEL: firrtl.module @NEQ
firrtl.module @NEQ(in %in1: !firrtl.uint<1>,
                   in %in4: !firrtl.uint<4>,
                   out %out: !firrtl.uint<1>) {
  // CHECK: firrtl.matchingconnect %out, %in
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %0 = firrtl.neq %in1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %1 = firrtl.neq %in1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.not %in1
  // CHECK-NEXT: firrtl.matchingconnect

  %2 = firrtl.neq %in4, %c0_ui1 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.orr %in4
  // CHECK-NEXT: firrtl.matchingconnect

  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %4 = firrtl.neq %in4, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: [[ANDR:%.+]] = firrtl.andr %in4
  // CHECK-NEXT: firrtl.not [[ANDR]]
  // CHECK-NEXT: firrtl.matchingconnect

  %c5_ui4 = firrtl.constant 5 : !firrtl.uint<4>
  %5 = firrtl.eq %in4, %c5_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %6 = firrtl.neq %5, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out, %6 : !firrtl.uint<1>
  // CHECK: firrtl.neq
  // CHECK-NEXT: firrtl.matchingconnect
}

// CHECK-LABEL: firrtl.module @Cat
firrtl.module @Cat(in %in4: !firrtl.uint<4>,
                   in %sin4: !firrtl.sint<4>,
                   out %out4: !firrtl.uint<4>,
                   out %outcst: !firrtl.uint<8>,
                   out %outcst2: !firrtl.uint<8>,
                   out %outu8: !firrtl.uint<8>,
                   in %in0 : !firrtl.uint<0>,
                   out %outpt1: !firrtl.uint<4>,
                   out %outpt2 : !firrtl.uint<4>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c0_si2 = firrtl.constant 0 : !firrtl.sint<2>

  // CHECK: firrtl.matchingconnect %out4, %in4
  %0 = firrtl.bits %in4 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %1 = firrtl.bits %in4 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %2 = firrtl.cat %0, %1 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %out4, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %outcst, %c243_ui8
  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %3 = firrtl.cat %c15_ui4, %c3_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>
  firrtl.connect %outcst, %3 : !firrtl.uint<8>, !firrtl.uint<8>

  // CHECK: firrtl.matchingconnect %outpt1, %in4
  %5 = firrtl.cat %in0, %in4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %outpt1, %5 : !firrtl.uint<4>, !firrtl.uint<4>
  // CHECK: firrtl.matchingconnect %outpt2, %in4
  %6 = firrtl.cat %in4, %in0 : (!firrtl.uint<4>, !firrtl.uint<0>) -> !firrtl.uint<4>
  firrtl.connect %outpt2, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.cat %c0_ui4, %in4
  %7 = firrtl.cat %c0_ui2, %in4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<6>
  %8 = firrtl.cat %c0_ui2, %7 : (!firrtl.uint<2>, !firrtl.uint<6>) -> !firrtl.uint<8>
  firrtl.connect %outcst, %8 : !firrtl.uint<8>, !firrtl.uint<8>

  // CHECK: firrtl.asUInt %sin4
  // CHECK-NEXT: firrtl.cat %c0_ui4
  %9  = firrtl.cat %c0_si2, %sin4 : (!firrtl.sint<2>, !firrtl.sint<4>) -> !firrtl.uint<6>
  %10 = firrtl.cat %c0_ui2, %9 : (!firrtl.uint<2>, !firrtl.uint<6>) -> !firrtl.uint<8>
  firrtl.connect %outcst, %10 : !firrtl.uint<8>, !firrtl.uint<8>

  // CHECK: %[[fixedsign:.*]] = firrtl.cat %sin4, %sin4
  // CHECK-NEXT: firrtl.matchingconnect %outu8, %[[fixedsign]]
  %tcast = firrtl.asUInt %sin4 : (!firrtl.sint<4>) -> !firrtl.uint<4>
  %11 = firrtl.cat %tcast, %tcast : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>
  firrtl.matchingconnect %outu8, %11 : !firrtl.uint<8>
}

// Regression test: CatCast pattern should not crash on non-IntType inputs
// CHECK-LABEL: firrtl.module @CatCastNoCrash
firrtl.module @CatCastNoCrash(
  in %clock: !firrtl.clock,
  in %asyncreset: !firrtl.asyncreset,
  in %sint: !firrtl.sint<4>,
  out %out1: !firrtl.uint<2>,
  out %out2: !firrtl.uint<2>,
  out %out3: !firrtl.uint<8>
) {
  // CatCast pattern should not crash on clock type
  // CHECK: %[[ASUINT1:.+]] = firrtl.asUInt %clock
  // CHECK: %[[ASUINT2:.+]] = firrtl.asUInt %clock
  // CHECK: firrtl.cat %[[ASUINT1]], %[[ASUINT2]]
  %0 = firrtl.asUInt %clock : (!firrtl.clock) -> !firrtl.uint<1>
  %1 = firrtl.asUInt %clock : (!firrtl.clock) -> !firrtl.uint<1>
  %2 = firrtl.cat %0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  firrtl.matchingconnect %out1, %2 : !firrtl.uint<2>

  // CatCast pattern should not crash on asyncreset type
  // CHECK: %[[ASUINT3:.+]] = firrtl.asUInt %asyncreset
  // CHECK: %[[ASUINT4:.+]] = firrtl.asUInt %asyncreset
  // CHECK: firrtl.cat %[[ASUINT3]], %[[ASUINT4]]
  %3 = firrtl.asUInt %asyncreset : (!firrtl.asyncreset) -> !firrtl.uint<1>
  %4 = firrtl.asUInt %asyncreset : (!firrtl.asyncreset) -> !firrtl.uint<1>
  %5 = firrtl.cat %3, %4 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  firrtl.matchingconnect %out2, %5 : !firrtl.uint<2>

  // CatCast pattern should match on SInt.
  // CHECK: firrtl.cat %sint, %sint
  %6 = firrtl.asUInt %sint : (!firrtl.sint<4>) -> !firrtl.uint<4>
  %7 = firrtl.asUInt %sint : (!firrtl.sint<4>) -> !firrtl.uint<4>
  %8 = firrtl.cat %6, %7 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>
  firrtl.matchingconnect %out3, %8 : !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @Bits
firrtl.module @Bits(in %in1: !firrtl.uint<1>,
                    in %in4: !firrtl.uint<4>,
                    out %out1: !firrtl.uint<1>,
                    out %out2: !firrtl.uint<2>,
                    out %out4: !firrtl.uint<4>,
                    out %out2b: !firrtl.uint<2>) {
  // CHECK: firrtl.matchingconnect %out1, %in1
  %0 = firrtl.bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %out4, %in4
  %1 = firrtl.bits %in4 3 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out4, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out2, %c1_ui2
  %c10_ui4 = firrtl.constant 10 : !firrtl.uint<4>
  %2 = firrtl.bits %c10_ui4 2 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  firrtl.connect %out2, %2 : !firrtl.uint<2>, !firrtl.uint<2>


  // CHECK: firrtl.bits %in4 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %out1, %
  %3 = firrtl.bits %in4 3 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  %4 = firrtl.bits %3 1 to 1 : (!firrtl.uint<3>) -> !firrtl.uint<1>
  firrtl.connect %out1, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %out1, %in1
  %5 = firrtl.bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %5 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %out2b, %c1_ui2
  %c11_ui4 = firrtl.constant 11 : !firrtl.uint<4>
  %6 = firrtl.mux( %in1, %c10_ui4, %c11_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  %7 = firrtl.bits %6 2 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  firrtl.matchingconnect %out2b, %7 : !firrtl.uint<2>
}

// CHECK-LABEL: firrtl.module @Head
firrtl.module @Head(in %in4u: !firrtl.uint<4>,
                    out %out1u: !firrtl.uint<1>,
                    out %out3u: !firrtl.uint<3>) {
  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 3
  // CHECK-NEXT: firrtl.matchingconnect %out1u, [[BITS]]
  %0 = firrtl.head %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 1
  // CHECK-NEXT: firrtl.matchingconnect %out3u, [[BITS]]
  %1 = firrtl.head %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %1 : !firrtl.uint<3>, !firrtl.uint<3>

  // CHECK: firrtl.matchingconnect %out3u, %c5_ui3
  %c10_ui4 = firrtl.constant 10 : !firrtl.uint<4>
  %2 = firrtl.head %c10_ui4, 3 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %2 : !firrtl.uint<3>, !firrtl.uint<3>
}

// CHECK-LABEL: firrtl.module @Mux
firrtl.module @Mux(in %in: !firrtl.uint<4>,
                   in %cond: !firrtl.uint<1>,
                   in %val1: !firrtl.uint<1>,
                   in %val2: !firrtl.uint<1>,
                   in %val0: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %out1: !firrtl.uint<1>,
                   out %out2: !firrtl.uint<0>,
                   out %out3: !firrtl.uint<1>,
                   out %out4: !firrtl.uint<4>,
                   out %out5: !firrtl.uint<1>,
                   out %out6: !firrtl.uint<1>) {
  // CHECK: firrtl.matchingconnect %out, %in
  %0 = firrtl.int.mux2cell (%cond, %in, %in) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out, %c7_ui4
  %c7_ui4 = firrtl.constant 7 : !firrtl.uint<4>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %2 = firrtl.mux (%c0_ui1, %in, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %out1, %cond
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %3 = firrtl.mux (%cond, %c1_ui1, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %3 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %out, %invalid_ui4
  %invalid_ui4 = firrtl.invalidvalue : !firrtl.uint<4>
  %7 = firrtl.mux (%cond, %invalid_ui4, %invalid_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

  %9 = firrtl.multibit_mux %c1_ui1, %c0_ui1, %cond : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %out1, %c0_ui1
  firrtl.connect %out1, %9 : !firrtl.uint<1>, !firrtl.uint<1>

  %10 = firrtl.multibit_mux %cond, %val1, %val2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: %[[MUX:.+]] = firrtl.mux(%cond, %val1, %val2)
  // CHECK-NEXT: firrtl.matchingconnect %out1, %[[MUX]]
  firrtl.connect %out1, %10 : !firrtl.uint<1>, !firrtl.uint<1>

  %11 = firrtl.multibit_mux %cond, %val1, %val1, %val1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %out1, %val1
  firrtl.connect %out1, %11 : !firrtl.uint<1>, !firrtl.uint<1>

  %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
  %12 = firrtl.multibit_mux %c0_ui0, %val1, %val1 :!firrtl.uint<0>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %out1, %val1
  firrtl.connect %out1, %12 : !firrtl.uint<1>, !firrtl.uint<1>

  %13 = firrtl.mux (%cond, %val0, %val0) : (!firrtl.uint<1>, !firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  // CHECK-NEXT: firrtl.matchingconnect %out2, %c0_ui0
  firrtl.matchingconnect %out2, %13 : !firrtl.uint<0>

  %14 = firrtl.mux (%cond, %c0_ui1, %c1_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK-NEXT: [[V1:%.+]] = firrtl.not %cond
  // CHECK-NEXT: firrtl.matchingconnect %out3, [[V1]]
  firrtl.connect %out3, %14 : !firrtl.uint<1>, !firrtl.uint<1>

  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
  %15 = firrtl.mux (%cond, %c0_ui4, %c1_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: [[V2:%.+]] = firrtl.mux(%cond
  // CHECK-NEXT: firrtl.matchingconnect %out4, [[V2]]
  firrtl.connect %out4, %15 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK-NEXT: firrtl.matchingconnect %out5, %val2
  %16 = firrtl.mux (%val0, %val1, %val2) : (!firrtl.uint<0>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out5, %16 : !firrtl.uint<1>

  // CHECK-NEXT: %[[SEL:.+]] = firrtl.pad %val1, 2 : (!firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK-NEXT: mux4cell(%[[SEL]],
  %17 = firrtl.int.mux4cell (%val1, %val1, %val2, %val1, %val2) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out6, %17 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Pad
firrtl.module @Pad(in %in1u: !firrtl.uint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>,
                   out %outs: !firrtl.sint<4>) {
  // CHECK: firrtl.matchingconnect %out1u, %in1u
  %0 = firrtl.pad %in1u, 1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %outu, %c1_ui4
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %1 = firrtl.pad %c1_ui1, 4 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  firrtl.connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.matchingconnect %outs, %c-1_si4
  %c1_si1 = firrtl.constant -1 : !firrtl.sint<1>
  %2 = firrtl.pad %c1_si1, 4 : (!firrtl.sint<1>) -> !firrtl.sint<4>
  firrtl.connect %outs, %2 : !firrtl.sint<4>, !firrtl.sint<4>
}

// CHECK-LABEL: firrtl.module @Shl
firrtl.module @Shl(in %in1u: !firrtl.uint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>) {
  // CHECK: firrtl.matchingconnect %out1u, %in1u
  %0 = firrtl.shl %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %outu, %c8_ui4
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %1 = firrtl.shl %c1_ui1, 3 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  firrtl.connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Shr
firrtl.module @Shr(in %in1u: !firrtl.uint<1>,
                   in %in4u: !firrtl.uint<4>,
                   in %inu: !firrtl.uint,
                   in %in1s: !firrtl.sint<1>,
                   in %in4s: !firrtl.sint<4>,
                   in %ins: !firrtl.sint,
                   in %in0u: !firrtl.uint<0>,
                   in %in0s: !firrtl.sint<0>,
                   out %out0u: !firrtl.uint<0>,
                   out %out1s: !firrtl.sint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %out4u: !firrtl.uint<4>,
                   out %out4s: !firrtl.sint<4>,
                   out %outu: !firrtl.uint,
                   out %outs: !firrtl.sint
                   ) {
  // CHECK: firrtl.matchingconnect %out1u, %in1u
  %0 = firrtl.shr %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %out1u, %c0_ui1
  %1 = firrtl.shr %in4u, 4 : (!firrtl.uint<4>) -> !firrtl.uint<0>
  firrtl.connect %out1u, %1 : !firrtl.uint<1>, !firrtl.uint<0>

  // CHECK: firrtl.matchingconnect %out1u, %c0_ui1
  %2 = firrtl.shr %in4u, 5 : (!firrtl.uint<4>) -> !firrtl.uint<0>
  firrtl.connect %out1u, %2 : !firrtl.uint<1>, !firrtl.uint<0>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.matchingconnect %out1s, [[CAST]]
  %3 = firrtl.shr %in4s, 3 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %3 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.matchingconnect %out1s, [[CAST]]
  %4 = firrtl.shr %in4s, 4 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %4 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.matchingconnect %out1s, [[CAST]]
  %5 = firrtl.shr %in4s, 5 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %5 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: firrtl.matchingconnect %out1u, %c1_ui1
  %c12_ui4 = firrtl.constant 12 : !firrtl.uint<4>
  %6 = firrtl.shr %c12_ui4, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %6 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 3
  // CHECK-NEXT: firrtl.matchingconnect %out1u, [[BITS]]
  %7 = firrtl.shr %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %7 : !firrtl.uint<1>, !firrtl.uint<1>

  // Issue #313: https://github.com/llvm/circt/issues/313
  // CHECK: firrtl.matchingconnect %out1s, %in1s : !firrtl.sint<1>
  %8 = firrtl.shr %in1s, 42 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %8 : !firrtl.sint<1>, !firrtl.sint<1>

  // Issue #1064: https://github.com/llvm/circt/issues/1064
  // CHECK: firrtl.matchingconnect %out1u, %c0_ui1
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %9 = firrtl.dshr %in0u, %c1_ui1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  firrtl.connect %out1u, %9 : !firrtl.uint<1>, !firrtl.uint<0>

  // Issue #6608: https://github.com/llvm/circt/issues/6608
  // CHECK: firrtl.matchingconnect %out0u, %c0_ui0
  %10 = firrtl.shr %in0u, 0 : (!firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.matchingconnect %out0u, %10 : !firrtl.uint<0>

  // Issue #6608: https://github.com/llvm/circt/issues/6608
  // CHECK: firrtl.matchingconnect %out1s, %c0_si1
  %11 = firrtl.shr %in0s, 0 : (!firrtl.sint<0>) -> !firrtl.sint<1>
  firrtl.matchingconnect %out1s, %11 : !firrtl.sint<1>

  // Issue #6608: https://github.com/llvm/circt/issues/6608
  // CHECK: firrtl.matchingconnect %out4u, %in4u
  %12 = firrtl.shr %in4u, 0 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out4u, %12 : !firrtl.uint<4>

  // Issue #6608: https://github.com/llvm/circt/issues/6608
  // CHECK: firrtl.matchingconnect %out4s, %in4s
  %13 = firrtl.shr %in4s, 0 : (!firrtl.sint<4>) -> !firrtl.sint<4>
  firrtl.matchingconnect %out4s, %13 : !firrtl.sint<4>

  // Issue #6608: https://github.com/llvm/circt/issues/6608
  // Will change to drop op once FIRRTL spec changes sizeof(shr(uint))
  // CHECK: %[[UINT:.+]] = firrtl.shr %inu
  // CHECK: firrtl.connect %outu, %[[UINT]]
  %14 = firrtl.shr %inu, 0 : (!firrtl.uint) -> !firrtl.uint
  firrtl.connect %outu, %14 : !firrtl.uint, !firrtl.uint

  // Issue #6608: https://github.com/llvm/circt/issues/6608
  // CHECK: %[[SINT:.+]] = firrtl.shr %ins
  // CHECK: firrtl.connect %outs, %[[SINT]]
  %15 = firrtl.shr %ins, 0 : (!firrtl.sint) -> !firrtl.sint
  firrtl.connect %outs, %15 : !firrtl.sint, !firrtl.sint

}

// CHECK-LABEL: firrtl.module @Tail
firrtl.module @Tail(in %in4u: !firrtl.uint<4>,
                    out %out1u: !firrtl.uint<1>,
                    out %out3u: !firrtl.uint<3>) {
  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 0 to 0
  // CHECK-NEXT: firrtl.matchingconnect %out1u, [[BITS]]
  %0 = firrtl.tail %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 2 to 0
  // CHECK-NEXT: firrtl.matchingconnect %out3u, [[BITS]]
  %1 = firrtl.tail %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %1 : !firrtl.uint<3>, !firrtl.uint<3>

  // CHECK: firrtl.matchingconnect %out3u, %c2_ui3
  %c10_ui4 = firrtl.constant 10 : !firrtl.uint<4>
  %2 = firrtl.tail %c10_ui4, 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %2 : !firrtl.uint<3>, !firrtl.uint<3>
}

// CHECK-LABEL: firrtl.module @Andr
firrtl.module @Andr(in %in0 : !firrtl.uint<0>, in %in1 : !firrtl.sint<2>,
                    in %in2 : !firrtl.uint<2>,
                    out %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                    out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>,
                    out %e: !firrtl.uint<1>, out %f: !firrtl.uint<1>,
                    out %g: !firrtl.uint<1>, in %h : !firrtl.uint<64>,
                    out %i: !firrtl.uint<1>, out %j: !firrtl.uint<1>) {
  %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<2>
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %cn2_si2 = firrtl.constant -2 : !firrtl.sint<2>
  %cn1_si2 = firrtl.constant -1 : !firrtl.sint<2>
  %0 = firrtl.andr %c2_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %1 = firrtl.andr %c3_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %2 = firrtl.andr %cn2_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %3 = firrtl.andr %cn1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %4 = firrtl.andr %in0 : (!firrtl.uint<0>) -> !firrtl.uint<1>
  // CHECK: %[[ZERO:.+]] = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK: %[[ONE:.+]] = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %a, %[[ZERO]]
  firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %b, %[[ONE]]
  firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %c, %[[ZERO]]
  firrtl.connect %c, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %d, %[[ONE]]
  firrtl.connect %d, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %e, %[[ONE]]
  firrtl.connect %e, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: %[[and1:.*]] = firrtl.andr %in1
  // CHECK-NEXT: firrtl.matchingconnect %e, %[[and1]]
  %cat = firrtl.cat %in1, %cn1_si2 : (!firrtl.sint<2>, !firrtl.sint<2>) -> !firrtl.uint<4>
  %andrcat = firrtl.andr %cat : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %e, %andrcat : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %e, %[[ZERO]]
  %cat2 = firrtl.cat %in1, %cn2_si2 : (!firrtl.sint<2>, !firrtl.sint<2>) -> !firrtl.uint<4>
  %andrcat2 = firrtl.andr %cat2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %e, %andrcat2 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %g, %[[ZERO]]
  %5 = firrtl.asSInt %h : (!firrtl.uint<64>) -> !firrtl.sint<64>
  %6 = firrtl.asUInt %5 : (!firrtl.sint<64>) -> !firrtl.uint<64>
  %9 = firrtl.cvt %6 : (!firrtl.uint<64>) -> !firrtl.sint<65>
  %10 = firrtl.andr %9 : (!firrtl.sint<65>) -> !firrtl.uint<1>
  firrtl.matchingconnect %g, %10 : !firrtl.uint<1>

  // CHECK: %[[andr:.*]] = firrtl.andr %in1
  // CHECK-NEXT: firrtl.matchingconnect %i, %[[andr]]
  %11 = firrtl.pad %in1, 3 : (!firrtl.sint<2>) -> !firrtl.sint<3>
  %12 = firrtl.andr %11 : (!firrtl.sint<3>) -> !firrtl.uint<1>
  firrtl.matchingconnect %i, %12 : !firrtl.uint<1>

  // CHECK: %[[cat:.*]] = firrtl.cat %in2, %h
  // CHECK-NEXT: firrtl.andr %[[cat]]
  %13 = firrtl.andr %in2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %14 = firrtl.cat %13, %h : (!firrtl.uint<1>, !firrtl.uint<64>) -> !firrtl.uint<65>
  %15 = firrtl.andr %14 : (!firrtl.uint<65>) -> !firrtl.uint<1>
  firrtl.matchingconnect %j, %15 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Orr
firrtl.module @Orr(in %in0 : !firrtl.uint<0>, in %in2 : !firrtl.uint<2>,
                   out %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                   out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>,
                   out %e: !firrtl.uint<1>, out %f: !firrtl.uint<1>,
                   out %g: !firrtl.uint<1>, in %h : !firrtl.uint<64>,
                   out %j : !firrtl.uint<1>) {
  %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<2>
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
  %cn0_si2 = firrtl.constant 0 : !firrtl.sint<2>
  %cn2_si2 = firrtl.constant -2 : !firrtl.sint<2>
  %0 = firrtl.orr %c0_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %1 = firrtl.orr %c2_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %2 = firrtl.orr %cn0_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %3 = firrtl.orr %cn2_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %4 = firrtl.orr %in0 : (!firrtl.uint<0>) -> !firrtl.uint<1>
  // CHECK-DAG: %[[ZERO:.+]] = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK-DAG: %[[ONE:.+]] = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %a, %[[ZERO]]
  firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %b, %[[ONE]]
  firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %c, %[[ZERO]]
  firrtl.connect %c, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %d, %[[ONE]]
  firrtl.connect %d, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %e, %[[ZERO]]
  firrtl.connect %e, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: %[[OR:.*]] = firrtl.orr %h
  // CHECK: firrtl.matchingconnect %g, %[[OR]]
  %5 = firrtl.asSInt %h : (!firrtl.uint<64>) -> !firrtl.sint<64>
  %6 = firrtl.asUInt %5 : (!firrtl.sint<64>) -> !firrtl.uint<64>
  %7 = firrtl.cat %6, %c0_ui2 : (!firrtl.uint<64>, !firrtl.uint<2>) -> !firrtl.uint<66>
  %8 = firrtl.cat %c0_ui2, %7 : (!firrtl.uint<2>, !firrtl.uint<66>) -> !firrtl.uint<68>
  %9 = firrtl.cvt %8 : (!firrtl.uint<68>) -> !firrtl.sint<69>
  %10 = firrtl.orr %9 : (!firrtl.sint<69>) -> !firrtl.uint<1>
  firrtl.matchingconnect %g, %10 : !firrtl.uint<1>

  // CHECK: %[[cat:.*]] = firrtl.cat %in2, %h
  // CHECK-NEXT: firrtl.orr %[[cat]]
  %13 = firrtl.orr %in2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %14 = firrtl.cat %13, %h : (!firrtl.uint<1>, !firrtl.uint<64>) -> !firrtl.uint<65>
  %15 = firrtl.orr %14 : (!firrtl.uint<65>) -> !firrtl.uint<1>
  firrtl.matchingconnect %j, %15 : !firrtl.uint<1>

}

// CHECK-LABEL: firrtl.module @Xorr
firrtl.module @Xorr(in %in0 : !firrtl.uint<0>, in %in2 : !firrtl.uint<2>,
                    out %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                    out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>,
                    out %e: !firrtl.uint<1>, out %f: !firrtl.uint<1>,
                    out %g: !firrtl.uint<1>, in %h : !firrtl.uint<64>,
                    out %j : !firrtl.uint<1>) {
  %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<2>
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %cn1_si2 = firrtl.constant -1 : !firrtl.sint<2>
  %cn2_si2 = firrtl.constant -2 : !firrtl.sint<2>
  %0 = firrtl.xorr %c3_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %1 = firrtl.xorr %c2_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %2 = firrtl.xorr %cn1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %3 = firrtl.xorr %cn2_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %4 = firrtl.xorr %in0 : (!firrtl.uint<0>) -> !firrtl.uint<1>
  // CHECK-DAG: %[[ZERO:.+]] = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK-DAG: %[[ONE:.+]] = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %a, %[[ZERO]]
  firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %b, %[[ONE]]
  firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %c, %[[ZERO]]
  firrtl.connect %c, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %d, %[[ONE]]
  firrtl.connect %d, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %e, %[[ZERO]]
  firrtl.connect %e, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: %[[XOR:.*]] = firrtl.xorr %h
  // CHECK: firrtl.matchingconnect %g, %[[OR]]
  %5 = firrtl.asSInt %h : (!firrtl.uint<64>) -> !firrtl.sint<64>
  %6 = firrtl.asUInt %5 : (!firrtl.sint<64>) -> !firrtl.uint<64>
  %7 = firrtl.cat %6, %c0_ui2 : (!firrtl.uint<64>, !firrtl.uint<2>) -> !firrtl.uint<66>
  %8 = firrtl.cat %c0_ui2, %7 : (!firrtl.uint<2>, !firrtl.uint<66>) -> !firrtl.uint<68>
  %9 = firrtl.cvt %8 : (!firrtl.uint<68>) -> !firrtl.sint<69>
  %10 = firrtl.xorr %9 : (!firrtl.sint<69>) -> !firrtl.uint<1>
  firrtl.matchingconnect %g, %10 : !firrtl.uint<1>

  // CHECK: %[[cat:.*]] = firrtl.cat %in2, %h
  // CHECK-NEXT: firrtl.xorr %[[cat]]
  %13 = firrtl.xorr %in2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %14 = firrtl.cat %13, %h : (!firrtl.uint<1>, !firrtl.uint<64>) -> !firrtl.uint<65>
  %15 = firrtl.xorr %14 : (!firrtl.uint<65>) -> !firrtl.uint<1>
  firrtl.matchingconnect %j, %15 : !firrtl.uint<1>

}

// CHECK-LABEL: firrtl.module @Reduce
firrtl.module @Reduce(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                      out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
  %0 = firrtl.andr %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
  %1 = firrtl.orr %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
  %2 = firrtl.xorr %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %b, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %b, %a
  firrtl.connect %c, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %c, %a
  firrtl.connect %d, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %d, %a
}

// CHECK-LABEL: firrtl.module @ReductionCat
firrtl.module @ReductionCat(in %a: !firrtl.uint<4>, in %b: !firrtl.uint<3>,
                           out %out_andr: !firrtl.uint<1>, out %out_orr: !firrtl.uint<1>,
                           out %out_xorr: !firrtl.uint<1>) {
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

  // Test AndR with cat containing zero constant - should become constant 0
  // CHECK: firrtl.matchingconnect %out_andr, %c0_ui1
  %cat1 = firrtl.cat %a, %c0_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>
  %andr1 = firrtl.andr %cat1 : (!firrtl.uint<6>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out_andr, %andr1 : !firrtl.uint<1>

  // Test AndR with cat containing all-ones constant - should reduce to andr(a)
  // CHECK: %[[ANDR_A:.+]] = firrtl.andr %a
  // CHECK: firrtl.matchingconnect %out_andr, %[[ANDR_A]]
  %cat2 = firrtl.cat %c15_ui4, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>
  %andr2 = firrtl.andr %cat2 : (!firrtl.uint<8>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out_andr, %andr2 : !firrtl.uint<1>

  // Test OrR with cat containing non-zero constant - should become constant 1
  // CHECK: firrtl.matchingconnect %out_orr, %c1_ui1
  %cat3 = firrtl.cat %c3_ui2, %a : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<6>
  %orr1 = firrtl.orr %cat3 : (!firrtl.uint<6>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out_orr, %orr1 : !firrtl.uint<1>

  // Test OrR with cat containing only zero constants - should reduce to orr(a)
  // CHECK: %[[ORR_A:.+]] = firrtl.orr %a
  // CHECK: firrtl.matchingconnect %out_orr, %[[ORR_A]]
  %cat4 = firrtl.cat %c0_ui2, %a, %c0_ui2 : (!firrtl.uint<2>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<8>
  %orr2 = firrtl.orr %cat4 : (!firrtl.uint<8>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out_orr, %orr2 : !firrtl.uint<1>

  // Test XorR with cat containing zero constants - should reduce to xorr(a)
  // CHECK: %[[XORR_A:.+]] = firrtl.xorr %a
  // CHECK: firrtl.matchingconnect %out_xorr, %[[XORR_A]]
  %cat5 = firrtl.cat %c0_ui2, %a : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<6>
  %xorr1 = firrtl.xorr %cat5 : (!firrtl.uint<6>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out_xorr, %xorr1 : !firrtl.uint<1>

  // Test XorR with cat containing non-zero constants - constants should remain
  // CHECK: %[[CAT_XOR:.+]] = firrtl.cat %c3_ui2, %a
  // CHECK: %[[XORR_CAT:.+]] = firrtl.xorr %[[CAT_XOR]]
  // CHECK: firrtl.matchingconnect %out_xorr, %[[XORR_CAT]]
  %cat6 = firrtl.cat %c3_ui2, %a, %c0_ui2 : (!firrtl.uint<2>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<8>
  %xorr2 = firrtl.xorr %cat6 : (!firrtl.uint<8>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out_xorr, %xorr2 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @FlattenCat
firrtl.module @FlattenCat(in %a: !firrtl.uint<2>, in %b: !firrtl.uint<3>, in %c: !firrtl.uint<4>,
                         in %d: !firrtl.uint<1>, in %e: !firrtl.uint<2>, in %f: !firrtl.uint<3>,
                         out %out1: !firrtl.uint<9>, out %out2: !firrtl.uint<15>,
                         out %out3: !firrtl.uint<10>) {
  // Test flattening nested cat operations - should flatten into single cat
  // CHECK: %[[FLAT_CAT:.+]] = firrtl.cat %a, %b, %c
  // CHECK: firrtl.matchingconnect %out1, %[[FLAT_CAT]]
  %cat1 = firrtl.cat %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<5>
  %cat2 = firrtl.cat %cat1, %c : (!firrtl.uint<5>, !firrtl.uint<4>) -> !firrtl.uint<9>
  firrtl.matchingconnect %out1, %cat2 : !firrtl.uint<9>

  // Test deep nesting (level 4) with multiple operands - should flatten completely
  // CHECK: %[[DEEP_FLAT:.+]] = firrtl.cat %a, %b, %c, %d, %e, %f
  // CHECK: firrtl.matchingconnect %out2, %[[DEEP_FLAT]]
  %inner1 = firrtl.cat %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<5>
  %inner2 = firrtl.cat %c, %d : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<5>
  %middle1 = firrtl.cat %inner1, %inner2 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %middle2 = firrtl.cat %e, %f : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<5>
  %outer = firrtl.cat %middle1, %middle2 : (!firrtl.uint<10>, !firrtl.uint<5>) -> !firrtl.uint<15>
  firrtl.matchingconnect %out2, %outer : !firrtl.uint<15>

  // CHECK: %[[ABAB:.+]] = firrtl.cat %a, %b, %a, %b
  // CHECK: firrtl.matchingconnect %out3, %[[ABAB]]
  %cat_a_b = firrtl.cat %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<5>
  %cat_a_b_a_b = firrtl.cat %cat_a_b, %cat_a_b : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>

  firrtl.matchingconnect %out3, %cat_a_b_a_b : !firrtl.uint<10>
}

// CHECK-LABEL: firrtl.module @FlattenCatEdgeCases
firrtl.module @FlattenCatEdgeCases(in %a: !firrtl.uint<4>, out %out1: !firrtl.uint<4>, out %out2: !firrtl.uint<8>) {
  %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>

  // Test single operand cat - should be replaced with the operand itself
  // CHECK: firrtl.matchingconnect %out1, %a
  %single_cat = firrtl.cat %a : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out1, %single_cat : !firrtl.uint<4>

  // Test flattening with zero-width operands - should eliminate zero-width operands
  // CHECK: firrtl.matchingconnect %out2, %[[ZERO_WIDTH_FLAT:.+]]
  %zero_cat1 = firrtl.cat %c0_ui0, %a : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<4>
  %zero_cat2 = firrtl.cat %zero_cat1, %c0_ui0, %a : (!firrtl.uint<4>, !firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<8>
  firrtl.matchingconnect %out2, %zero_cat2 : !firrtl.uint<8>
}

// Issue 8618
// CHECK-LABEL: firrtl.module @FlattenCatDifferentSigns
firrtl.module @FlattenCatDifferentSigns(in %a: !firrtl.uint<1>, in %b: !firrtl.sint<1>, in %c: !firrtl.sint<1>, out %d: !firrtl.uint<3>, out %e: !firrtl.uint<4>) {
  // CHECK-NEXT: %[[AS_UINT1:.+]] = firrtl.asUInt %b : (!firrtl.sint<1>) -> !firrtl.uint<1>
  // CHECK-NEXT: %[[AS_UINT2:.+]] = firrtl.asUInt %b : (!firrtl.sint<1>) -> !firrtl.uint<1>
  // CHECK-NEXT: %[[CAT1:.+]] = firrtl.cat %a, %[[AS_UINT1]], %[[AS_UINT2]] : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<3>
  // CHECK-NEXT: firrtl.matchingconnect %d, %[[CAT1]]
  %0 = firrtl.cat %b, %b : (!firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.uint<2>
  %1 = firrtl.cat %a, %0 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  firrtl.matchingconnect %d, %1 : !firrtl.uint<3>

  // CHECK-NEXT: %[[CAT2:.+]] = firrtl.cat %b, %c, %c, %b : (!firrtl.sint<1>, !firrtl.sint<1>, !firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %e, %[[CAT2]]
  %2 = firrtl.cat %b, %c : (!firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.uint<2>
  %3 = firrtl.cat %c, %b : (!firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.uint<2>
  %4 = firrtl.cat %2, %3 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.matchingconnect %e, %4 : !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @CatOfConstant
firrtl.module @CatOfConstant(in %a: !firrtl.uint<4>, out %out1: !firrtl.uint<12>,
                            out %out2: !firrtl.uint<10>) {
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %c5_ui3 = firrtl.constant 5 : !firrtl.uint<3>
  %c7_ui3 = firrtl.constant 7 : !firrtl.uint<3>

  // Test folding successive constants - should fold 3,5 into single constant
  // CHECK: %[[FOLDED_CONST2:.+]] = firrtl.constant 47 : !firrtl.uint<6>
  // CHECK: %[[FOLDED_CONST:.+]] = firrtl.constant 29 : !firrtl.uint<5>
  // CHECK: %[[CAT_FOLDED:.+]] = firrtl.cat %[[FOLDED_CONST]], %a, %c7_ui3
  // CHECK: firrtl.matchingconnect %out1, %[[CAT_FOLDED]]
  %cat1 = firrtl.cat %c3_ui2, %c5_ui3, %a, %c7_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<12>
  firrtl.matchingconnect %out1, %cat1 : !firrtl.uint<12>

  // Test folding all constants at the end
  // CHECK: %[[CAT_FOLDED2:.+]] = firrtl.cat %a, %[[FOLDED_CONST2]]
  // CHECK: firrtl.matchingconnect %out2, %[[CAT_FOLDED2]]
  %cat2 = firrtl.cat %a, %c5_ui3, %c7_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<10>
  firrtl.matchingconnect %out2, %cat2 : !firrtl.uint<10>
}

// CHECK-LABEL: firrtl.module @BitsOfCat
firrtl.module @BitsOfCat(in %a: !firrtl.uint<4>, in %b: !firrtl.uint<3>, in %c: !firrtl.uint<2>,
                        out %out1: !firrtl.uint<2>, out %out2: !firrtl.uint<1>, out %out3: !firrtl.uint<5>) {
  // CHECK: %[[CAT_MULTI:.+]] = firrtl.cat %a, %b, %c
  // Test bits extraction from cat that can be optimized to extract from single operand

  // bits[3:2] should extract from %b (bits 1:0 of %b)
  // CHECK: %[[BITS_B:.+]] = firrtl.bits %b 1 to 0
  // CHECK: firrtl.matchingconnect %out1, %[[BITS_B]]
  %cat1 = firrtl.cat %a, %b, %c : (!firrtl.uint<4>, !firrtl.uint<3>, !firrtl.uint<2>) -> !firrtl.uint<9>
  %bits1 = firrtl.bits %cat1 3 to 2 : (!firrtl.uint<9>) -> !firrtl.uint<2>
  firrtl.matchingconnect %out1, %bits1 : !firrtl.uint<2>

  // Test bits extraction from the most significant operand
  // bits[8:8] should extract from %a (bit 0 of %a)
  // CHECK: %[[BITS_A:.+]] = firrtl.bits %a 3 to 3
  // CHECK: firrtl.matchingconnect %out2, %[[BITS_A]]
  %bits2 = firrtl.bits %cat1 8 to 8 : (!firrtl.uint<9>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out2, %bits2 : !firrtl.uint<1>

  // Test bits extraction that spans multiple operands - should not be optimized
  // bits[5:1] spans from %b into %c, so should remain as-is
  // CHECK: %[[BITS_MULTI:.+]] = firrtl.bits %[[CAT_MULTI]] 5 to 1
  // CHECK: firrtl.matchingconnect %out3, %[[BITS_MULTI]]
  %bits3 = firrtl.bits %cat1 5 to 1 : (!firrtl.uint<9>) -> !firrtl.uint<5>
  firrtl.matchingconnect %out3, %bits3 : !firrtl.uint<5>
}

// CHECK-LABEL: firrtl.module @ComplexCatOptimization
firrtl.module @ComplexCatOptimization(in %x: !firrtl.uint<8>, in %y: !firrtl.uint<4>,
                                     out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<4>) {
  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>

  // Test complex pattern: reduction of cat with nested operations
  // The inner cat should be flattened, constants folded, then reduction applied
  %inner_cat = firrtl.cat %c15_ui4, %c0_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>
  %outer_cat = firrtl.cat %x, %inner_cat, %y : (!firrtl.uint<8>, !firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<20>

  // AndR should detect the zero constant and return 0
  // CHECK: firrtl.matchingconnect %out1, %c0_ui1
  %andr_result = firrtl.andr %outer_cat : (!firrtl.uint<20>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %andr_result : !firrtl.uint<1>

  // Test bits extraction from the optimized cat
  // Should extract directly from the appropriate operand
  %bits_result = firrtl.bits %outer_cat 7 to 4 : (!firrtl.uint<20>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out2, %bits_result : !firrtl.uint<4>
  // CHECK: firrtl.matchingconnect %out2, %c0_ui4
}


// CHECK-LABEL: firrtl.module @subaccess
firrtl.module @subaccess(out %result: !firrtl.uint<8>, in %vec0: !firrtl.vector<uint<8>, 16>) {
  // CHECK: [[TMP:%.+]] = firrtl.subindex %vec0[11]
  // CHECK-NEXT: firrtl.matchingconnect %result, [[TMP]]
  %c11_ui8 = firrtl.constant 11 : !firrtl.uint<8>
  %0 = firrtl.subaccess %vec0[%c11_ui8] : !firrtl.vector<uint<8>, 16>, !firrtl.uint<8>
  firrtl.connect %result, %0 :!firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: firrtl.subaccess
  %c16_ui8 = firrtl.constant 16 : !firrtl.uint<8>
  %1 = firrtl.subaccess %vec0[%c16_ui8] : !firrtl.vector<uint<8>, 16>, !firrtl.uint<8>
  firrtl.connect %result, %1 :!firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @subindex
firrtl.module @subindex(out %out : !firrtl.uint<8>) {
  // CHECK: %c8_ui8 = firrtl.constant 8 : !firrtl.uint<8>
  // CHECK: firrtl.matchingconnect %out, %c8_ui8 : !firrtl.uint<8>
  %0 = firrtl.aggregateconstant [8 : ui8] : !firrtl.vector<uint<8>, 1>
  %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<8>, 1>
  firrtl.matchingconnect %out, %1 : !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @subindex_agg
firrtl.module @subindex_agg(out %out : !firrtl.bundle<a: uint<8>>) {
  // CHECK: %0 = firrtl.aggregateconstant [8 : ui8] : !firrtl.bundle<a: uint<8>>
  // CHECK: firrtl.matchingconnect %out, %0 : !firrtl.bundle<a: uint<8>>
  %0 = firrtl.aggregateconstant [[8 : ui8]] : !firrtl.vector<bundle<a: uint<8>>, 1>
  %1 = firrtl.subindex %0[0] : !firrtl.vector<bundle<a: uint<8>>, 1>
  firrtl.matchingconnect %out, %1 : !firrtl.bundle<a: uint<8>>
}

// CHECK-LABEL: firrtl.module @subfield
firrtl.module @subfield(out %out : !firrtl.uint<8>) {
  // CHECK: %c8_ui8 = firrtl.constant 8 : !firrtl.uint<8>
  // CHECK: firrtl.matchingconnect %out, %c8_ui8 : !firrtl.uint<8>
  %0 = firrtl.aggregateconstant [8 : ui8] : !firrtl.bundle<a: uint<8>>
  %1 = firrtl.subfield %0[a] : !firrtl.bundle<a: uint<8>>
  firrtl.matchingconnect %out, %1 : !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @subfield_agg
firrtl.module @subfield_agg(out %out : !firrtl.vector<uint<8>, 1>) {
  // CHECK: %0 = firrtl.aggregateconstant [8 : ui8] : !firrtl.vector<uint<8>, 1>
  // CHECK: firrtl.matchingconnect %out, %0 : !firrtl.vector<uint<8>, 1>
  %0 = firrtl.aggregateconstant [[8 : ui8]] : !firrtl.bundle<a: vector<uint<8>, 1>>
  %1 = firrtl.subfield %0[a] : !firrtl.bundle<a: vector<uint<8>, 1>>
  firrtl.matchingconnect %out, %1 : !firrtl.vector<uint<8>, 1>
}

// CHECK-LABEL: firrtl.module @issue326
firrtl.module @issue326(out %tmp57: !firrtl.sint<1>) {
  %c29_si7 = firrtl.constant 29 : !firrtl.sint<7>
  %0 = firrtl.shr %c29_si7, 47 : (!firrtl.sint<7>) -> !firrtl.sint<1>
   // CHECK: c0_si1 = firrtl.constant 0 : !firrtl.sint<1>
   firrtl.connect %tmp57, %0 : !firrtl.sint<1>, !firrtl.sint<1>
}

// CHECK-LABEL: firrtl.module @issue331
firrtl.module @issue331(out %tmp81: !firrtl.sint<1>) {
  // CHECK: %c-1_si1 = firrtl.constant -1 : !firrtl.sint<1>
  %c-1_si1 = firrtl.constant -1 : !firrtl.sint<1>
  %0 = firrtl.shr %c-1_si1, 3 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  firrtl.connect %tmp81, %0 : !firrtl.sint<1>, !firrtl.sint<1>
}

// CHECK-LABEL: firrtl.module @issue432
firrtl.module @issue432(out %tmp8: !firrtl.uint<10>) {
  %c130_si10 = firrtl.constant 130 : !firrtl.sint<10>
  %0 = firrtl.tail %c130_si10, 0 : (!firrtl.sint<10>) -> !firrtl.uint<10>
  firrtl.connect %tmp8, %0 : !firrtl.uint<10>, !firrtl.uint<10>
  // CHECK-NEXT: %c130_ui10 = firrtl.constant 130 : !firrtl.uint<10>
  // CHECK-NEXT: firrtl.matchingconnect %tmp8, %c130_ui10
}

// CHECK-LABEL: firrtl.module @issue437
firrtl.module @issue437(out %tmp19: !firrtl.uint<1>) {
  // CHECK-NEXT: %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c-1_si1 = firrtl.constant -1 : !firrtl.sint<1>
  %0 = firrtl.bits %c-1_si1 0 to 0 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  firrtl.connect %tmp19, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @issue446
// CHECK-NEXT: [[TMP:%.+]] = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT: firrtl.matchingconnect %tmp10, [[TMP]] : !firrtl.uint<1>
firrtl.module @issue446(in %inp_1: !firrtl.sint<0>, out %tmp10: !firrtl.uint<1>) {
  %0 = firrtl.xor %inp_1, %inp_1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<0>
  firrtl.connect %tmp10, %0 : !firrtl.uint<1>, !firrtl.uint<0>
}

// CHECK-LABEL: firrtl.module @xorUnsized
// CHECK-NEXT: %c0_ui = firrtl.constant 0 : !firrtl.uint
firrtl.module @xorUnsized(in %inp_1: !firrtl.sint, out %tmp10: !firrtl.uint) {
  %0 = firrtl.xor %inp_1, %inp_1 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
  firrtl.connect %tmp10, %0 : !firrtl.uint, !firrtl.uint
}

// https://github.com/llvm/circt/issues/516
// CHECK-LABEL: @issue516
// CHECK-NEXT: [[TMP:%.+]] = firrtl.constant 0 : !firrtl.uint<0>
// CHECK-NEXT: firrtl.matchingconnect %tmp3, [[TMP]] : !firrtl.uint<0>
firrtl.module @issue516(in %inp_0: !firrtl.uint<0>, out %tmp3: !firrtl.uint<0>) {
  %0 = firrtl.div %inp_0, %inp_0 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %tmp3, %0 : !firrtl.uint<0>, !firrtl.uint<0>
}

// https://github.com/llvm/circt/issues/591
// CHECK-LABEL: @reg_cst_prop1
// CHECK-NEXT:   %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %c5_ui8 : !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop1(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  %_tmp_a = firrtl.reg droppable_name %clock {name = "_tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %tmp_b = firrtl.reg droppable_name %clock {name = "_tmp_b"} : !firrtl.clock, !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %tmp_b, %_tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %out_b, %tmp_b : !firrtl.uint<8>, !firrtl.uint<8>
}

// Check for DontTouch annotation
// CHECK-LABEL: @reg_cst_prop1_DontTouch
// CHECK-NEXT:      %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
// CHECK-NEXT:      %tmp_a = firrtl.reg sym @reg1 %clock : !firrtl.clock, !firrtl.uint<8>
// CHECK-NEXT:      %tmp_b = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<8>
// CHECK-NEXT:      firrtl.matchingconnect %tmp_a, %c5_ui8 : !firrtl.uint<8>
// CHECK-NEXT:      firrtl.matchingconnect %tmp_b, %tmp_a : !firrtl.uint<8>
// CHECK-NEXT:      firrtl.matchingconnect %out_b, %tmp_b : !firrtl.uint<8>

firrtl.module @reg_cst_prop1_DontTouch(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  %_tmp_a = firrtl.reg  sym @reg1 %clock {name = "tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %_tmp_b = firrtl.reg %clock {name = "tmp_b"} : !firrtl.clock, !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %_tmp_b, %_tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %out_b, %_tmp_b : !firrtl.uint<8>, !firrtl.uint<8>
}
// CHECK-LABEL: @reg_cst_prop2
// CHECK-NEXT:   %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %c5_ui8 : !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop2(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %_tmp_b = firrtl.reg droppable_name %clock {name = "_tmp_b"} : !firrtl.clock, !firrtl.uint<8>
  firrtl.connect %out_b, %_tmp_b : !firrtl.uint<8>, !firrtl.uint<8>

  %_tmp_a = firrtl.reg droppable_name %clock {name = "_tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %_tmp_b, %_tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: @reg_cst_prop3
// CHECK-NEXT:   %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %c0_ui8 : !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop3(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %_tmp_a = firrtl.reg droppable_name %clock {name = "_tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = firrtl.xor %_tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  firrtl.connect %out_b, %xor : !firrtl.uint<8>, !firrtl.uint<8>
}

// https://github.com/llvm/circt/issues/788

// CHECK-LABEL: @AttachMerge
firrtl.module @AttachMerge(in %a: !firrtl.analog<1>, in %b: !firrtl.analog<1>,
                           in %c: !firrtl.analog<1>) {
  // CHECK-NEXT: firrtl.attach %c, %b, %a :
  // CHECK-NEXT: }
  firrtl.attach %b, %a : !firrtl.analog<1>, !firrtl.analog<1>
  firrtl.attach %c, %b : !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachDeadWire
firrtl.module @AttachDeadWire(in %a: !firrtl.analog<1>, in %b: !firrtl.analog<1>) {
  // CHECK-NEXT: firrtl.attach %a, %b :
  // CHECK-NEXT: }
  %c = firrtl.wire  : !firrtl.analog<1>
  firrtl.attach %a, %b, %c : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachOpts
firrtl.module @AttachOpts(in %a: !firrtl.analog<1>) {
  // CHECK-NEXT: }
  %b = firrtl.wire  : !firrtl.analog<1>
  firrtl.attach %b, %a : !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachDeadWireDontTouch
firrtl.module @AttachDeadWireDontTouch(in %a: !firrtl.analog<1>, in %b: !firrtl.analog<1>) {
  // CHECK-NEXT: %c = firrtl.wire
  // CHECK-NEXT: firrtl.attach %a, %b, %c :
  // CHECK-NEXT: }
  %c = firrtl.wire sym @s1 : !firrtl.analog<1>
  firrtl.attach %a, %b, %c : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @wire_cst_prop1
// CHECK-NEXT:   %c10_ui9 = firrtl.constant 10 : !firrtl.uint<9>
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %c10_ui9 : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %_tmp_a = firrtl.wire droppable_name : !firrtl.uint<8>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = firrtl.add %_tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %xor : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @wire_port_prop1
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %in_a : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_port_prop1(in %in_a: !firrtl.uint<9>, out %out_b: !firrtl.uint<9>) {
  %_tmp = firrtl.wire droppable_name : !firrtl.uint<9>
  firrtl.connect %_tmp, %in_a : !firrtl.uint<9>, !firrtl.uint<9>

  firrtl.connect %out_b, %_tmp : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @LEQWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %e = firrtl.geq %a, %c42_ui
firrtl.module @LEQWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.leq %0, %a {name = "e"} : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @LTWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.gt %a, %c42_ui
firrtl.module @LTWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.lt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @GEQWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.leq %a, %c42_ui
firrtl.module @GEQWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.geq %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @GTWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.lt %a, %c42_ui
firrtl.module @GTWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.gt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @CompareWithSelf
firrtl.module @CompareWithSelf(
  in %a: !firrtl.uint,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant

  %0 = firrtl.leq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c1_ui1

  %1 = firrtl.lt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c0_ui1

  %2 = firrtl.geq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c1_ui1

  %3 = firrtl.gt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c0_ui1

  %4 = firrtl.eq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c1_ui1

  %5 = firrtl.neq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c0_ui1
}

// CHECK-LABEL: @LEQOutsideBounds
firrtl.module @LEQOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm5_si = firrtl.constant -5 : !firrtl.sint
  %cm6_si = firrtl.constant -6 : !firrtl.sint
  %c3_si = firrtl.constant 3 : !firrtl.sint
  %c4_si = firrtl.constant 4 : !firrtl.sint
  %c7_ui = firrtl.constant 7 : !firrtl.uint
  %c8_ui = firrtl.constant 8 : !firrtl.uint

  // a <= 7 -> 1
  // a <= 8 -> 1
  %0 = firrtl.leq %a, %c7_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.leq %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c1_ui1

  // b <= 3 -> 1
  // b <= 4 -> 1
  %2 = firrtl.leq %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.leq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c1_ui1

  // b <= -5 -> 0
  // b <= -6 -> 0
  %4 = firrtl.leq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.leq %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c0_ui1
}

// CHECK-LABEL: @LTOutsideBounds
firrtl.module @LTOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm4_si = firrtl.constant -4 : !firrtl.sint
  %cm5_si = firrtl.constant -5 : !firrtl.sint
  %c4_si = firrtl.constant 4 : !firrtl.sint
  %c5_si = firrtl.constant 5 : !firrtl.sint
  %c8_ui = firrtl.constant 8 : !firrtl.uint
  %c9_ui = firrtl.constant 9 : !firrtl.uint

  // a < 8 -> 1
  // a < 9 -> 1
  %0 = firrtl.lt %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.lt %a, %c9_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c1_ui1

  // b < 4 -> 1
  // b < 5 -> 1
  %2 = firrtl.lt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.lt %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c1_ui1

  // b < -4 -> 0
  // b < -5 -> 0
  %4 = firrtl.lt %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.lt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c0_ui1
}

// CHECK-LABEL: @GEQOutsideBounds
firrtl.module @GEQOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm4_si = firrtl.constant -4 : !firrtl.sint
  %cm5_si = firrtl.constant -5 : !firrtl.sint
  %c4_si = firrtl.constant 4 : !firrtl.sint
  %c5_si = firrtl.constant 5 : !firrtl.sint
  %c8_ui = firrtl.constant 8 : !firrtl.uint
  %c9_ui = firrtl.constant 9 : !firrtl.uint

  // a >= 8 -> 0
  // a >= 9 -> 0
  %0 = firrtl.geq %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.geq %a, %c9_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c0_ui1

  // b >= 4 -> 0
  // b >= 5 -> 0
  %2 = firrtl.geq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.geq %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c0_ui1

  // b >= -4 -> 1
  // b >= -5 -> 1
  %4 = firrtl.geq %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.geq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c1_ui1
}

// CHECK-LABEL: @GTOutsideBounds
firrtl.module @GTOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm5_si = firrtl.constant -5 : !firrtl.sint
  %cm6_si = firrtl.constant -6 : !firrtl.sint
  %c3_si = firrtl.constant 3 : !firrtl.sint
  %c4_si = firrtl.constant 4 : !firrtl.sint
  %c7_ui = firrtl.constant 7 : !firrtl.uint
  %c8_ui = firrtl.constant 8 : !firrtl.uint

  // a > 7 -> 0
  // a > 8 -> 0
  %0 = firrtl.gt %a, %c7_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.gt %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c0_ui1

  // b > 3 -> 0
  // b > 4 -> 0
  %2 = firrtl.gt %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.gt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c0_ui1

  // b > -5 -> 1
  // b > -6 -> 1
  %4 = firrtl.gt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.gt %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfDifferentWidths
firrtl.module @ComparisonOfDifferentWidths(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c3_si3 = firrtl.constant 3 : !firrtl.sint<3>
  %c4_si4 = firrtl.constant 4 : !firrtl.sint<4>
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %c4_ui3 = firrtl.constant 4 : !firrtl.uint<3>

  %0 = firrtl.leq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %1 = firrtl.leq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %2 = firrtl.lt %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %3 = firrtl.lt %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %4 = firrtl.geq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %5 = firrtl.geq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %6 = firrtl.gt %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %7 = firrtl.gt %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %8 = firrtl.eq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %9 = firrtl.eq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %10 = firrtl.neq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %11 = firrtl.neq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>

  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y11, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfUnsizedAndSized
firrtl.module @ComparisonOfUnsizedAndSized(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c3_si = firrtl.constant 3 : !firrtl.sint
  %c4_si4 = firrtl.constant 4 : !firrtl.sint<4>
  %c3_ui = firrtl.constant 3 : !firrtl.uint
  %c4_ui3 = firrtl.constant 4 : !firrtl.uint<3>

  %0 = firrtl.leq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %1 = firrtl.leq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %2 = firrtl.lt %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %3 = firrtl.lt %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %4 = firrtl.geq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %5 = firrtl.geq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %6 = firrtl.gt %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %7 = firrtl.gt %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %8 = firrtl.eq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %9 = firrtl.eq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %10 = firrtl.neq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %11 = firrtl.neq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>

  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y11, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfUnsized
firrtl.module @ComparisonOfUnsized(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c0_si = firrtl.constant 0 : !firrtl.sint
  %c4_si = firrtl.constant 4 : !firrtl.sint
  %c0_ui = firrtl.constant 0 : !firrtl.uint
  %c4_ui = firrtl.constant 4 : !firrtl.uint

  %0 = firrtl.leq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.leq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %2 = firrtl.lt %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %3 = firrtl.lt %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %4 = firrtl.geq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %5 = firrtl.geq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %6 = firrtl.gt %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %7 = firrtl.gt %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %8 = firrtl.eq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %9 = firrtl.eq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %10 = firrtl.neq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %11 = firrtl.neq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>

  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y11, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfZeroAndNonzeroWidths
firrtl.module @ComparisonOfZeroAndNonzeroWidths(
  in %xu: !firrtl.uint<0>,
  in %xs: !firrtl.sint<0>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>,
  out %y12: !firrtl.uint<1>,
  out %y13: !firrtl.uint<1>,
  out %y14: !firrtl.uint<1>,
  out %y15: !firrtl.uint<1>,
  out %y16: !firrtl.uint<1>,
  out %y17: !firrtl.uint<1>,
  out %y18: !firrtl.uint<1>,
  out %y19: !firrtl.uint<1>,
  out %y20: !firrtl.uint<1>,
  out %y21: !firrtl.uint<1>,
  out %y22: !firrtl.uint<1>,
  out %y23: !firrtl.uint<1>
) {
  %c0_si4 = firrtl.constant 0 : !firrtl.sint<4>
  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %c4_si4 = firrtl.constant 4 : !firrtl.sint<4>
  %c4_ui4 = firrtl.constant 4 : !firrtl.uint<4>

  %0 = firrtl.leq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %1 = firrtl.leq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %2 = firrtl.leq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %3 = firrtl.leq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %4 = firrtl.lt %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %5 = firrtl.lt %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %6 = firrtl.lt %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %7 = firrtl.lt %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %8 = firrtl.geq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %9 = firrtl.geq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %10 = firrtl.geq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %11 = firrtl.geq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %12 = firrtl.gt %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %13 = firrtl.gt %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %14 = firrtl.gt %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %15 = firrtl.gt %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %16 = firrtl.eq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %17 = firrtl.eq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %18 = firrtl.eq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %19 = firrtl.eq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %20 = firrtl.neq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %21 = firrtl.neq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %22 = firrtl.neq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %23 = firrtl.neq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>

  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y12, %12 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y13, %13 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y14, %14 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y15, %15 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y16, %16 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y17, %17 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y18, %18 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y19, %19 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y20, %20 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y21, %21 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y22, %22 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y23, %23 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %y0, %c1_ui1
  // CHECK: firrtl.matchingconnect %y1, %c1_ui1
  // CHECK: firrtl.matchingconnect %y2, %c1_ui1
  // CHECK: firrtl.matchingconnect %y3, %c1_ui1
  // CHECK: firrtl.matchingconnect %y4, %c0_ui1
  // CHECK: firrtl.matchingconnect %y5, %c1_ui1
  // CHECK: firrtl.matchingconnect %y6, %c0_ui1
  // CHECK: firrtl.matchingconnect %y7, %c1_ui1
  // CHECK: firrtl.matchingconnect %y8, %c1_ui1
  // CHECK: firrtl.matchingconnect %y9, %c0_ui1
  // CHECK: firrtl.matchingconnect %y10, %c1_ui1
  // CHECK: firrtl.matchingconnect %y11, %c0_ui1
  // CHECK: firrtl.matchingconnect %y12, %c0_ui1
  // CHECK: firrtl.matchingconnect %y13, %c0_ui1
  // CHECK: firrtl.matchingconnect %y14, %c0_ui1
  // CHECK: firrtl.matchingconnect %y15, %c0_ui1
  // CHECK: firrtl.matchingconnect %y16, %c1_ui1
  // CHECK: firrtl.matchingconnect %y17, %c0_ui1
  // CHECK: firrtl.matchingconnect %y18, %c1_ui1
  // CHECK: firrtl.matchingconnect %y19, %c0_ui1
  // CHECK: firrtl.matchingconnect %y20, %c0_ui1
  // CHECK: firrtl.matchingconnect %y21, %c1_ui1
  // CHECK: firrtl.matchingconnect %y22, %c0_ui1
  // CHECK: firrtl.matchingconnect %y23, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfZeroWidths
firrtl.module @ComparisonOfZeroWidths(
  in %xu0: !firrtl.uint<0>,
  in %xu1: !firrtl.uint<0>,
  in %xs0: !firrtl.sint<0>,
  in %xs1: !firrtl.sint<0>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>,
  out %y12: !firrtl.uint<1>,
  out %y13: !firrtl.uint<1>,
  out %y14: !firrtl.uint<1>,
  out %y15: !firrtl.uint<1>,
  out %y16: !firrtl.uint<1>,
  out %y17: !firrtl.uint<1>,
  out %y18: !firrtl.uint<1>,
  out %y19: !firrtl.uint<1>,
  out %y20: !firrtl.uint<1>,
  out %y21: !firrtl.uint<1>,
  out %y22: !firrtl.uint<1>,
  out %y23: !firrtl.uint<1>
) {
  %0 = firrtl.leq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %1 = firrtl.leq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %2 = firrtl.lt %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %3 = firrtl.lt %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %4 = firrtl.geq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %5 = firrtl.geq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %6 = firrtl.gt %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %7 = firrtl.gt %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %8 = firrtl.eq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %9 = firrtl.eq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %10 = firrtl.neq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %11 = firrtl.neq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>

  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %y0, %c1_ui1
  // CHECK: firrtl.matchingconnect %y1, %c1_ui1
  // CHECK: firrtl.matchingconnect %y2, %c0_ui1
  // CHECK: firrtl.matchingconnect %y3, %c0_ui1
  // CHECK: firrtl.matchingconnect %y4, %c1_ui1
  // CHECK: firrtl.matchingconnect %y5, %c1_ui1
  // CHECK: firrtl.matchingconnect %y6, %c0_ui1
  // CHECK: firrtl.matchingconnect %y7, %c0_ui1
  // CHECK: firrtl.matchingconnect %y8, %c1_ui1
  // CHECK: firrtl.matchingconnect %y9, %c1_ui1
  // CHECK: firrtl.matchingconnect %y10, %c0_ui1
  // CHECK: firrtl.matchingconnect %y11, %c0_ui1
}

// CHECK-LABEL: @ComparisonOfConsts
firrtl.module @ComparisonOfConsts(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>,
  out %y12: !firrtl.uint<1>,
  out %y13: !firrtl.uint<1>,
  out %y14: !firrtl.uint<1>,
  out %y15: !firrtl.uint<1>,
  out %y16: !firrtl.uint<1>,
  out %y17: !firrtl.uint<1>,
  out %y18: !firrtl.uint<1>,
  out %y19: !firrtl.uint<1>,
  out %y20: !firrtl.uint<1>,
  out %y21: !firrtl.uint<1>,
  out %y22: !firrtl.uint<1>,
  out %y23: !firrtl.uint<1>
) {
  %c0_si0 = firrtl.constant 0 : !firrtl.sint<0>
  %c2_si4 = firrtl.constant 2 : !firrtl.sint<4>
  %c-3_si3 = firrtl.constant -3 : !firrtl.sint<3>
  %c2_ui4 = firrtl.constant 2 : !firrtl.uint<4>
  %c5_ui3 = firrtl.constant 5 : !firrtl.uint<3>

  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant

  %0 = firrtl.leq %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %1 = firrtl.leq %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %2 = firrtl.leq %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %3 = firrtl.leq %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %4 = firrtl.leq %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  %5 = firrtl.lt %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %6 = firrtl.lt %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %7 = firrtl.lt %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %8 = firrtl.lt %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %9 = firrtl.lt %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  %10 = firrtl.geq %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %11 = firrtl.geq %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %12 = firrtl.geq %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %13 = firrtl.geq %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %14 = firrtl.geq %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  %15 = firrtl.gt %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %16 = firrtl.gt %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %17 = firrtl.gt %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %18 = firrtl.gt %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %19 = firrtl.gt %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  firrtl.connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>

  firrtl.connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y12, %12 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y13, %13 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y14, %14 : !firrtl.uint<1>, !firrtl.uint<1>

  firrtl.connect %y15, %15 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y16, %16 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y17, %17 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y18, %18 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y19, %19 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y3, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y4, %c0_ui1

  // CHECK-NEXT: firrtl.matchingconnect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y6, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y7, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y9, %c0_ui1

  // CHECK-NEXT: firrtl.matchingconnect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y11, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y12, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y13, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y14, %c1_ui1

  // CHECK-NEXT: firrtl.matchingconnect %y15, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y16, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y17, %c0_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y18, %c1_ui1
  // CHECK-NEXT: firrtl.matchingconnect %y19, %c1_ui1
}

// CHECK-LABEL: @zeroWidth(
// CHECK-NEXT:   %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   firrtl.matchingconnect %out, %c0_ui2 : !firrtl.uint<2>
// CHECK-NEXT:  }
firrtl.module @zeroWidth(out %out: !firrtl.uint<2>, in %in1 : !firrtl.uint<0>, in %in2 : !firrtl.uint<0>) {
  %add = firrtl.add %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %sub = firrtl.sub %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %mul = firrtl.mul %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %div = firrtl.div %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %rem = firrtl.rem %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshl = firrtl.dshl %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshlw = firrtl.dshlw %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshr = firrtl.dshr %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %and = firrtl.and %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %or = firrtl.or %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %xor = firrtl.xor %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %ret1 = firrtl.cat %add, %sub : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  %ret2 = firrtl.cat %ret1, %mul : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret3 = firrtl.cat %ret2, %div : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret4 = firrtl.cat %ret3, %rem : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret5 = firrtl.cat %ret4, %dshl : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret6 = firrtl.cat %ret5, %dshlw : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret7 = firrtl.cat %ret6, %dshr : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret8 = firrtl.cat %ret7, %and : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret9 = firrtl.cat %ret8, %or : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret10 = firrtl.cat %ret9, %xor : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  firrtl.matchingconnect %out, %ret10 : !firrtl.uint<2>
}

// CHECK-LABEL: @zeroWidthOperand(
// CHECK-NEXT:   %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
// CHECK-NEXT:   firrtl.matchingconnect %y6, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   firrtl.matchingconnect %y8, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   firrtl.matchingconnect %y9, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   firrtl.matchingconnect %y12, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   firrtl.matchingconnect %y14, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:  }
firrtl.module @zeroWidthOperand(
  in %in0 : !firrtl.uint<0>,
  in %in1 : !firrtl.uint<1>,
  out %y6: !firrtl.uint<0>,
  out %y8: !firrtl.uint<0>,
  out %y9: !firrtl.uint<0>,
  out %y12: !firrtl.uint<0>,
  out %y14: !firrtl.uint<0>
) {
  %div1 = firrtl.div %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  %rem1 = firrtl.rem %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  %rem2 = firrtl.rem %in1, %in0 : (!firrtl.uint<1>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshlw1 = firrtl.dshlw %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  %dshr1 = firrtl.dshr %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>

  firrtl.matchingconnect %y6, %div1 : !firrtl.uint<0>
  firrtl.matchingconnect %y8, %rem1 : !firrtl.uint<0>
  firrtl.matchingconnect %y9, %rem2 : !firrtl.uint<0>
  firrtl.matchingconnect %y12, %dshlw1 : !firrtl.uint<0>
  firrtl.matchingconnect %y14, %dshr1 : !firrtl.uint<0>
}

// CHECK-LABEL: @add_cst_prop1
// CHECK-NEXT:   %c11_ui9 = firrtl.constant 11 : !firrtl.uint<9>
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %c11_ui9 : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %c6_ui7 = firrtl.constant 6 : !firrtl.uint<7>
  %_tmp_a = firrtl.wire droppable_name : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.add %_tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %add : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @add_cst_prop2
// CHECK-NEXT:   %c-1_si9 = firrtl.constant -1 : !firrtl.sint<9>
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %c-1_si9 : !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop2(out %out_b: !firrtl.sint<9>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %_tmp_a = firrtl.wire droppable_name: !firrtl.sint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.sint<8>
  firrtl.connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.add %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  firrtl.connect %out_b, %add : !firrtl.sint<9>, !firrtl.sint<9>
}

// CHECK-LABEL: @add_cst_prop3
// CHECK-NEXT:   %c-2_si4 = firrtl.constant -2 : !firrtl.sint<4>
// CHECK-NEXT:   firrtl.matchingconnect %out_b, %c-2_si4 : !firrtl.sint<4>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop3(out %out_b: !firrtl.sint<4>) {
  %c1_si2 = firrtl.constant -1 : !firrtl.sint<2>
  %_tmp_a = firrtl.wire droppable_name : !firrtl.sint<2>
  %c1_si3 = firrtl.constant -1 : !firrtl.sint<3>
  firrtl.connect %_tmp_a, %c1_si2 : !firrtl.sint<2>, !firrtl.sint<2>
  %add = firrtl.add %_tmp_a, %c1_si3 : (!firrtl.sint<2>, !firrtl.sint<3>) -> !firrtl.sint<4>
  firrtl.connect %out_b, %add : !firrtl.sint<4>, !firrtl.sint<4>
}

// CHECK-LABEL: @add_cst_prop5
// CHECK: %[[pad:.+]] = firrtl.pad %tmp_a, 5
// CHECK-NEXT: firrtl.matchingconnect %out_b, %[[pad]]
// CHECK-NEXT: %[[pad:.+]] = firrtl.pad %tmp_a, 5
// CHECK-NEXT: firrtl.matchingconnect %out_b, %[[pad]]
firrtl.module @add_cst_prop5(out %out_b: !firrtl.uint<5>) {
  %tmp_a = firrtl.wire : !firrtl.uint<4>
  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %add = firrtl.add %tmp_a, %c0_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  firrtl.connect %out_b, %add : !firrtl.uint<5>, !firrtl.uint<5>
  %add2 = firrtl.add %c0_ui4, %tmp_a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  firrtl.connect %out_b, %add2 : !firrtl.uint<5>, !firrtl.uint<5>
}

// CHECK-LABEL: @add_double
// CHECK: %[[shl:.+]] = firrtl.shl %in, 1
// CHECK-NEXT: firrtl.matchingconnect %out, %[[shl]]
firrtl.module @add_double(out %out: !firrtl.uint<5>, in %in: !firrtl.uint<4>) {
  %add = firrtl.add %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  firrtl.connect %out, %add : !firrtl.uint<5>, !firrtl.uint<5>
}

// CHECK-LABEL: @add_narrow
// CHECK-NEXT: %[[add1:.+]] = firrtl.add %in2, %in1
// CHECK-NEXT: %[[pad1:.+]] = firrtl.pad %[[add1]], 7
// CHECK-NEXT: %[[add2:.+]] = firrtl.add %in2, %in1
// CHECK-NEXT: %[[pad2:.+]] = firrtl.pad %[[add2]], 7
// CHECK-NEXT: %[[add3:.+]] = firrtl.add %in1, %in2
// CHECK-NEXT: %[[pad3:.+]] = firrtl.pad %[[add3]], 7
// CHECK-NEXT: firrtl.matchingconnect %out1, %[[pad1]]
// CHECK-NEXT: firrtl.matchingconnect %out2, %[[pad2]]
// CHECK-NEXT: firrtl.matchingconnect %out3, %[[pad3]]
firrtl.module @add_narrow(out %out1: !firrtl.uint<7>, out %out2: !firrtl.uint<7>, out %out3: !firrtl.uint<7>, in %in1: !firrtl.uint<4>, in %in2: !firrtl.uint<2>) {
  %t1 = firrtl.pad %in1, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %t2 = firrtl.pad %in2, 6 : (!firrtl.uint<2>) -> !firrtl.uint<6>
  %add1 = firrtl.add %t1, %t2 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add2 = firrtl.add %in1, %t2 : (!firrtl.uint<4>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add3 = firrtl.add %t1, %in2 : (!firrtl.uint<6>, !firrtl.uint<2>) -> !firrtl.uint<7>
  firrtl.matchingconnect %out1, %add1 : !firrtl.uint<7>
  firrtl.matchingconnect %out2, %add2 : !firrtl.uint<7>
  firrtl.matchingconnect %out3, %add3 : !firrtl.uint<7>
}

// CHECK-LABEL: @adds_narrow
// CHECK-NEXT: %[[add1:.+]] = firrtl.add %in2, %in1
// CHECK-NEXT: %[[pad1:.+]] = firrtl.pad %[[add1]], 7
// CHECK-NEXT: %[[add2:.+]] = firrtl.add %in2, %in1
// CHECK-NEXT: %[[pad2:.+]] = firrtl.pad %[[add2]], 7
// CHECK-NEXT: %[[add3:.+]] = firrtl.add %in1, %in2
// CHECK-NEXT: %[[pad3:.+]] = firrtl.pad %[[add3]], 7
// CHECK-NEXT: firrtl.matchingconnect %out1, %[[pad1]]
// CHECK-NEXT: firrtl.matchingconnect %out2, %[[pad2]]
// CHECK-NEXT: firrtl.matchingconnect %out3, %[[pad3]]
firrtl.module @adds_narrow(out %out1: !firrtl.sint<7>, out %out2: !firrtl.sint<7>, out %out3: !firrtl.sint<7>, in %in1: !firrtl.sint<4>, in %in2: !firrtl.sint<2>) {
  %t1 = firrtl.pad %in1, 6 : (!firrtl.sint<4>) -> !firrtl.sint<6>
  %t2 = firrtl.pad %in2, 6 : (!firrtl.sint<2>) -> !firrtl.sint<6>
  %add1 = firrtl.add %t1, %t2 : (!firrtl.sint<6>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add2 = firrtl.add %in1, %t2 : (!firrtl.sint<4>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add3 = firrtl.add %t1, %in2 : (!firrtl.sint<6>, !firrtl.sint<2>) -> !firrtl.sint<7>
  firrtl.matchingconnect %out1, %add1 : !firrtl.sint<7>
  firrtl.matchingconnect %out2, %add2 : !firrtl.sint<7>
  firrtl.matchingconnect %out3, %add3 : !firrtl.sint<7>
}

// CHECK-LABEL: @sub_narrow
// CHECK-NEXT: %[[add1:.+]] = firrtl.sub %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
// CHECK-NEXT: %[[pad1:.+]] = firrtl.pad %[[add1]], 7 : (!firrtl.uint<5>) -> !firrtl.uint<7>
// CHECK-NEXT: %[[add2:.+]] = firrtl.sub %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
// CHECK-NEXT: %[[pad2:.+]] = firrtl.pad %[[add2]], 7 : (!firrtl.uint<5>) -> !firrtl.uint<7>
// CHECK-NEXT: %[[add3:.+]] = firrtl.sub %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
// CHECK-NEXT: %[[pad3:.+]] = firrtl.pad %[[add3]], 7 : (!firrtl.uint<5>) -> !firrtl.uint<7>
// CHECK-NEXT: firrtl.matchingconnect %out1, %[[pad1]]
// CHECK-NEXT: firrtl.matchingconnect %out2, %[[pad2]]
// CHECK-NEXT: firrtl.matchingconnect %out3, %[[pad3]]
firrtl.module @sub_narrow(out %out1: !firrtl.uint<7>, out %out2: !firrtl.uint<7>, out %out3: !firrtl.uint<7>, in %in1: !firrtl.uint<4>, in %in2: !firrtl.uint<2>) {
  %t1 = firrtl.pad %in1, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %t2 = firrtl.pad %in2, 6 : (!firrtl.uint<2>) -> !firrtl.uint<6>
  %add1 = firrtl.sub %t1, %t2 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add2 = firrtl.sub %in1, %t2 : (!firrtl.uint<4>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add3 = firrtl.sub %t1, %in2 : (!firrtl.uint<6>, !firrtl.uint<2>) -> !firrtl.uint<7>
  firrtl.matchingconnect %out1, %add1 : !firrtl.uint<7>
  firrtl.matchingconnect %out2, %add2 : !firrtl.uint<7>
  firrtl.matchingconnect %out3, %add3 : !firrtl.uint<7>
}

// CHECK-LABEL: @subs_narrow
// CHECK-NEXT: %[[add1:.+]] = firrtl.sub %in1, %in2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.sint<5>
// CHECK-NEXT: %[[pad1:.+]] = firrtl.pad %[[add1]], 7 : (!firrtl.sint<5>) -> !firrtl.sint<7>
// CHECK-NEXT: %[[add2:.+]] = firrtl.sub %in1, %in2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.sint<5>
// CHECK-NEXT: %[[pad2:.+]] = firrtl.pad %[[add2]], 7 : (!firrtl.sint<5>) -> !firrtl.sint<7>
// CHECK-NEXT: %[[add3:.+]] = firrtl.sub %in1, %in2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.sint<5>
// CHECK-NEXT: %[[pad3:.+]] = firrtl.pad %[[add3]], 7 : (!firrtl.sint<5>) -> !firrtl.sint<7>
// CHECK-NEXT: firrtl.matchingconnect %out1, %[[pad1]]
// CHECK-NEXT: firrtl.matchingconnect %out2, %[[pad2]]
// CHECK-NEXT: firrtl.matchingconnect %out3, %[[pad3]]
firrtl.module @subs_narrow(out %out1: !firrtl.sint<7>, out %out2: !firrtl.sint<7>, out %out3: !firrtl.sint<7>, in %in1: !firrtl.sint<4>, in %in2: !firrtl.sint<2>) {
  %t1 = firrtl.pad %in1, 6 : (!firrtl.sint<4>) -> !firrtl.sint<6>
  %t2 = firrtl.pad %in2, 6 : (!firrtl.sint<2>) -> !firrtl.sint<6>
  %add1 = firrtl.sub %t1, %t2 : (!firrtl.sint<6>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add2 = firrtl.sub %in1, %t2 : (!firrtl.sint<4>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add3 = firrtl.sub %t1, %in2 : (!firrtl.sint<6>, !firrtl.sint<2>) -> !firrtl.sint<7>
  firrtl.matchingconnect %out1, %add1 : !firrtl.sint<7>
  firrtl.matchingconnect %out2, %add2 : !firrtl.sint<7>
  firrtl.matchingconnect %out3, %add3 : !firrtl.sint<7>
}

// CHECK-LABEL: @sub_cst_prop1
// CHECK-NEXT:      %c1_ui9 = firrtl.constant 1 : !firrtl.uint<9>
// CHECK-NEXT:      firrtl.matchingconnect %out_b, %c1_ui9 : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %c6_ui7 = firrtl.constant 6 : !firrtl.uint<7>
  %_tmp_a = firrtl.wire droppable_name : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.sub %_tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %add : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @sub_cst_prop2
// CHECK-NEXT:      %c-11_si9 = firrtl.constant -11 : !firrtl.sint<9>
// CHECK-NEXT:      firrtl.matchingconnect %out_b, %c-11_si9 : !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop2(out %out_b: !firrtl.sint<9>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %_tmp_a = firrtl.wire droppable_name : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.sint<8>
  firrtl.connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.sub %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  firrtl.connect %out_b, %add : !firrtl.sint<9>, !firrtl.sint<9>
}

// CHECK-LABEL: @sub_double
// CHECK: %[[cst:.+]] = firrtl.constant 0 : !firrtl.uint<5>
// CHECK-NEXT: firrtl.matchingconnect %out, %[[cst]]
firrtl.module @sub_double(out %out: !firrtl.uint<5>, in %in: !firrtl.uint<4>) {
  %add = firrtl.sub %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  firrtl.connect %out, %add : !firrtl.uint<5>, !firrtl.uint<5>
}

// CHECK-LABEL: @mul_cst_prop1
// CHECK-NEXT:      %c30_ui15 = firrtl.constant 30 : !firrtl.uint<15>
// CHECK-NEXT:      firrtl.matchingconnect %out_b, %c30_ui15 : !firrtl.uint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop1(out %out_b: !firrtl.uint<15>) {
  %c6_ui7 = firrtl.constant 6 : !firrtl.uint<7>
  %_tmp_a = firrtl.wire droppable_name : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %_tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.mul %_tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<15>
  firrtl.connect %out_b, %add : !firrtl.uint<15>, !firrtl.uint<15>
}

// CHECK-LABEL: @mul_cst_prop2
// CHECK-NEXT:      %c-30_si15 = firrtl.constant -30 : !firrtl.sint<15>
// CHECK-NEXT:      firrtl.matchingconnect %out_b, %c-30_si15 : !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop2(out %out_b: !firrtl.sint<15>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %_tmp_a = firrtl.wire droppable_name : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.sint<8>
  firrtl.connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.mul %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  firrtl.connect %out_b, %add : !firrtl.sint<15>, !firrtl.sint<15>
}

// CHECK-LABEL: @mul_cst_prop3
// CHECK-NEXT:      %c30_si15 = firrtl.constant 30 : !firrtl.sint<15>
// CHECK-NEXT:      firrtl.matchingconnect %out_b, %c30_si15 : !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop3(out %out_b: !firrtl.sint<15>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %_tmp_a = firrtl.wire droppable_name : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant -5 : !firrtl.sint<8>
  firrtl.connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.mul %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  firrtl.connect %out_b, %add : !firrtl.sint<15>, !firrtl.sint<15>
}

// CHECK-LABEL: firrtl.module @MuxCanon
firrtl.module @MuxCanon(in %c1: !firrtl.uint<1>, in %c2: !firrtl.uint<1>, in %d1: !firrtl.uint<5>, in %d2: !firrtl.uint<5>, in %d3: !firrtl.uint<5>, out %foo: !firrtl.uint<5>, out %foo2: !firrtl.uint<5>, out %foo3: !firrtl.uint<5>, out %foo4: !firrtl.uint<5>, out %foo5: !firrtl.uint<10>, out %foo6: !firrtl.uint<10>) {
  %0 = firrtl.mux(%c1, %d2, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %1 = firrtl.mux(%c1, %d1, %0) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %2 = firrtl.mux(%c1, %0, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %3 = firrtl.mux(%c1, %d1, %d2) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %4 = firrtl.mux(%c2, %3, %d2) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %5 = firrtl.mux(%c2, %d1, %3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %6 = firrtl.cat %d1, %d2 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %7 = firrtl.cat %d1, %d3 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %8 = firrtl.cat %d1, %d2 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %9 = firrtl.cat %d3, %d2 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %10 = firrtl.mux(%c1, %6, %7) : (!firrtl.uint<1>, !firrtl.uint<10>, !firrtl.uint<10>) -> !firrtl.uint<10>
  %11 = firrtl.mux(%c2, %8, %9) : (!firrtl.uint<1>, !firrtl.uint<10>, !firrtl.uint<10>) -> !firrtl.uint<10>
  firrtl.connect %foo, %1 : !firrtl.uint<5>, !firrtl.uint<5>
  firrtl.connect %foo2, %2 : !firrtl.uint<5>, !firrtl.uint<5>
  firrtl.connect %foo3, %4 : !firrtl.uint<5>, !firrtl.uint<5>
  firrtl.connect %foo4, %5 : !firrtl.uint<5>, !firrtl.uint<5>
  firrtl.connect %foo5, %10 : !firrtl.uint<10>, !firrtl.uint<10>
  firrtl.connect %foo6, %11 : !firrtl.uint<10>, !firrtl.uint<10>
  // CHECK: firrtl.mux(%c1, %d1, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  // CHECK: firrtl.mux(%c1, %d2, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  // CHECK: %[[and:.*]] = firrtl.and %c2, %c1
  // CHECK: %[[andmux:.*]] = firrtl.mux(%[[and]], %d1, %d2)
  // CHECK: %[[or:.*]] = firrtl.or %c2, %c1
  // CHECK: %[[ormux:.*]] = firrtl.mux(%[[or]], %d1, %d2)
  // CHECK: %[[mux1:.*]] = firrtl.mux(%c1, %d2, %d3)
  // CHECK: firrtl.cat %d1, %[[mux1]]
  // CHECK: %[[mux2:.*]] = firrtl.mux(%c2, %d1, %d3)
  // CHECK: firrtl.cat %[[mux2]], %d2
}

// CHECK-LABEL: firrtl.module @MuxShorten
firrtl.module @MuxShorten(
  in %c1: !firrtl.uint<1>, in %c2: !firrtl.uint<1>,
  in %d1: !firrtl.uint<5>, in %d2: !firrtl.uint<5>,
  in %d3: !firrtl.uint<5>, in %d4: !firrtl.uint<5>,
  in %d5: !firrtl.uint<5>, in %d6: !firrtl.uint<5>,
  out %foo: !firrtl.uint<5>, out %foo2: !firrtl.uint<5>) {

  %0 = firrtl.mux(%c1, %d2, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %1 = firrtl.mux(%c2, %0, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %2 = firrtl.mux(%c1, %d4, %d5) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %3 = firrtl.mux(%c2, %2, %d6) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %11 = firrtl.mux(%c1, %1, %3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  firrtl.connect %foo, %11 : !firrtl.uint<5>, !firrtl.uint<5>
  firrtl.connect %foo2, %3 : !firrtl.uint<5>, !firrtl.uint<5>

  // CHECK: %[[n1:.*]] = firrtl.mux(%c2, %d2, %d1)
  // CHECK: %[[rem1:.*]] = firrtl.mux(%c1, %d4, %d5)
  // CHECK: %[[rem:.*]] = firrtl.mux(%c2, %[[rem1]], %d6)
  // CHECK: %[[n2:.*]] = firrtl.mux(%c2, %d5, %d6)
  // CHECK: %[[prim:.*]] = firrtl.mux(%c1, %[[n1]], %[[n2]])
  // CHECK: firrtl.matchingconnect %foo, %[[prim]]
  // CHECK: firrtl.matchingconnect %foo2, %[[rem]]
}


// CHECK-LABEL: firrtl.module @RegresetToReg
firrtl.module @RegresetToReg(in %clock: !firrtl.clock, in %dummy : !firrtl.uint<1>, out %foo1: !firrtl.uint<1>, out %foo2: !firrtl.uint<1>) {
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %zero_asyncreset = firrtl.asAsyncReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  %one_asyncreset = firrtl.asAsyncReset %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  // CHECK: %bar1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %foo2, %dummy : !firrtl.uint<1>
  %bar1 = firrtl.regreset %clock, %zero_asyncreset, %dummy : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  %bar2 = firrtl.regreset %clock, %one_asyncreset, %dummy : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

  firrtl.matchingconnect %bar2, %bar1 : !firrtl.uint<1> // Force a use to trigger a crash on a sink replacement

  firrtl.connect %foo1, %bar1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %foo2, %bar2 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @ForceableRegResetToNode
// Correctness, revisit if this is "valid" if forceable.
firrtl.module @ForceableRegResetToNode(in %clock: !firrtl.clock, in %dummy : !firrtl.uint<1>, out %foo: !firrtl.uint<1>, out %ref : !firrtl.rwprobe<uint<1>>) {
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %one_asyncreset = firrtl.asAsyncReset %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  // CHECK: %reg, %reg_ref = firrtl.node %dummy forceable : !firrtl.uint<1>
  %reg, %reg_f = firrtl.regreset %clock, %one_asyncreset, %dummy forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
  firrtl.ref.define %ref, %reg_f: !firrtl.rwprobe<uint<1>>

  firrtl.connect %reg, %dummy: !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %foo, %reg: !firrtl.uint<1>, !firrtl.uint<1>
}

// https://github.com/llvm/circt/issues/8348
// CHECK-LABEL: firrtl.module @RegResetInvalidResetValueType
// We cannot replace a regreset with its reset value, when the reset value's type does not match.
firrtl.module @RegResetInvalidResetValueType(in %c : !firrtl.clock, out %out : !firrtl.uint<2>) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %c1_asyncreset = firrtl.specialconstant 1 : !firrtl.asyncreset
  // CHECK: %reg = firrtl.regreset
  %reg = firrtl.regreset %c, %c1_asyncreset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<2>
  // CHECK: firrtl.matchingconnect %out, %reg : !firrtl.uint<2>
  firrtl.matchingconnect %out, %reg : !firrtl.uint<2>
  firrtl.matchingconnect %reg, %c0_ui2 : !firrtl.uint<2>
}

// https://github.com/llvm/circt/issues/929
// CHECK-LABEL: firrtl.module @MuxInvalidTypeOpt
firrtl.module @MuxInvalidTypeOpt(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<4>) {
  %c7_ui4 = firrtl.constant 7 : !firrtl.uint<4>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %0 = firrtl.mux (%in, %c7_ui4, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  %1 = firrtl.mux (%in, %c1_ui2, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}
// CHECK: firrtl.mux(%in, %c7_ui4, %c0_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK: firrtl.mux(%in, %c1_ui4, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

// CHECK-LABEL: firrtl.module @issue1100
// CHECK: firrtl.matchingconnect %tmp62, %c1_ui1
  firrtl.module @issue1100(out %tmp62: !firrtl.uint<1>) {
    %c-1_si2 = firrtl.constant -1 : !firrtl.sint<2>
    %0 = firrtl.orr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    firrtl.connect %tmp62, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }

// CHECK-LABEL: firrtl.module @zeroWidthMem
// CHECK-NEXT:  }
firrtl.module @zeroWidthMem(in %clock: !firrtl.clock) {
  // FIXME(Issue #1125): Add a test for zero width memory elimination.
}

// CHECK-LABEL: firrtl.module @issue1116
firrtl.module @issue1116(out %z: !firrtl.uint<1>) {
  %c844336_ui = firrtl.constant 844336 : !firrtl.uint
  %c161_ui8 = firrtl.constant 161 : !firrtl.uint<8>
  %0 = firrtl.leq %c844336_ui, %c161_ui8 : (!firrtl.uint, !firrtl.uint<8>) -> !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %z, %c0_ui1
  firrtl.connect %z, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// Sign casts must not be folded into unsized constants.
// CHECK-LABEL: firrtl.module @issue1118
firrtl.module @issue1118(out %z0: !firrtl.uint, out %z1: !firrtl.sint) {
  // CHECK: %0 = firrtl.asUInt %c4232_si : (!firrtl.sint) -> !firrtl.uint
  // CHECK: %1 = firrtl.asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
  // CHECK: firrtl.connect %z0, %0 : !firrtl.uint
  // CHECK: firrtl.connect %z1, %1 : !firrtl.sint
  %c4232_si = firrtl.constant 4232 : !firrtl.sint
  %c4232_ui = firrtl.constant 4232 : !firrtl.uint
  %0 = firrtl.asUInt %c4232_si : (!firrtl.sint) -> !firrtl.uint
  %1 = firrtl.asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
  firrtl.connect %z0, %0 : !firrtl.uint, !firrtl.uint
  firrtl.connect %z1, %1 : !firrtl.sint, !firrtl.sint
}

// CHECK-LABEL: firrtl.module @issue1139
firrtl.module @issue1139(out %z: !firrtl.uint<4>) {
  // CHECK-NEXT: %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %z, %c0_ui4 : !firrtl.uint<4>
  %c4_ui4 = firrtl.constant 4 : !firrtl.uint<4>
  %c674_ui = firrtl.constant 674 : !firrtl.uint
  %0 = firrtl.dshr %c4_ui4, %c674_ui : (!firrtl.uint<4>, !firrtl.uint) -> !firrtl.uint<4>
  firrtl.connect %z, %0 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @issue1142
firrtl.module @issue1142(in %cond: !firrtl.uint<1>, out %z: !firrtl.uint) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c42_ui = firrtl.constant 42 : !firrtl.uint
  %c43_ui = firrtl.constant 43 : !firrtl.uint

  // Don't fold away constant selects if widths are unknown.
  // CHECK: %0 = firrtl.mux(%c0_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %1 = firrtl.mux(%c1_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %0 = firrtl.mux(%c0_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %1 = firrtl.mux(%c1_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint

  // Don't fold nested muxes with same condition if widths are unknown.
  // CHECK: %2 = firrtl.mux(%cond, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %3 = firrtl.mux(%cond, %2, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %4 = firrtl.mux(%cond, %c42_ui, %2) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %2 = firrtl.mux(%cond, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %3 = firrtl.mux(%cond, %2, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %4 = firrtl.mux(%cond, %c42_ui, %2) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint

  firrtl.connect %z, %0 : !firrtl.uint, !firrtl.uint
  firrtl.connect %z, %1 : !firrtl.uint, !firrtl.uint
  firrtl.connect %z, %3 : !firrtl.uint, !firrtl.uint
  firrtl.connect %z, %4 : !firrtl.uint, !firrtl.uint
}

// CHECK-LABEL: firrtl.module @PadMuxOperands
firrtl.module @PadMuxOperands(
  in %cond: !firrtl.uint<1>,
  in %ui: !firrtl.uint,
  in %ui11: !firrtl.uint<11>,
  in %ui17: !firrtl.uint<17>,
  out %z: !firrtl.uint
) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

  // Smaller operand should pad to result width.
  // CHECK: %0 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %1 = firrtl.mux(%cond, %0, %ui17) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>) -> !firrtl.uint<17>
  // CHECK: %2 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %3 = firrtl.mux(%cond, %ui17, %2) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %0 = firrtl.mux(%cond, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %1 = firrtl.mux(%cond, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>

  // Unknown result width should prevent padding.
  // CHECK: %4 = firrtl.mux(%cond, %ui11, %ui) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint) -> !firrtl.uint
  // CHECK: %5 = firrtl.mux(%cond, %ui, %ui11) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint<11>) -> !firrtl.uint
  %2 = firrtl.mux(%cond, %ui11, %ui) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint) -> !firrtl.uint
  %3 = firrtl.mux(%cond, %ui, %ui11) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint<11>) -> !firrtl.uint

  // Padding to equal width operands should enable constant-select folds.
  // CHECK: %6 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %7 = firrtl.pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: firrtl.connect %z, %ui17 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: firrtl.connect %z, %6 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: firrtl.connect %z, %7 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: firrtl.connect %z, %ui17 : !firrtl.uint, !firrtl.uint<17>
  %4 = firrtl.mux(%c0_ui1, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %5 = firrtl.mux(%c0_ui1, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>
  %6 = firrtl.mux(%c1_ui1, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %7 = firrtl.mux(%c1_ui1, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>

  firrtl.connect %z, %0 : !firrtl.uint, !firrtl.uint<17>
  firrtl.connect %z, %1 : !firrtl.uint, !firrtl.uint<17>
  firrtl.connect %z, %2 : !firrtl.uint, !firrtl.uint
  firrtl.connect %z, %3 : !firrtl.uint, !firrtl.uint
  firrtl.connect %z, %4 : !firrtl.uint, !firrtl.uint<17>
  firrtl.connect %z, %5 : !firrtl.uint, !firrtl.uint<17>
  firrtl.connect %z, %6 : !firrtl.uint, !firrtl.uint<17>
  firrtl.connect %z, %7 : !firrtl.uint, !firrtl.uint<17>
}

// CHECK-LABEL: firrtl.module @regsyncreset
firrtl.module @regsyncreset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %foo : !firrtl.uint<2>, out %bar: !firrtl.uint<2>) attributes {firrtl.random_init_width = 2 : ui64} {
  // CHECK: %[[const:.*]] = firrtl.constant 1
  // CHECK-NEXT: firrtl.regreset %clock, %reset, %[[const]] {firrtl.random_init_end = 1 : ui64, firrtl.random_init_start = 0 : ui64}
  // CHECK-NEXT:  firrtl.matchingconnect %bar, %d : !firrtl.uint<2>
  // CHECK-NEXT:  firrtl.matchingconnect %d, %foo : !firrtl.uint<2>
  // CHECK-NEXT: }
  %d = firrtl.reg %clock {firrtl.random_init_end = 1 : ui64, firrtl.random_init_start = 0 : ui64} : !firrtl.clock, !firrtl.uint<2>
  firrtl.connect %bar, %d : !firrtl.uint<2>, !firrtl.uint<2>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %1 = firrtl.mux(%reset, %c1_ui2, %foo) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
  firrtl.connect %d, %1 : !firrtl.uint<2>, !firrtl.uint<2>
}

// CHECK-LABEL: firrtl.module @regsyncreset_no
firrtl.module @regsyncreset_no(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %foo : !firrtl.uint, out %bar: !firrtl.uint) {
  // CHECK: %[[const:.*]] = firrtl.constant 1
  // CHECK: firrtl.reg %clock
  // CHECK-NEXT:  firrtl.connect %bar, %d : !firrtl.uint
  // CHECK-NEXT:  %0 = firrtl.mux(%reset, %[[const]], %foo) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK-NEXT:  firrtl.connect %d, %0 : !firrtl.uint
  // CHECK-NEXT: }
  %d = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint
  firrtl.connect %bar, %d : !firrtl.uint, !firrtl.uint
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint
  %1 = firrtl.mux(%reset, %c1_ui2, %foo) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  firrtl.connect %d, %1 : !firrtl.uint, !firrtl.uint
}

// https://github.com/llvm/circt/issues/1215
// CHECK-LABEL: firrtl.module @dshifts_to_ishifts
firrtl.module @dshifts_to_ishifts(in %a_in: !firrtl.sint<58>,
                                  out %a_out: !firrtl.sint<58>,
                                  in %b_in: !firrtl.uint<8>,
                                  out %b_out: !firrtl.uint<23>,
                                  in %c_in: !firrtl.sint<58>,
                                  out %c_out: !firrtl.sint<58>) {
  // CHECK: %0 = firrtl.bits %a_in 57 to 4 : (!firrtl.sint<58>) -> !firrtl.uint<54>
  // CHECK: %1 = firrtl.asSInt %0 : (!firrtl.uint<54>) -> !firrtl.sint<54>
  // CHECK: %2 = firrtl.pad %1, 58 : (!firrtl.sint<54>) -> !firrtl.sint<58>
  // CHECK: firrtl.matchingconnect %a_out, %2 : !firrtl.sint<58>
  %c4_ui10 = firrtl.constant 4 : !firrtl.uint<10>
  %0 = firrtl.dshr %a_in, %c4_ui10 : (!firrtl.sint<58>, !firrtl.uint<10>) -> !firrtl.sint<58>
  firrtl.connect %a_out, %0 : !firrtl.sint<58>, !firrtl.sint<58>

  // CHECK: %3 = firrtl.shl %b_in, 4 : (!firrtl.uint<8>) -> !firrtl.uint<12>
  // CHECK: %4 = firrtl.pad %3, 23 : (!firrtl.uint<12>) -> !firrtl.uint<23>
  // CHECK: firrtl.matchingconnect %b_out, %4 : !firrtl.uint<23>
  %c4_ui4 = firrtl.constant 4 : !firrtl.uint<4>
  %1 = firrtl.dshl %b_in, %c4_ui4 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<23>
  firrtl.connect %b_out, %1 : !firrtl.uint<23>, !firrtl.uint<23>

  // CHECK: %5 = firrtl.bits %c_in 57 to 57 : (!firrtl.sint<58>) -> !firrtl.uint<1>
  // CHECK: %6 = firrtl.asSInt %5 : (!firrtl.uint<1>) -> !firrtl.sint<1>
  // CHECK: %7 = firrtl.pad %6, 58 : (!firrtl.sint<1>) -> !firrtl.sint<58>
  // CHECK: firrtl.matchingconnect %c_out, %7 : !firrtl.sint<58>
  %c438_ui10 = firrtl.constant 438 : !firrtl.uint<10>
  %2 = firrtl.dshr %c_in, %c438_ui10 : (!firrtl.sint<58>, !firrtl.uint<10>) -> !firrtl.sint<58>
  firrtl.connect %c_out, %2 : !firrtl.sint<58>, !firrtl.sint<58>
}

// CHECK-LABEL: firrtl.module @constReg
firrtl.module @constReg(in %clock: !firrtl.clock,
              in %en: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %r1 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %0 = firrtl.mux(%en, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %r1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:  %[[C11:.+]] = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK:  firrtl.matchingconnect %out, %[[C11]]
}

// CHECK-LABEL: firrtl.module @SingleConnectInWhen
firrtl.module @SingleConnectInWhen(in %p1: !firrtl.uint<8>, in %p2: !firrtl.uint<1>, out %out: !firrtl.uint<8>) {
  %w = firrtl.wire : !firrtl.uint<8>
  // CHECK: firrtl.when
  // CHECK-NEXT: firrtl.matchingconnect %w, %p1
  // CHECK: firrtl.matchingconnect %out, %w
  firrtl.when %p2 : !firrtl.uint<1> {
    firrtl.matchingconnect %w, %p1 : !firrtl.uint<8>
  }
  firrtl.matchingconnect %out, %w : !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @constReg
firrtl.module @constReg2(in %clock: !firrtl.clock,
              in %en: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %r1 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
  %r2 = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %0 = firrtl.mux(%en, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %1 = firrtl.mux(%en, %r2, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  %2 = firrtl.xor %r1, %r2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:  %[[C12:.+]] = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK:  firrtl.matchingconnect %out, %[[C12]]
}

// CHECK-LABEL: firrtl.module @constReg3
firrtl.module @constReg3(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = firrtl.constant 11 : !firrtl.uint<8>
  %r = firrtl.regreset %clock, %reset, %c11_ui8  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %0 = firrtl.mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  firrtl.connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK:  %[[C14:.+]] = firrtl.constant 11
  // CHECK: firrtl.matchingconnect %z, %[[C14]]
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @constReg4
firrtl.module @constReg4(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = firrtl.constant 11 : !firrtl.uint<8>
  %c11_ui4 = firrtl.constant 11 : !firrtl.uint<8>
  %r = firrtl.regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %0 = firrtl.mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  firrtl.connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK:  %[[C13:.+]] = firrtl.constant 11
  // CHECK: firrtl.matchingconnect %z, %[[C13]]
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @constReg6
firrtl.module @constReg6(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = firrtl.constant 11 : !firrtl.uint<8>
  %c11_ui4 = firrtl.constant 11 : !firrtl.uint<8>
  %resCond = firrtl.and %reset, %cond : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %r = firrtl.regreset %clock, %resCond, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %0 = firrtl.mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  firrtl.connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK:  %[[C13:.+]] = firrtl.constant 11
  // CHECK: firrtl.matchingconnect %z, %[[C13]]
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// Cannot optimize if bit mismatch with constant reset.
// CHECK-LABEL: firrtl.module @constReg5
firrtl.module @constReg5(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = firrtl.constant 11 : !firrtl.uint<8>
  %c11_ui4 = firrtl.constant 11 : !firrtl.uint<4>
  %r = firrtl.regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<8>
  // CHECK: %0 = firrtl.mux(%cond, %c11_ui8, %r)
  %0 = firrtl.mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  // CHECK: firrtl.matchingconnect %r, %0
  firrtl.connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// Should not crash when the reset value is a block argument.
firrtl.module @constReg7(in %v: !firrtl.uint<1>, in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
  %r = firrtl.regreset %clock, %reset, %v  : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<4>
}

// Check that firrtl.regreset reset mux folding doesn't respects
// DontTouchAnnotations or other annotations.
// CHECK-LABEL: firrtl.module @constReg8
firrtl.module @constReg8(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK: firrtl.regreset sym @s2
  %r1 = firrtl.regreset sym @s2 %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  %0 = firrtl.mux(%reset, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out1, %r1 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.regreset
  // CHECK-SAME: Foo
  %r2 = firrtl.regreset  %clock, %reset, %c1_ui1 {annotations = [{class = "Foo"}]} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  %1 = firrtl.mux(%reset, %c1_ui1, %r2) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out2, %r2 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHeck that a register that is only driven by constant and itself
// is canonicalized into a constant regardless of its reset signal.
// CHECK-LABEL: @constReg9
firrtl.module @constReg9(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %en_0: !firrtl.uint<1>, in %en_1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK-NOT: firrtl.reg
  // CHECK: firrtl.matchingconnect %out, %c0_ui1
  %r = firrtl.reg %clock {firrtl.random_init_start = 0 : ui64} : !firrtl.clock, !firrtl.uint<1>
  %0 = firrtl.and %en_0, %en_1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %1 = firrtl.mux(%0, %c0_ui1, %r) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.matchingconnect %out, %r : !firrtl.uint<1>
}

firrtl.module @BitCast(out %o:!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>> ) {
  %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
  %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<3>)
  %b2 = firrtl.bitcast %b : (!firrtl.uint<3>) -> (!firrtl.uint<3>)
  %c = firrtl.bitcast %b2 :  (!firrtl.uint<3>)-> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>)
  firrtl.connect %o, %c : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
  // CHECK: firrtl.matchingconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
}

// Check that we can create bundles directly
// CHECK-LABEL: firrtl.module @MergeBundle
firrtl.module @MergeBundle(out %o:!firrtl.bundle<valid: uint<1>, ready: uint<1>>, in %i:!firrtl.uint<1> ) {
  %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a0 = firrtl.subfield %a[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a1 = firrtl.subfield %a[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  firrtl.matchingconnect %a0, %i : !firrtl.uint<1>
  firrtl.matchingconnect %a1, %i : !firrtl.uint<1>
  firrtl.matchingconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  // CHECK: %0 = firrtl.bundlecreate %i, %i
  // CHECK-NEXT: firrtl.matchingconnect %a, %0
}

// Check that we can create vectors directly
// CHECK-LABEL: firrtl.module @MergeVector
firrtl.module @MergeVector(out %o:!firrtl.vector<uint<1>, 3>, in %i:!firrtl.uint<1> ) {
  %a = firrtl.wire : !firrtl.vector<uint<1>, 3>
  %a0 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 3>
  %a1 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 3>
  %a2 = firrtl.subindex %a[2] : !firrtl.vector<uint<1>, 3>
  firrtl.matchingconnect %a0, %i : !firrtl.uint<1>
  firrtl.matchingconnect %a1, %i : !firrtl.uint<1>
  firrtl.matchingconnect %a2, %i : !firrtl.uint<1>
  firrtl.matchingconnect %o, %a : !firrtl.vector<uint<1>, 3>
  // CHECK: %0 = firrtl.vectorcreate %i, %i, %i
  // CHECK-NEXT: firrtl.matchingconnect %a, %0
}

// Check that we can create vectors directly
// CHECK-LABEL: firrtl.module @MergeAgg
firrtl.module @MergeAgg(out %o: !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3> ) {
  %c = firrtl.constant 0 : !firrtl.uint<1>
  %a = firrtl.wire : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a0 = firrtl.subindex %a[0] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a1 = firrtl.subindex %a[1] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a2 = firrtl.subindex %a[2] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a00 = firrtl.subfield %a0[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a01 = firrtl.subfield %a0[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a10 = firrtl.subfield %a1[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a11 = firrtl.subfield %a1[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a20 = firrtl.subfield %a2[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a21 = firrtl.subfield %a2[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  firrtl.matchingconnect %a00, %c : !firrtl.uint<1>
  firrtl.matchingconnect %a01, %c : !firrtl.uint<1>
  firrtl.matchingconnect %a10, %c : !firrtl.uint<1>
  firrtl.matchingconnect %a11, %c : !firrtl.uint<1>
  firrtl.matchingconnect %a20, %c : !firrtl.uint<1>
  firrtl.matchingconnect %a21, %c : !firrtl.uint<1>
  firrtl.matchingconnect %o, %a :  !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK: [0 : ui1, 0 : ui1], [0 : ui1, 0 : ui1], [0 : ui1, 0 : ui1]] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK-NEXT: %a = firrtl.wire   : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK-NEXT: firrtl.matchingconnect %o, %a : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK-NEXT: firrtl.matchingconnect %a, %0 : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
}

// Don't collect connections in a when block.
// CHECK-LABEL: firrtl.module @DontMergeVector
firrtl.module @DontMergeVector(out %o:!firrtl.vector<uint<1>, 1>, in %i:!firrtl.uint<1> ) {
  %a = firrtl.wire : !firrtl.vector<uint<1>, 1>
  %0 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 1>
  firrtl.when %i : !firrtl.uint<1> {
    firrtl.connect %0, %i : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.matchingconnect %o, %a : !firrtl.vector<uint<1>, 1>
  // CHECK:      firrtl.when %i
  // CHECK-NEXT: firrtl.matchingconnect %0, %i
}

// TODO: Move to an appropriate place
// Issue #2197
// CHECK-LABEL: @Issue2197
firrtl.module @Issue2197(in %clock: !firrtl.clock, out %x: !firrtl.uint<2>) {
//  // COM: CHECK: [[ZERO:%.+]] = firrtl.constant 0 : !firrtl.uint<2>
//  // COM: CHECK-NEXT: firrtl.matchingconnect %x, [[ZERO]] : !firrtl.uint<2>
//  %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
//  %_reg = firrtl.reg droppable_name %clock : !firrtl.clock, !firrtl.uint<2>
//  %0 = firrtl.pad %invalid_ui1, 2 : (!firrtl.uint<1>) -> !firrtl.uint<2>
//  firrtl.connect %_reg, %0 : !firrtl.uint<2>, !firrtl.uint<2>
//  firrtl.connect %x, %_reg : !firrtl.uint<2>, !firrtl.uint<2>
}

// This is checking the behavior of sign extension of zero-width constants that
// results from trying to primops.
// CHECK-LABEL: @ZeroWidthAdd
firrtl.module @ZeroWidthAdd(out %a: !firrtl.sint<1>) {
  %zw = firrtl.constant 0 : !firrtl.sint<0>
  %0 = firrtl.constant 0 : !firrtl.sint<0>
  %1 = firrtl.add %0, %zw : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.sint<1>
  firrtl.connect %a, %1 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK:      %[[zero:.+]] = firrtl.constant 0 : !firrtl.sint<1>
  // CHECK-NEXT: firrtl.matchingconnect %a, %[[zero]]
}

// CHECK-LABEL: @ZeroWidthDshr
firrtl.module @ZeroWidthDshr(in %a: !firrtl.sint<0>, out %b: !firrtl.sint<0>) {
  %zw = firrtl.constant 0 : !firrtl.uint<0>
  %0 = firrtl.dshr %a, %zw : (!firrtl.sint<0>, !firrtl.uint<0>) -> !firrtl.sint<0>
  firrtl.connect %b, %0 : !firrtl.sint<0>, !firrtl.sint<0>
  // CHECK:      %[[zero:.+]] = firrtl.constant 0 : !firrtl.sint<0>
  // CHECK-NEXT: firrtl.matchingconnect %b, %[[zero]]
}

// CHECK-LABEL: @ZeroWidthPad
firrtl.module @ZeroWidthPad(out %b: !firrtl.sint<1>) {
  %zw = firrtl.constant 0 : !firrtl.sint<0>
  %0 = firrtl.pad %zw, 1 : (!firrtl.sint<0>) -> !firrtl.sint<1>
  firrtl.connect %b, %0 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK:      %[[zero:.+]] = firrtl.constant 0 : !firrtl.sint<1>
  // CHECK-NEXT: firrtl.matchingconnect %b, %[[zero]]
}

// CHECK-LABEL: @ZeroWidthCat
firrtl.module @ZeroWidthCat(out %a: !firrtl.uint<1>) {
  %one = firrtl.constant 1 : !firrtl.uint<1>
  %zw = firrtl.constant 0 : !firrtl.uint<0>
  %0 = firrtl.cat %one, %zw : (!firrtl.uint<1>, !firrtl.uint<0>) -> !firrtl.uint<1>
  firrtl.connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:      %[[one:.+]] = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.matchingconnect %a, %[[one]]
}

//TODO: Move to an appropriate place
// Issue mentioned in PR #2251
// CHECK-LABEL: @Issue2251
firrtl.module @Issue2251(out %o: !firrtl.sint<15>) {
//  // pad used to always return an unsigned constant
//  %invalid_si1 = firrtl.invalidvalue : !firrtl.sint<1>
//  %0 = firrtl.pad %invalid_si1, 15 : (!firrtl.sint<1>) -> !firrtl.sint<15>
//  firrtl.connect %o, %0 : !firrtl.sint<15>, !firrtl.sint<15>
//  // COM: CHECK:      %[[zero:.+]] = firrtl.constant 0 : !firrtl.sint<15>
//  // COM: CHECK-NEXT: firrtl.matchingconnect %o, %[[zero]]
}

// Issue mentioned in #2289
// CHECK-LABEL: @Issue2289
firrtl.module @Issue2289(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<5>) {
  %r = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
  firrtl.connect %r, %r : !firrtl.uint<1>, !firrtl.uint<1>
  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %0 = firrtl.dshl %c1_ui1, %r : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  %1 = firrtl.sub %c0_ui4, %0 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
  firrtl.connect %out, %1 : !firrtl.uint<5>, !firrtl.uint<5>
  // CHECK:      %[[dshl:.+]] = firrtl.dshl
  // CHECK-NEXT: %[[neg:.+]] = firrtl.neg %[[dshl]]
  // CHECK-NEXT: %[[pad:.+]] = firrtl.pad %[[neg]], 5
  // CHECK-NEXT: %[[cast:.+]] = firrtl.asUInt %[[pad]]
  // CHECK-NEXT: firrtl.matchingconnect %out, %[[cast]]
}

// Issue mentioned in #2291
// CHECK-LABEL: @Issue2291
firrtl.module @Issue2291(out %out: !firrtl.uint<1>) {
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %clock = firrtl.asClock %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
  %0 = firrtl.asUInt %clock : (!firrtl.clock) -> !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// Crasher from issue #3043
// CHECK-LABEL: @Issue3043
firrtl.module @Issue3043(out %a: !firrtl.vector<uint<5>, 3>) {
  %_b = firrtl.wire  : !firrtl.vector<uint<5>, 3>
  %b = firrtl.node sym @b %_b  : !firrtl.vector<uint<5>, 3>
  %invalid = firrtl.invalidvalue : !firrtl.vector<uint<5>, 3>
  firrtl.matchingconnect %_b, %invalid : !firrtl.vector<uint<5>, 3>
  firrtl.connect %a, %_b : !firrtl.vector<uint<5>, 3>, !firrtl.vector<uint<5>, 3>
}

// Test behaviors folding with zero-width constants, issue #2514.
// CHECK-LABEL: @Issue2514
firrtl.module @Issue2514(
  in %s: !firrtl.sint<0>,
  in %u: !firrtl.uint<0>,
  out %geq_0: !firrtl.uint<1>,
  out %geq_1: !firrtl.uint<1>,
  out %geq_2: !firrtl.uint<1>,
  out %geq_3: !firrtl.uint<1>,
  out %gt_0:  !firrtl.uint<1>,
  out %gt_1:  !firrtl.uint<1>,
  out %gt_2:  !firrtl.uint<1>,
  out %gt_3:  !firrtl.uint<1>,
  out %lt_0:  !firrtl.uint<1>,
  out %lt_1:  !firrtl.uint<1>,
  out %lt_2:  !firrtl.uint<1>,
  out %lt_3:  !firrtl.uint<1>,
  out %leq_0: !firrtl.uint<1>,
  out %leq_1: !firrtl.uint<1>,
  out %leq_2: !firrtl.uint<1>,
  out %leq_3: !firrtl.uint<1>
) {
  %t = firrtl.constant 0: !firrtl.sint<0>
  %v = firrtl.constant 0: !firrtl.uint<0>

  // CHECK-DAG: %[[zero_i1:.+]] = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK-DAG: %[[one_i1:.+]] = firrtl.constant 1 : !firrtl.uint<1>

  // geq(x, y) -> 1 when x and y are both zero-width (and here, one is a constant)
  %3 = firrtl.geq %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %4 = firrtl.geq %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %5 = firrtl.geq %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %6 = firrtl.geq %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  firrtl.matchingconnect %geq_0, %3 : !firrtl.uint<1>
  firrtl.matchingconnect %geq_1, %4 : !firrtl.uint<1>
  firrtl.matchingconnect %geq_2, %5 : !firrtl.uint<1>
  firrtl.matchingconnect %geq_3, %6 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %geq_0, %[[one_i1]]
  // CHECK: firrtl.matchingconnect %geq_1, %[[one_i1]]
  // CHECK: firrtl.matchingconnect %geq_2, %[[one_i1]]
  // CHECK: firrtl.matchingconnect %geq_3, %[[one_i1]]

  // gt(x, y) -> 0 when x and y are both zero-width (and here, one is a constant)
  %7 = firrtl.gt %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %8 = firrtl.gt %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %9 = firrtl.gt %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %10 = firrtl.gt %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  firrtl.matchingconnect %gt_0, %7 : !firrtl.uint<1>
  firrtl.matchingconnect %gt_1, %8 : !firrtl.uint<1>
  firrtl.matchingconnect %gt_2, %9 : !firrtl.uint<1>
  firrtl.matchingconnect %gt_3, %10 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %gt_0, %[[zero_i1]]
  // CHECK: firrtl.matchingconnect %gt_1, %[[zero_i1]]
  // CHECK: firrtl.matchingconnect %gt_2, %[[zero_i1]]
  // CHECK: firrtl.matchingconnect %gt_3, %[[zero_i1]]

  // lt(x, y) -> 0 when x and y are both zero-width (and here, one is a constant)
  %11 = firrtl.lt %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %12 = firrtl.lt %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %13 = firrtl.lt %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %14 = firrtl.lt %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  firrtl.matchingconnect %lt_0, %11 : !firrtl.uint<1>
  firrtl.matchingconnect %lt_1, %12 : !firrtl.uint<1>
  firrtl.matchingconnect %lt_2, %13 : !firrtl.uint<1>
  firrtl.matchingconnect %lt_3, %14 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %lt_0, %[[zero_i1]]
  // CHECK: firrtl.matchingconnect %lt_1, %[[zero_i1]]
  // CHECK: firrtl.matchingconnect %lt_2, %[[zero_i1]]
  // CHECK: firrtl.matchingconnect %lt_3, %[[zero_i1]]

  // leq(x, y) -> 1 when x and y are both zero-width (and here, one is a constant)
  %15 = firrtl.leq %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %16 = firrtl.leq %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %17 = firrtl.leq %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %18 = firrtl.leq %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  firrtl.matchingconnect %leq_0, %15 : !firrtl.uint<1>
  firrtl.matchingconnect %leq_1, %16 : !firrtl.uint<1>
  firrtl.matchingconnect %leq_2, %17 : !firrtl.uint<1>
  firrtl.matchingconnect %leq_3, %18 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %leq_0, %[[one_i1]]
  // CHECK: firrtl.matchingconnect %leq_1, %[[one_i1]]
  // CHECK: firrtl.matchingconnect %leq_2, %[[one_i1]]
  // CHECK: firrtl.matchingconnect %leq_3, %[[one_i1]]
}

// CHECK-LABEL: @NamePropagation
firrtl.module @NamePropagation(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, in %c: !firrtl.uint<4>, out %res1: !firrtl.uint<2>, out %res2: !firrtl.uint<2>) {
  // CHECK-NEXT: %e = firrtl.bits %c 1 to 0 {name = "e"}
  %1 = firrtl.bits %c 2 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  %e = firrtl.bits %1 1 to 0 {name = "e"}: (!firrtl.uint<3>) -> !firrtl.uint<2>
  // CHECK-NEXT: firrtl.matchingconnect %res1, %e
  firrtl.matchingconnect %res1, %e : !firrtl.uint<2>

  // CHECK-NEXT: %name_node = firrtl.not %e {name = "name_node"} : (!firrtl.uint<2>) -> !firrtl.uint<2>
  // CHECK-NEXT: firrtl.matchingconnect %res2, %name_node
  %2 = firrtl.not %e : (!firrtl.uint<2>) -> !firrtl.uint<2>
  %name_node = firrtl.node droppable_name %2 : !firrtl.uint<2>
  firrtl.matchingconnect %res2, %name_node : !firrtl.uint<2>
}

// Issue 3319: https://github.com/llvm/circt/issues/3319
// CHECK-LABEL: @Foo3319
firrtl.module @Foo3319(in %i: !firrtl.uint<1>, out %o : !firrtl.uint<1>) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %0 = firrtl.and %c0_ui1, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %n = firrtl.node interesting_name %c0_ui1
  %n = firrtl.node interesting_name %0  : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %o, %n
  firrtl.matchingconnect %o, %n : !firrtl.uint<1>
}

// CHECK-LABEL: @WireByPass
firrtl.module @WireByPass(in %i: !firrtl.uint<1>, out %o : !firrtl.uint<1>) {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %n = firrtl.wire interesting_name : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %n, %c0_ui1
  firrtl.matchingconnect %n, %c0_ui1 : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %o, %n
  firrtl.matchingconnect %o, %n : !firrtl.uint<1>
}

// Check that canonicalizeSingleSetConnect doesn't remove a wire with an
// Annotation on it.
//
// CHECK-LABEL: @AnnotationsBlockRemoval
firrtl.module @AnnotationsBlockRemoval(
  in %a: !firrtl.uint<1>,
  out %b: !firrtl.uint<1>
) {
  // CHECK: %w = firrtl.wire
  %w = firrtl.wire droppable_name {annotations = [{class = "Foo"}]} : !firrtl.uint<1>
  firrtl.matchingconnect %w, %a : !firrtl.uint<1>
  firrtl.matchingconnect %b, %w : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Verification
firrtl.module @Verification(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %o : !firrtl.uint<1>) {
  %c0 = firrtl.constant 0 : !firrtl.uint<1>
  %c1 = firrtl.constant 1 : !firrtl.uint<1>

  // Never enabled.
  // CHECK-NOT: firrtl.assert
  firrtl.assert %clock, %p, %c0, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.assume
  firrtl.assume %clock, %p, %c0, "assume0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.int.unclocked_assume
  firrtl.int.unclocked_assume %p, %c0, "assume_edged0" : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.cover
  firrtl.cover %clock, %p, %c0, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>

  // Never fired.
  // CHECK-NOT: firrtl.assert
  firrtl.assert %clock, %c1, %p, "assert1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.assume
  firrtl.assume %clock, %c1, %p, "assume1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.int.unclocked_assume
  firrtl.int.unclocked_assume %c1, %p, "assume_edged1" : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.cover
  firrtl.cover %clock, %c0, %p, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.int.isX
  %x = firrtl.int.isX %c0 : !firrtl.uint<1>
  firrtl.matchingconnect %o, %x : !firrtl.uint<1>

  // Never fired when enabled.
  // CHECK-NOT: firrtl.assert
  firrtl.assert %clock, %p, %p, "assert1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.assume
  firrtl.assume %clock, %p, %p, "assume1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: firrtl.int.unclocked_assume
  firrtl.int.unclocked_assume %p, %p, "assume_edged1" : !firrtl.uint<1>, !firrtl.uint<1>
}

// COMMON-LABEL:  firrtl.module @MultibitMux
// COMMON-NEXT:      %0 = firrtl.subaccess %a[%sel] : !firrtl.vector<uint<1>, 3>, !firrtl.uint<2>
// COMMON-NEXT:      firrtl.matchingconnect %b, %0 : !firrtl.uint<1>
firrtl.module @MultibitMux(in %a: !firrtl.vector<uint<1>, 3>, in %sel: !firrtl.uint<2>, out %b: !firrtl.uint<1>) {
  %0 = firrtl.subindex %a[2] : !firrtl.vector<uint<1>, 3>
  %1 = firrtl.subindex %a[1] : !firrtl.vector<uint<1>, 3>
  %2 = firrtl.subindex %a[0] : !firrtl.vector<uint<1>, 3>
  %3 = firrtl.multibit_mux %sel, %0, %1, %2 : !firrtl.uint<2>, !firrtl.uint<1>
  firrtl.matchingconnect %b, %3 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @NameProp
firrtl.module @NameProp(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %0 = firrtl.or %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %_useless_name_1 = firrtl.node  %0  : !firrtl.uint<1>
  %useful_name = firrtl.node %_useless_name_1  : !firrtl.uint<1>
  %_useless_name_2 = firrtl.node  %useful_name  : !firrtl.uint<1>
  // CHECK-NEXT: %useful_name = firrtl.or %in0, %in1
  // CHECK-NEXT: firrtl.matchingconnect %out, %useful_name
  firrtl.matchingconnect %out, %_useless_name_2 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @CrashAllUnusedPorts
firrtl.module @CrashAllUnusedPorts() {
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %foo, %bar = firrtl.mem  Undefined  {depth = 3 : i64, groupID = 4 : ui32, name = "whatever", portNames = ["MPORT_1", "MPORT_5"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<2>, mask: uint<1>>, !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<2>>
  %26 = firrtl.subfield %foo[en] : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<2>, mask: uint<1>>
  firrtl.matchingconnect %26, %c0_ui1 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Issue6237
// CHECK-NEXT:    %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
// CHECK-NEXT:    firrtl.matchingconnect %out, %c0_ui0 : !firrtl.uint<0>
firrtl.module @Issue6237(out %out: !firrtl.uint<0>) {
  %foo, %bar = firrtl.mem  Undefined  {depth = 3 : i64, groupID = 4 : ui32, name = "whatever", portNames = ["MPORT_1", "MPORT_5"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>, !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<0>>
  %a = firrtl.subfield %bar[data] : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<0>>
  firrtl.matchingconnect %out, %a : !firrtl.uint<0>
}


// CHECK-LABEL: firrtl.module @CrashRegResetWithOneReset
firrtl.module @CrashRegResetWithOneReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_d: !firrtl.uint<1>, out %io_q: !firrtl.uint<1>, in %io_en: !firrtl.uint<1>) {
  %c1_asyncreset = firrtl.specialconstant 1 : !firrtl.asyncreset
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %reg = firrtl.regreset  %clock, %c1_asyncreset, %c0_ui1  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  %0 = firrtl.mux(%io_en, %io_d, %reg) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %reg, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %io_q, %reg : !firrtl.uint<1>, !firrtl.uint<1>
}

// A read-only memory with memory initialization should not be removed.
// CHECK-LABEL: firrtl.module @ReadOnlyFileInitialized
firrtl.module @ReadOnlyFileInitialized(
  in %clock: !firrtl.clock,
  in %reset: !firrtl.uint<1>,
  in %read_en: !firrtl.uint<1>,
  out %read_data: !firrtl.uint<8>,
  in %read_addr: !firrtl.uint<5>
) {
  // CHECK-NEXT: firrtl.mem
  // CHECK-SAME:   name = "withInit"
  %m_r = firrtl.mem Undefined {
    depth = 32 : i64,
    groupID = 1 : ui32,
    init = #firrtl.meminit<"mem1.hex.txt", false, true>,
    name = "withInit",
    portNames = ["m_r"],
    readLatency = 1 : i32,
    writeLatency = 1 : i32
  } : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %0 = firrtl.subfield %m_r[addr] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %1 = firrtl.subfield %m_r[en] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %2 = firrtl.subfield %m_r[clk] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %3 = firrtl.subfield %m_r[data] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  firrtl.matchingconnect %0, %read_addr : !firrtl.uint<5>
  firrtl.matchingconnect %1, %read_en : !firrtl.uint<1>
  firrtl.matchingconnect %2, %clock : !firrtl.clock
  firrtl.matchingconnect %read_data, %3 : !firrtl.uint<8>
}

// CHECK-LABEL: @MuxCondWidth
firrtl.module @MuxCondWidth(in %cond: !firrtl.uint<1>, out %foo: !firrtl.uint<3>) {
  // Don't canonicalize if the type is not UInt<1>
  // CHECK: %0 = firrtl.mux(%cond, %c0_ui3, %c1_ui3) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
  // CHECK-NEXT:  firrtl.matchingconnect %foo, %0
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui3 = firrtl.constant 1 : !firrtl.uint<3>
  %0 = firrtl.mux(%cond, %c0_ui1, %c1_ui3) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<3>
  firrtl.matchingconnect %foo, %0 : !firrtl.uint<3>
}

// CHECK-LABEL: @MuxEQ
firrtl.module @MuxEQ(in %a: !firrtl.uint<4>,
                     in %b: !firrtl.uint<4>,
                     in %c: !firrtl.uint<4>,
                     out %out1: !firrtl.uint<4>,
                     out %out2: !firrtl.uint<4>,
                     out %out3: !firrtl.uint<4>,
                     out %out4: !firrtl.uint<4>,
                     out %out5: !firrtl.uint<4>) {
  %eq = firrtl.eq %a, %b : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %0 = firrtl.mux (%eq, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %out1, %b
  firrtl.matchingconnect %out1, %0 : !firrtl.uint<4>

  %eq_swapped = firrtl.eq %b, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %1 = firrtl.mux (%eq_swapped, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %out2, %b
  firrtl.matchingconnect %out2, %1 : !firrtl.uint<4>

  %neq = firrtl.neq %a, %b : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %2 = firrtl.mux (%neq, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %out3, %a
  firrtl.matchingconnect %out3, %2 : !firrtl.uint<4>

  %neq_swapped = firrtl.neq %b, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %3 = firrtl.mux (%neq_swapped, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %out4, %a
  firrtl.matchingconnect %out4, %3 : !firrtl.uint<4>

  // CHECK-NEXT: [[EQ:%.+]] = firrtl.eq %a, %b : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK-NEXT: [[MUX:%.+]] = firrtl.mux([[EQ]], %c, %a) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.matchingconnect %out5, [[MUX]]
  %4 = firrtl.mux (%neq, %a, %c) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out5, %4 : !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @RemoveUnusedInvalid
firrtl.module @RemoveUnusedInvalid() {
  // CHECK-NOT: firrtl.invalidvalue
  %0 = firrtl.invalidvalue : !firrtl.uint<1>
}
// CHECK-NEXT: }

// CHECK-LABEL: firrtl.module @PropInvalids
firrtl.module @PropInvalids(out %out : !firrtl.uint<4>, out %outs : !firrtl.sint<4>, out %out1 : !firrtl.uint<1>) {
  // CHECK-NOT firrtl.not
  %inv = firrtl.invalidvalue : !firrtl.uint<4>
  %not = firrtl.not %inv : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out, %not : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.bits
  %inv2 = firrtl.invalidvalue : !firrtl.uint<5>
  %bits = firrtl.bits %inv2 3 to 0 : (!firrtl.uint<5>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out, %bits : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.head
  %inv3 = firrtl.invalidvalue : !firrtl.uint<5>
  %head = firrtl.head %inv3, 4 : (!firrtl.uint<5>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out, %head : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.tail
  %inv4 = firrtl.invalidvalue : !firrtl.uint<5>
  %tail = firrtl.tail %inv4, 1 : (!firrtl.uint<5>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out, %tail : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.asSInt
  %inv5 = firrtl.invalidvalue : !firrtl.uint<4>
  %assint = firrtl.asSInt %inv5 : (!firrtl.uint<4>) -> !firrtl.sint<4>
  firrtl.matchingconnect %outs, %assint : !firrtl.sint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.asUInt
  %inv6 = firrtl.invalidvalue : !firrtl.sint<4>
  %asuint = firrtl.asUInt %inv6 : (!firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.matchingconnect %out, %asuint : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.subfield
  %inv7 = firrtl.invalidvalue : !firrtl.bundle<a: uint<4>, b: uint<5>>
  %subfield = firrtl.subfield %inv7[a] : !firrtl.bundle<a: uint<4>, b: uint<5>>
  firrtl.matchingconnect %out, %subfield : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.subindex
  %inv8 = firrtl.invalidvalue : !firrtl.vector<uint<4>, 6 >
  %subindex = firrtl.subindex %inv8[2] : !firrtl.vector<uint<4>, 6 >
  firrtl.matchingconnect %out, %subindex : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.bitcast
  %inv9 = firrtl.invalidvalue : !firrtl.vector<uint<4>, 1 >
  %bitcast = firrtl.bitcast %inv9 : (!firrtl.vector<uint<4>, 1 >) -> !firrtl.uint<4>
  firrtl.matchingconnect %out, %bitcast : !firrtl.uint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.andr
  %inva = firrtl.invalidvalue : !firrtl.uint<4>
  %andr = firrtl.andr %inva : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %andr : !firrtl.uint<1>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.orr
  %invb = firrtl.invalidvalue : !firrtl.uint<4>
  %orr = firrtl.orr %invb : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %orr : !firrtl.uint<1>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.xorr
  %invc = firrtl.invalidvalue : !firrtl.uint<4>
  %xorr = firrtl.xorr %invc : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %xorr : !firrtl.uint<1>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

  // This one depends on the folder
  // CHECK-NOT firrtl.invalidvalue
  // CHECK-NOT firrtl.xorr
  %invd = firrtl.invalidvalue : !firrtl.uint<0>
  %zbits = firrtl.xorr %invd : (!firrtl.uint<0>) -> !firrtl.uint<1>
  firrtl.matchingconnect %out1, %zbits : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %out1, %c0_ui1

  // CHECK-NOT firrtl.cvt
  %inve = firrtl.invalidvalue : !firrtl.uint<3>
  %cvtu = firrtl.cvt %inve : (!firrtl.uint<3>) -> !firrtl.sint<4>
  firrtl.matchingconnect %outs, %cvtu : !firrtl.sint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.pad
  // CHECK-NEXT: firrtl.asSInt
  // CHECK-NEXT: firrtl.matchingconnect

  // CHECK-NOT firrtl.cvt
  %invf = firrtl.invalidvalue : !firrtl.sint<4>
  %cvts = firrtl.cvt %invf : (!firrtl.sint<4>) -> !firrtl.sint<4>
  firrtl.matchingconnect %outs, %cvts : !firrtl.sint<4>

  // CHECK: firrtl.invalidvalue
  // CHECK-NEXT: firrtl.matchingconnect

}

// CHECK-LABEL: firrtl.module @AggregateCreate(
firrtl.module @AggregateCreate(in %vector_in: !firrtl.vector<uint<1>, 2>,
                               in %bundle_in: !firrtl.bundle<a: uint<1>, b: uint<1>>,
                               out %vector_out: !firrtl.vector<uint<1>, 2>,
                               out %bundle_out: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
  %0 = firrtl.subindex %vector_in[0] : !firrtl.vector<uint<1>, 2>
  %1 = firrtl.subindex %vector_in[1] : !firrtl.vector<uint<1>, 2>
  %vector = firrtl.vectorcreate %0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  firrtl.matchingconnect %vector_out, %vector : !firrtl.vector<uint<1>, 2>

  %2 = firrtl.subfield %bundle_in["a"] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  %3 = firrtl.subfield %bundle_in["b"] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  %bundle = firrtl.bundlecreate %2, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  firrtl.matchingconnect %bundle_out, %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK-NEXT: firrtl.matchingconnect %vector_out, %vector_in : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: firrtl.matchingconnect %bundle_out, %bundle_in : !firrtl.bundle<a: uint<1>, b: uint<1>>
}

// CHECK-LABEL: firrtl.module @AggregateCreateSingle(
firrtl.module @AggregateCreateSingle(in %vector_in: !firrtl.vector<uint<1>, 1>,
                               in %bundle_in: !firrtl.bundle<a: uint<1>>,
                               out %vector_out: !firrtl.vector<uint<1>, 1>,
                               out %bundle_out: !firrtl.bundle<a: uint<1>>) {

  %0 = firrtl.subindex %vector_in[0] : !firrtl.vector<uint<1>, 1>
  %vector = firrtl.vectorcreate %0 : (!firrtl.uint<1>) -> !firrtl.vector<uint<1>, 1>
  firrtl.matchingconnect %vector_out, %vector : !firrtl.vector<uint<1>, 1>

  %2 = firrtl.subfield %bundle_in["a"] : !firrtl.bundle<a: uint<1>>
  %bundle = firrtl.bundlecreate %2 : (!firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>>
  firrtl.matchingconnect %bundle_out, %bundle : !firrtl.bundle<a: uint<1>>
  // CHECK-NEXT: firrtl.matchingconnect %vector_out, %vector_in : !firrtl.vector<uint<1>, 1>
  // CHECK-NEXT: firrtl.matchingconnect %bundle_out, %bundle_in : !firrtl.bundle<a: uint<1>>
}

// CHECK-LABEL: firrtl.module @AggregateCreateEmpty(
firrtl.module @AggregateCreateEmpty(
                               out %vector_out: !firrtl.vector<uint<1>, 0>,
                               out %bundle_out: !firrtl.bundle<>) {

  %vector = firrtl.vectorcreate : () -> !firrtl.vector<uint<1>, 0>
  firrtl.matchingconnect %vector_out, %vector : !firrtl.vector<uint<1>, 0>

  %bundle = firrtl.bundlecreate : () -> !firrtl.bundle<>
  firrtl.matchingconnect %bundle_out, %bundle : !firrtl.bundle<>
  // CHECK-DAG: %[[VEC:.+]] = firrtl.aggregateconstant [] : !firrtl.vector<uint<1>, 0>
  // CHECK-DAG: %[[BUNDLE:.+]] = firrtl.aggregateconstant [] : !firrtl.bundle<>
  // CHECK-DAG: firrtl.matchingconnect %vector_out, %[[VEC]] : !firrtl.vector<uint<1>, 0>
  // CHECK-DAG: firrtl.matchingconnect %bundle_out, %[[BUNDLE]] : !firrtl.bundle<>
}

// CHECK-LABEL: firrtl.module @AggregateCreateConst(
firrtl.module @AggregateCreateConst(
                               out %vector_out: !firrtl.vector<uint<1>, 2>,
                               out %bundle_out: !firrtl.bundle<a: uint<1>, b: uint<1>>) {

  %const = firrtl.constant 0 : !firrtl.uint<1>
  %vector = firrtl.vectorcreate %const, %const : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  firrtl.matchingconnect %vector_out, %vector : !firrtl.vector<uint<1>, 2>

  %bundle = firrtl.bundlecreate %const, %const : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  firrtl.matchingconnect %bundle_out, %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK-DAG: %[[VEC:.+]] = firrtl.aggregateconstant [0 : ui1, 0 : ui1] : !firrtl.vector<uint<1>, 2>
  // CHECK-DAG: %[[BUNDLE:.+]] = firrtl.aggregateconstant [0 : ui1, 0 : ui1] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK-DAG: firrtl.matchingconnect %vector_out, %[[VEC]] : !firrtl.vector<uint<1>, 2>
  // CHECK-DAG: firrtl.matchingconnect %bundle_out, %[[BUNDLE]] : !firrtl.bundle<a: uint<1>, b: uint<1>>
}


// CHECK-LABEL: firrtl.module private @RWProbeUnused
firrtl.module private @RWProbeUnused(in %in: !firrtl.uint<4>, in %clk: !firrtl.clock, out %out: !firrtl.uint) {
  // CHECK-NOT: forceable
  %n, %n_ref = firrtl.node interesting_name %in forceable : !firrtl.uint<4>
  %w, %w_ref = firrtl.wire interesting_name forceable : !firrtl.uint, !firrtl.rwprobe<uint>
  firrtl.connect %w, %n : !firrtl.uint, !firrtl.uint<4>
  %r, %r_ref = firrtl.reg interesting_name %clk forceable : !firrtl.clock, !firrtl.uint, !firrtl.rwprobe<uint>
  firrtl.connect %r, %w : !firrtl.uint, !firrtl.uint
  firrtl.connect %out, %r : !firrtl.uint, !firrtl.uint
}


// CHECK-LABEL: firrtl.module @ClockGateIntrinsic
firrtl.module @ClockGateIntrinsic(in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>, in %testEnable: !firrtl.uint<1>) {
  // CHECK-NEXT: firrtl.specialconstant 0
  %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

  // CHECK-NEXT: %zeroClock = firrtl.node interesting_name %c0_clock
  %0 = firrtl.int.clock_gate %c0_clock, %enable
  %zeroClock = firrtl.node interesting_name %0 : !firrtl.clock

  // CHECK-NEXT: %alwaysOff1 = firrtl.node interesting_name %c0_clock
  // CHECK-NEXT: %alwaysOff2 = firrtl.node interesting_name %c0_clock
  %1 = firrtl.int.clock_gate %clock, %c0_ui1
  %2 = firrtl.int.clock_gate %clock, %c0_ui1, %c0_ui1
  %alwaysOff1 = firrtl.node interesting_name %1 : !firrtl.clock
  %alwaysOff2 = firrtl.node interesting_name %2 : !firrtl.clock

  // CHECK-NEXT: %alwaysOn1 = firrtl.node interesting_name %clock
  // CHECK-NEXT: %alwaysOn2 = firrtl.node interesting_name %clock
  // CHECK-NEXT: %alwaysOn3 = firrtl.node interesting_name %clock
  %3 = firrtl.int.clock_gate %clock, %c1_ui1
  %4 = firrtl.int.clock_gate %clock, %c1_ui1, %testEnable
  %5 = firrtl.int.clock_gate %clock, %enable, %c1_ui1
  %alwaysOn1 = firrtl.node interesting_name %3 : !firrtl.clock
  %alwaysOn2 = firrtl.node interesting_name %4 : !firrtl.clock
  %alwaysOn3 = firrtl.node interesting_name %5 : !firrtl.clock

  // CHECK-NEXT: [[TMP:%.+]] = firrtl.int.clock_gate %clock, %enable
  // CHECK-NEXT: %dropTestEnable = firrtl.node interesting_name [[TMP]]
  %6 = firrtl.int.clock_gate %clock, %enable, %c0_ui1
  %dropTestEnable = firrtl.node interesting_name %6 : !firrtl.clock
}

// CHECK-LABEL: firrtl.module @RefTypes
firrtl.module @RefTypes(
    out %x: !firrtl.bundle<a flip: uint<1>>,
    out %y: !firrtl.bundle<a: uint<1>>) {

  %a = firrtl.wire : !firrtl.uint<1>
  %b = firrtl.wire : !firrtl.uint<1>
  %a_ref = firrtl.ref.send  %a : !firrtl.uint<1>
  %a_read_ref = firrtl.ref.resolve %a_ref : !firrtl.probe<uint<1>>
  // CHECK: firrtl.matchingconnect %b, %a
  firrtl.matchingconnect %b, %a_read_ref : !firrtl.uint<1>

  // Don't collapse if types don't match.
  // CHECK: ref.resolve
  %x_ref = firrtl.ref.send %x : !firrtl.bundle<a flip: uint<1>>
  %x_read = firrtl.ref.resolve %x_ref : !firrtl.probe<bundle<a: uint<1>>>
  firrtl.matchingconnect %y, %x_read : !firrtl.bundle<a: uint<1>>

  // CHECK-NOT: forceable
  // CHECK: firrtl.matchingconnect %f_wire, %b
  // CHECK-NOT: forceable
  %f, %f_rw = firrtl.node %b forceable : !firrtl.uint<1>
  %f_read = firrtl.ref.resolve %f_rw : !firrtl.rwprobe<uint<1>>
  %f_wire = firrtl.wire : !firrtl.uint<1>
  firrtl.matchingconnect %f_wire, %f_read : !firrtl.uint<1>

  // CHECK: firrtl.wire forceable
  // CHECK: ref.resolve
  %flipbundle, %flipbundle_rw = firrtl.wire forceable : !firrtl.bundle<a flip: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>>>
  %flipbundle_read = firrtl.ref.resolve %flipbundle_rw : !firrtl.rwprobe<bundle<a: uint<1>>>
  %flipbundle_wire = firrtl.wire : !firrtl.bundle<a : uint<1>>
  firrtl.matchingconnect %flipbundle_wire, %flipbundle_read : !firrtl.bundle<a: uint<1>>
}

// Do not rename InstanceOp: https://github.com/llvm/circt/issues/5351
firrtl.extmodule @System(out foo: !firrtl.uint<1>)
firrtl.module @DonotUpdateInstanceName(in %in: !firrtl.uint<1>, out %a: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
  %system_foo = firrtl.instance system @System(out foo: !firrtl.uint<1>)
  // CHECK: firrtl.instance system
  %b = firrtl.node %system_foo : !firrtl.uint<1>
  firrtl.matchingconnect %a, %b : !firrtl.uint<1>
}

// CHECK-LABEL: @RefCastSame
firrtl.module private @RefCastSame(in %in: !firrtl.uint<1>, out %out: !firrtl.probe<uint<1>>) {
  // Drop no-op ref.cast's.
  // CHECK-NEXT:  %[[P:.+]] = firrtl.ref.send
  // CHECK-NEXT:  firrtl.ref.define %out, %[[P]]
  // CHECK-NEXT:  }
  %probe = firrtl.ref.send %in : !firrtl.uint<1>
  %same_as_in = firrtl.ref.cast %probe : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>>
  firrtl.ref.define %out, %same_as_in : !firrtl.probe<uint<1>>
}

// CHECK-LABEL: @Issue5527
firrtl.module @Issue5527(in %x: !firrtl.uint<1>, out %out: !firrtl.uint<2>) attributes {convention = #firrtl<convention scalarized>} {
  %0 = firrtl.cvt %x : (!firrtl.uint<1>) -> !firrtl.sint<2>
  %c2_si4 = firrtl.constant 2 : !firrtl.sint<4>
  %1 = firrtl.and %0, %c2_si4 : (!firrtl.sint<2>, !firrtl.sint<4>) -> !firrtl.uint<4>
  %2 = firrtl.tail %1, 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  // CHECK: firrtl.matchingconnect %out, %c0_ui2
  firrtl.matchingconnect %out, %2 : !firrtl.uint<2>
}

// Test dropping force/release statements with constant-zero predicates.
// CHECK-LABEL: @RefMe(
firrtl.module private @RefMe(out %p: !firrtl.rwprobe<uint<4>>) {
  %x, %x_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
  firrtl.ref.define %p, %x_ref : !firrtl.rwprobe<uint<4>>
}
// CHECK-LABEL: @ForceRelease(
firrtl.module @ForceRelease(in %clock: !firrtl.clock, in %x: !firrtl.uint<4>) {
    %c = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: firrtl.instance
    %r_p = firrtl.instance r @RefMe(out p: !firrtl.rwprobe<uint<4>>)

    // CHECK-NOT: firrtl.ref
    firrtl.ref.force %clock, %c, %r_p, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.force_initial %c, %r_p, %x : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    firrtl.ref.release %clock, %c, %r_p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    firrtl.ref.release_initial %c, %r_p : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    // CHECK-NEXT: }
}

// Don't produce invalid IR (matchingconnect w/flips).
// CHECK-LABEL: @Issue5650(
firrtl.module @Issue5650(in %io_y: !firrtl.uint<1>, out %io_x: !firrtl.uint<1>) {
  %io = firrtl.wire : !firrtl.bundle<y flip: uint<1>, x: uint<1>>
  %2 = firrtl.subfield %io[y] : !firrtl.bundle<y flip: uint<1>, x: uint<1>>
  firrtl.matchingconnect %2, %io_y : !firrtl.uint<1>
  %3 = firrtl.subfield %io[x] : !firrtl.bundle<y flip: uint<1>, x: uint<1>>
  firrtl.matchingconnect %io_x, %3 : !firrtl.uint<1>
  firrtl.matchingconnect %3, %2 : !firrtl.uint<1>
}

// CHECK-LABEL: @HasBeenReset
firrtl.module @HasBeenReset(in %clock: !firrtl.clock, in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.asyncreset, in %reset3: !firrtl.reset) {
  // CHECK-NEXT: %c0_clock = firrtl.specialconstant 0
  // CHECK-NEXT: %c1_clock = firrtl.specialconstant 1
  // CHECK-NEXT: %c0_ui1 = firrtl.constant 0
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c0_asyncreset = firrtl.specialconstant 0 : !firrtl.asyncreset
  %c1_asyncreset = firrtl.specialconstant 1 : !firrtl.asyncreset
  %c0_reset = firrtl.specialconstant 0 : !firrtl.reset
  %c1_reset = firrtl.specialconstant 1 : !firrtl.reset
  %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
  %c1_clock = firrtl.specialconstant 1 : !firrtl.clock

  // CHECK-NEXT: firrtl.node sym @constResetS0 %c0_ui1
  // CHECK-NEXT: firrtl.node sym @constResetS1 %c0_ui1
  // CHECK-NEXT: firrtl.node sym @constResetA0 %c0_ui1
  // CHECK-NEXT: firrtl.node sym @constResetA1 %c0_ui1
  // CHECK-NEXT: firrtl.node sym @constResetR0 %c0_ui1
  // CHECK-NEXT: firrtl.node sym @constResetR1 %c0_ui1
  %r0 = firrtl.int.has_been_reset %clock, %c0_ui1 : !firrtl.uint<1>
  %r1 = firrtl.int.has_been_reset %clock, %c1_ui1 : !firrtl.uint<1>
  %r2 = firrtl.int.has_been_reset %clock, %c0_asyncreset : !firrtl.asyncreset
  %r3 = firrtl.int.has_been_reset %clock, %c1_asyncreset : !firrtl.asyncreset
  %r4 = firrtl.int.has_been_reset %clock, %c0_reset : !firrtl.reset
  %r5 = firrtl.int.has_been_reset %clock, %c1_reset : !firrtl.reset
  %constResetS0 = firrtl.node sym @constResetS0 %r0 : !firrtl.uint<1>
  %constResetS1 = firrtl.node sym @constResetS1 %r1 : !firrtl.uint<1>
  %constResetA0 = firrtl.node sym @constResetA0 %r2 : !firrtl.uint<1>
  %constResetA1 = firrtl.node sym @constResetA1 %r3 : !firrtl.uint<1>
  %constResetR0 = firrtl.node sym @constResetR0 %r4 : !firrtl.uint<1>
  %constResetR1 = firrtl.node sym @constResetR1 %r5 : !firrtl.uint<1>

  // CHECK-NEXT: [[TMP1:%.+]] = firrtl.int.has_been_reset %c0_clock, %reset2
  // CHECK-NEXT: [[TMP2:%.+]] = firrtl.int.has_been_reset %c1_clock, %reset2
  // CHECK-NEXT: [[TMP3:%.+]] = firrtl.int.has_been_reset %c0_clock, %reset3
  // CHECK-NEXT: [[TMP4:%.+]] = firrtl.int.has_been_reset %c1_clock, %reset3
  // CHECK-NEXT: firrtl.node sym @constClockS0 %c0_ui1
  // CHECK-NEXT: firrtl.node sym @constClockS1 %c0_ui1
  // CHECK-NEXT: firrtl.node sym @constClockA0 [[TMP1]]
  // CHECK-NEXT: firrtl.node sym @constClockA1 [[TMP2]]
  // CHECK-NEXT: firrtl.node sym @constClockR0 [[TMP3]]
  // CHECK-NEXT: firrtl.node sym @constClockR1 [[TMP4]]
  %c0 = firrtl.int.has_been_reset %c0_clock, %reset1 : !firrtl.uint<1>
  %c1 = firrtl.int.has_been_reset %c1_clock, %reset1 : !firrtl.uint<1>
  %c2 = firrtl.int.has_been_reset %c0_clock, %reset2 : !firrtl.asyncreset
  %c3 = firrtl.int.has_been_reset %c1_clock, %reset2 : !firrtl.asyncreset
  %c4 = firrtl.int.has_been_reset %c0_clock, %reset3 : !firrtl.reset
  %c5 = firrtl.int.has_been_reset %c1_clock, %reset3 : !firrtl.reset
  %constClockS0 = firrtl.node sym @constClockS0 %c0 : !firrtl.uint<1>
  %constClockS1 = firrtl.node sym @constClockS1 %c1 : !firrtl.uint<1>
  %constClockA0 = firrtl.node sym @constClockA0 %c2 : !firrtl.uint<1>
  %constClockA1 = firrtl.node sym @constClockA1 %c3 : !firrtl.uint<1>
  %constClockR0 = firrtl.node sym @constClockR0 %c4 : !firrtl.uint<1>
  %constClockR1 = firrtl.node sym @constClockR1 %c5 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Whens
firrtl.module @Whens(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>, in %reset: !firrtl.uint<1>) {
  %true = firrtl.constant 1: !firrtl.uint<1>
  %false = firrtl.constant 0: !firrtl.uint<1>

  // Erase empty whens.
  // CHECK-NOT: when %a
  firrtl.when %a : !firrtl.uint<1> {
  }
  firrtl.when %a : !firrtl.uint<1> {
  } else {
  }
  // Erase an empty else block.
  // CHECK:      firrtl.when %reset : !firrtl.uint<1> {
  // CHECK-NEXT:   firrtl.printf %clock, %reset, "foo!"  : !firrtl.clock, !firrtl.uint<1>
  // CHECK-NEXT: }
  firrtl.when %reset : !firrtl.uint<1> {
    firrtl.printf %clock, %reset, "foo!"  : !firrtl.clock, !firrtl.uint<1>
  } else {
  }

  // CHECK-NEXT:  firrtl.printf %clock, %reset, "bar!"  : !firrtl.clock, !firrtl.uint<1>
  firrtl.when %false : !firrtl.uint<1> {
  } else {
    firrtl.printf %clock, %reset, "bar!"  : !firrtl.clock, !firrtl.uint<1>
  }

  // CHECK-NOT:  firrtl.when
  firrtl.when %true : !firrtl.uint<1> {
  } else {
    firrtl.printf %clock, %reset, "baz!"  : !firrtl.clock, !firrtl.uint<1>
  }
}

// CHECK-LABEL: Probes
firrtl.module @Probes(in %clock: !firrtl.clock) {
  // CHECK-NOT: firrtl.int.fpga_probe %clock, %zero_width : !firrtl.uint<0>
  %zero_width = firrtl.wire : !firrtl.uint<0>
  firrtl.int.fpga_probe %clock, %zero_width : !firrtl.uint<0>
  // CHECK-NOT: firrtl.int.fpga_probe %clock, %empty_bundle : !firrtl.bundle<a: uint<0>>
  %empty_bundle = firrtl.wire : !firrtl.bundle<a: uint<0>>
  firrtl.int.fpga_probe %clock, %empty_bundle : !firrtl.bundle<a: uint<0>>
}

// CHECK-LABEL: sizeof
firrtl.module @sizeof(in %clock: !firrtl.clock,
                      in %vec: !firrtl.vector<uint<3>,3>,
                      in %bundle: !firrtl.bundle<a: uint<2>, b: uint<2>>) {
  // CHECK: %c4_ui32 = firrtl.constant 4 : !firrtl.uint<32>
  // CHECK: %c9_ui32 = firrtl.constant 9 : !firrtl.uint<32>
  // CHECK: %c1_ui32 = firrtl.constant 1 : !firrtl.uint<32>
  // CHECK: %n_c = firrtl.node interesting_name %c1_ui32
  // CHECK: %n_vec = firrtl.node interesting_name %c9_ui32
  // CHECK: %n_bundle = firrtl.node interesting_name %c4_ui32
  %s_c = firrtl.int.sizeof %clock : (!firrtl.clock) -> !firrtl.uint<32>
  %n_c = firrtl.node interesting_name %s_c : !firrtl.uint<32>
  %s_vec = firrtl.int.sizeof %vec : (!firrtl.vector<uint<3>,3>) -> !firrtl.uint<32>
  %n_vec = firrtl.node interesting_name %s_vec : !firrtl.uint<32>
  %s_bundle = firrtl.int.sizeof %bundle : (!firrtl.bundle<a: uint<2>, b: uint<2>>) -> !firrtl.uint<32>
  %n_bundle = firrtl.node interesting_name %s_bundle : !firrtl.uint<32>
}

// CHECK-LABEL: @multibit_mux_drop_front
firrtl.module @multibit_mux_drop_front(in %vec_0: !firrtl.uint<8>, in %vec_1: !firrtl.uint<8>,
                                       in %vec_2: !firrtl.uint<8>, in %vec_3: !firrtl.uint<8>,
                                       in %vec_4: !firrtl.uint<8>, in %vec_5: !firrtl.uint<8>,
                                       in %vec_6: !firrtl.uint<8>, in %vec_7: !firrtl.uint<8>,
                                       in %index: !firrtl.uint<2>, in %index_unknown_width: !firrtl.uint,
                                       out %b: !firrtl.uint<8>, out %c: !firrtl.uint<8>) {
  // CHECK-NEXT:  %0 = firrtl.multibit_mux %index, %vec_3, %vec_2, %vec_1, %vec_0
  %0 = firrtl.multibit_mux %index, %vec_7, %vec_6, %vec_5, %vec_4, %vec_3, %vec_2, %vec_1, %vec_0 : !firrtl.uint<2>, !firrtl.uint<8>
  firrtl.matchingconnect %b, %0 : !firrtl.uint<8>

  // CHECK:  %1 = firrtl.multibit_mux %index_unknown_width, %vec_7
  %1 = firrtl.multibit_mux %index_unknown_width, %vec_7, %vec_6, %vec_5, %vec_4, %vec_3, %vec_2, %vec_1, %vec_0 : !firrtl.uint, !firrtl.uint<8>
  firrtl.matchingconnect %c, %1 : !firrtl.uint<8>
}

firrtl.module private @Issue7562(in %sel : !firrtl.uint<1>, in %a : !firrtl.const.uint<1>, out %out : !firrtl.uint) {
  %res = firrtl.mux(%sel, %a, %a) : (!firrtl.uint<1>, !firrtl.const.uint<1>, !firrtl.const.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %res : !firrtl.uint, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.class @PropertyArithmetic
firrtl.class @PropertyArithmetic(in %in: !firrtl.integer, out %out0: !firrtl.integer, out %out1: !firrtl.integer, out %out2: !firrtl.integer, out %out3: !firrtl.integer) {
  // CHECK: [[C4:%.+]] = firrtl.integer 4
  // CHECK: [[C1:%.+]] = firrtl.integer 1
  %0 = firrtl.integer 0
  %1 = firrtl.integer 1
  %2 = firrtl.integer 2

  %3 = firrtl.integer.shl %1, %2 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %4 = firrtl.integer.shl %in, %0 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

  %5 = firrtl.integer.shr %2, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %6 = firrtl.integer.shr %in, %0 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

  // CHECK: firrtl.propassign %out0, [[C4]]
  // CHECK: firrtl.propassign %out1, %in
  // CHECK: firrtl.propassign %out2, [[C1]]
  // CHECK: firrtl.propassign %out3, %in
  firrtl.propassign %out0, %3 : !firrtl.integer
  firrtl.propassign %out1, %4 : !firrtl.integer
  firrtl.propassign %out2, %5 : !firrtl.integer
  firrtl.propassign %out3, %6 : !firrtl.integer
}

// CHECK-LABEL: firrtl.module private @LayerBlocks
firrtl.module private @LayerBlocks() {
  // CHECK-NOT: firrtl.layerblock @A
  firrtl.layerblock @A {}
}

// CHECK-LABEL: firrtl.module @name_prop
firrtl.module @name_prop(in %clock: !firrtl.clock, in %next: !firrtl.uint<8>, out %out_b: !firrtl.uint<8>) {
  // CHECK: %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  %reg = firrtl.reg droppable_name %clock : !firrtl.clock, !firrtl.uint<8>
  // CHECK-NOT: firrtl.node
  firrtl.connect %reg, %next : !firrtl.uint<8>, !firrtl.uint<8>
  %n = firrtl.node %reg : !firrtl.uint<8>
  firrtl.connect %out_b, %n : !firrtl.uint<8>, !firrtl.uint<8>

  // CHECK: %m = firrtl.wire
  %wire = firrtl.wire droppable_name : !firrtl.uint<8>
  // CHECK-NOT: firrtl.node
  firrtl.connect %wire, %reg : !firrtl.uint<8>, !firrtl.uint<8>
  %m = firrtl.node %wire : !firrtl.uint<8>
  firrtl.connect %out_b, %m : !firrtl.uint<8>, !firrtl.uint<8>
}

hw.hierpath @xmr [@XMRTest::@target]

// CHECK-LABEL: firrtl.module private @XMRTest
firrtl.module private @XMRTest() {
  %target = firrtl.wire sym @target : !firrtl.uint<1>

  // CHECK-NOT: firrtl.xmr.deref
  %0 = firrtl.xmr.deref @xmr : !firrtl.uint<1>

  // CHECK-NOT: firrtl.xmr.ref
  %1 = firrtl.xmr.ref @xmr : !firrtl.ref<uint<1>>
}

// CHECK-LABEL: firrtl.module private @RefTest
firrtl.module private @RefTest() {
  %target = firrtl.wire : !firrtl.bundle<a: uint<1>>

  // CHECK-NOT: firrtl.ref.send
  %0 = firrtl.ref.send %target : !firrtl.bundle<a: uint<1>>

  // CHECK-NOT: firrtl.ref.resolve
  %1 = firrtl.ref.resolve %0 : !firrtl.probe<bundle<a: uint<1>>>

  // CHECK-NOT: firrtl.ref.cast
   %2 = firrtl.ref.cast %0 : (!firrtl.probe<bundle<a: uint<1>>>) -> !firrtl.probe<bundle<a: uint<1>>, @A>

  // CHECK-NOT: firrtl.ref.sub
  %3 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<a: uint<1>>>
}

// CHECK-LABEL: firrtl.module @Domains
firrtl.module @Domains(
) {

  %a = firrtl.wire : !firrtl.uint<1>
  %b = firrtl.wire : !firrtl.uint<1>
  %0 = firrtl.unsafe_domain_cast %a : !firrtl.uint<1>
  // CHECK: firrtl.matchingconnect %b, %a
  firrtl.matchingconnect %b, %0 : !firrtl.uint<1>

}

// CHECK-LABEL: firrtl.module @ResetCast
firrtl.module @ResetCast(out %a0: !firrtl.reset, out %a1: !firrtl.reset) {
  // CHECK-DAG: [[R0:%.+]] = firrtl.specialconstant 0 : !firrtl.reset
  // CHECK-DAG: [[R1:%.+]] = firrtl.specialconstant 1 : !firrtl.reset
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

  // CHECK: firrtl.matchingconnect %a0, [[R0]]
  %0 = firrtl.asReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.reset
  firrtl.matchingconnect %a0, %0 : !firrtl.reset

  // CHECK: firrtl.matchingconnect %a1, [[R1]]
  %1 = firrtl.asReset %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.reset
  firrtl.matchingconnect %a1, %1 : !firrtl.reset
}

}
