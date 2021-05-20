// RUN: circt-opt -simple-canonicalizer %s | FileCheck %s

firrtl.circuit "And" {

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

  // COM: Check that 'div(a, a) -> 1' works for known UInt widths
  // CHECK: firrtl.connect %b, [[ONE_i4]]
  %0 = firrtl.div %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %b, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // COM: Check that 'div(c, c) -> 1' works for known SInt widths
  // CHECK: firrtl.connect %d, [[ONE_s5]] : !firrtl.sint<5>, !firrtl.sint<5>
  %1 = firrtl.div %c, %c : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.sint<5>
  firrtl.connect %d, %1 : !firrtl.sint<5>, !firrtl.sint<5>

  // COM: Check that 'div(e, e) -> 1' works for unknown UInt widths
  // CHECK: firrtl.connect %f, [[ONE_i2]]
  %2 = firrtl.div %e, %e : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
  firrtl.connect %f, %2 : !firrtl.uint, !firrtl.uint

  // COM: Check that 'div(g, g) -> 1' works for unknown SInt widths
  // CHECK: firrtl.connect %h, [[ONE_s2]]
  %3 = firrtl.div %g, %g : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
  firrtl.connect %h, %3 : !firrtl.sint, !firrtl.sint

  // COM: Check that 'div(a, 1) -> a' for known UInt widths
  // CHECK: firrtl.connect %b, %a
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %4 = firrtl.div %a, %c1_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %b, %4 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %i, %c5_ui4
  %c1_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %5 = firrtl.div %c1_ui4, %c3_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %i, %5 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @And
firrtl.module @And(in %in: !firrtl.uint<4>,
                   in %sin: !firrtl.sint<4>,
                   out %out: !firrtl.uint<4>) {
  // CHECK: firrtl.connect %out, %c1_ui4
  %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.and %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %1 = firrtl.and %in, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c0_ui4
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.and %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %3 = firrtl.and %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // Mixed type inputs - the constant is zero extended, not sign extended, so it
  // cannot be folded!

  // CHECK: firrtl.and %in, %c3_ui2
  // CHECK-NEXT: firrtl.connect %out,
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %4 = firrtl.and %in, %c3_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %out, %4 : !firrtl.uint<4>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.connect %out, %c1_ui4
  %c1_si4 = firrtl.constant 1 : !firrtl.sint<4>
  %5 = firrtl.and %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %5 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: [[AND:%.+]] = firrtl.and %sin, %sin
  // CHECK-NEXT: firrtl.connect %out, [[AND]]
  %6 = firrtl.and %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Or
firrtl.module @Or(in %in: !firrtl.uint<4>,
                  in %sin: !firrtl.sint<4>,
                  out %out: !firrtl.uint<4>) {
  // CHECK: firrtl.connect %out, %c7_ui4
  %c4_ui4 = firrtl.constant 4 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.or %c3_ui4, %c4_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c15_ui4
  %c1_ui15 = firrtl.constant 15 : !firrtl.uint<4>
  %1 = firrtl.or %in, %c1_ui15 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.or %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %3 = firrtl.or %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.connect %out, %c1_ui4
  %c1_si4 = firrtl.constant 1 : !firrtl.sint<4>
  %5 = firrtl.or %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %5 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: [[OR:%.+]] = firrtl.or %sin, %sin
  // CHECK-NEXT: firrtl.connect %out, [[OR]]
  %6 = firrtl.or %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Xor
firrtl.module @Xor(in %in: !firrtl.uint<4>,
                   in %sin: !firrtl.sint<4>,
                   out %out: !firrtl.uint<4>) {
  // CHECK: firrtl.connect %out, %c2_ui4
  %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %0 = firrtl.xor %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<4>
  %2 = firrtl.xor %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c0_ui4
  %3 = firrtl.xor %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.connect %out, %c0_ui4
  %6 = firrtl.xor %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @EQ
firrtl.module @EQ(in %in: !firrtl.uint<1>,
                  out %out: !firrtl.uint<1>) {
  // CHECK: firrtl.connect %out, %in
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %0 = firrtl.eq %in, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // Issue #368: https://github.com/llvm/circt/issues/368
  %c3_ui2 = firrtl.constant 3 : !firrtl.uint<2>
  %1 = firrtl.eq %in, %c3_ui2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<1>
  firrtl.connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: firrtl.eq %in, %c3_ui2
  // CHECK: firrtl.connect
}

// CHECK-LABEL: firrtl.module @NEQ
firrtl.module @NEQ(in %in: !firrtl.uint<1>,
                   out %out: !firrtl.uint<1>) {
  // CHECK: firrtl.connect %out, %in
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<1>
  %0 = firrtl.neq %in, %c1_ui0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Cat
firrtl.module @Cat(in %in4: !firrtl.uint<4>,
                   out %out4: !firrtl.uint<4>,
                   out %outcst: !firrtl.uint<8>) {

  // CHECK: firrtl.connect %out4, %in4
  %0 = firrtl.bits %in4 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %1 = firrtl.bits %in4 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %2 = firrtl.cat %0, %1 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %out4, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %outcst, %c243_ui8
  %c15_ui4 = firrtl.constant 15 : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
  %3 = firrtl.cat %c15_ui4, %c3_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>
  firrtl.connect %outcst, %3 : !firrtl.uint<8>, !firrtl.uint<8>
 }

// CHECK-LABEL: firrtl.module @Bits
firrtl.module @Bits(in %in1: !firrtl.uint<1>,
                    in %in4: !firrtl.uint<4>,
                    out %out1: !firrtl.uint<1>,
                    out %out2: !firrtl.uint<2>,
                    out %out4: !firrtl.uint<4>) {
  // CHECK: firrtl.connect %out1, %in1
  %0 = firrtl.bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out4, %in4
  %1 = firrtl.bits %in4 3 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out4, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out2, %c1_ui2
  %c10_ui4 = firrtl.constant 10 : !firrtl.uint<4>
  %2 = firrtl.bits %c10_ui4 2 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  firrtl.connect %out2, %2 : !firrtl.uint<2>, !firrtl.uint<2>


  // CHECK: firrtl.bits %in4 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %out1, %
  %3 = firrtl.bits %in4 3 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  %4 = firrtl.bits %3 1 to 1 : (!firrtl.uint<3>) -> !firrtl.uint<1>
  firrtl.connect %out1, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out1, %in1
  %5 = firrtl.bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %5 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Head
firrtl.module @Head(in %in4u: !firrtl.uint<4>,
                    out %out1u: !firrtl.uint<1>,
                    out %out3u: !firrtl.uint<3>) {
  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 3
  // CHECK-NEXT: firrtl.connect %out1u, [[BITS]]
  %0 = firrtl.head %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 1
  // CHECK-NEXT: firrtl.connect %out3u, [[BITS]]
  %1 = firrtl.head %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %1 : !firrtl.uint<3>, !firrtl.uint<3>

  // CHECK: firrtl.connect %out3u, %c5_ui3
  %c10_ui4 = firrtl.constant 10 : !firrtl.uint<4>
  %2 = firrtl.head %c10_ui4, 3 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %2 : !firrtl.uint<3>, !firrtl.uint<3>
}

// CHECK-LABEL: firrtl.module @Mux
firrtl.module @Mux(in %in: !firrtl.uint<4>,
                   in %cond: !firrtl.uint<1>,
                   out %out: !firrtl.uint<4>,
                   out %out1: !firrtl.uint<1>) {
  // CHECK: firrtl.connect %out, %in
  %0 = firrtl.mux (%cond, %in, %in) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c7_ui4
  %c7_ui4 = firrtl.constant 7 : !firrtl.uint<4>
  %c1_ui0 = firrtl.constant 0 : !firrtl.uint<1>
  %2 = firrtl.mux (%c1_ui0, %in, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out1, %cond
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %3 = firrtl.mux (%cond, %c1_ui1, %c1_ui0) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %3 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Pad
firrtl.module @Pad(in %in1u: !firrtl.uint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>,
                   out %outs: !firrtl.sint<4>) {
  // CHECK: firrtl.connect %out1u, %in1u
  %0 = firrtl.pad %in1u, 1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %outu, %c1_ui4
  %c1_ui0 = firrtl.constant 1 : !firrtl.uint<1>
  %1 = firrtl.pad %c1_ui0, 4 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  firrtl.connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: firrtl.connect %outs, %c-1_si4
  %c1_si1 = firrtl.constant -1 : !firrtl.sint<1>
  %2 = firrtl.pad %c1_si1, 4 : (!firrtl.sint<1>) -> !firrtl.sint<4>
  firrtl.connect %outs, %2 : !firrtl.sint<4>, !firrtl.sint<4>
}

// CHECK-LABEL: firrtl.module @Shl
firrtl.module @Shl(in %in1u: !firrtl.uint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>) {
  // CHECK: firrtl.connect %out1u, %in1u
  %0 = firrtl.shl %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %outu, %c8_ui4
  %c1_ui0 = firrtl.constant 1 : !firrtl.uint<1>
  %1 = firrtl.shl %c1_ui0, 3 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  firrtl.connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Shr
firrtl.module @Shr(in %in1u: !firrtl.uint<1>,
                   in %in4u: !firrtl.uint<4>,
                   in %in1s: !firrtl.sint<1>,
                   in %in4s: !firrtl.sint<4>,
                   in %in0u: !firrtl.uint<0>,
                   out %out1s: !firrtl.sint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>) {
  // CHECK: firrtl.connect %out1u, %in1u
  %0 = firrtl.shr %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out1u, %c0_ui1
  %1 = firrtl.shr %in4u, 4 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %1 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out1u, %c0_ui1
  %2 = firrtl.shr %in4u, 5 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %2 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.connect %out1s, [[CAST]]
  %3 = firrtl.shr %in4s, 3 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %3 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.connect %out1s, [[CAST]]
  %4 = firrtl.shr %in4s, 4 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %4 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.connect %out1s, [[CAST]]
  %5 = firrtl.shr %in4s, 5 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %5 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: firrtl.connect %out1u, %c1_ui1
  %c12_ui4 = firrtl.constant 12 : !firrtl.uint<4>
  %6 = firrtl.shr %c12_ui4, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %6 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 3
  // CHECK-NEXT: firrtl.connect %out1u, [[BITS]]
  %7 = firrtl.shr %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %7 : !firrtl.uint<1>, !firrtl.uint<1>

  // Issue #313: https://github.com/llvm/circt/issues/313
  // CHECK: firrtl.connect %out1s, %in1s : !firrtl.sint<1>, !firrtl.sint<1>
  %8 = firrtl.shr %in1s, 42 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %8 : !firrtl.sint<1>, !firrtl.sint<1>

  // Issue #1064: https://github.com/llvm/circt/issues/1064
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %9 = firrtl.dshr %in0u, %c1_ui1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  firrtl.connect %out1u, %9 : !firrtl.uint<1>, !firrtl.uint<0>
}

// CHECK-LABEL: firrtl.module @Tail
firrtl.module @Tail(in %in4u: !firrtl.uint<4>,
                    out %out1u: !firrtl.uint<1>,
                    out %out3u: !firrtl.uint<3>) {
  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 0 to 0
  // CHECK-NEXT: firrtl.connect %out1u, [[BITS]]
  %0 = firrtl.tail %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 2 to 0
  // CHECK-NEXT: firrtl.connect %out3u, [[BITS]]
  %1 = firrtl.tail %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %1 : !firrtl.uint<3>, !firrtl.uint<3>


  // CHECK: firrtl.connect %out3u, %c2_ui3
  %c10_ui4 = firrtl.constant 10 : !firrtl.uint<4>
  %2 = firrtl.tail %c10_ui4, 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %2 : !firrtl.uint<3>, !firrtl.uint<3>
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
  // CHECK-NEXT: firrtl.connect %tmp8, %c130_ui10
}

// CHECK-LABEL: firrtl.module @issue437
firrtl.module @issue437(out %tmp19: !firrtl.uint<1>) {
  // CHECK-NEXT: %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %c-1_si1 = firrtl.constant -1 : !firrtl.sint<1>
  %0 = firrtl.bits %c-1_si1 0 to 0 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  firrtl.connect %tmp19, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @issue446
// CHECK-NEXT: firrtl.xor %inp_1, %inp_1
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
// CHECK-NEXT: firrtl.div
firrtl.module @issue516(in %inp_0: !firrtl.uint<0>, out %tmp3: !firrtl.uint<0>) {
  %0 = firrtl.div %inp_0, %inp_0 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %tmp3, %0 : !firrtl.uint<0>, !firrtl.uint<0>
}

// https://github.com/llvm/circt/issues/591
// CHECK-LABEL: @reg_cst_prop1
// CHECK-NEXT:   %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %out_b, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop1(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  %tmp_a = firrtl.reg %clock {name = "tmp_a"} : (!firrtl.clock) -> !firrtl.uint<8>
  %tmp_b = firrtl.reg %clock {name = "tmp_b"} : (!firrtl.clock) -> !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %tmp_b, %tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %out_b, %tmp_b : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: @reg_cst_prop2
// CHECK-NEXT:   %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %out_b, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop2(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %tmp_b = firrtl.reg %clock {name = "tmp_b"} : (!firrtl.clock) -> !firrtl.uint<8>
  firrtl.connect %out_b, %tmp_b : !firrtl.uint<8>, !firrtl.uint<8>

  %tmp_a = firrtl.reg %clock {name = "tmp_a"} : (!firrtl.clock) -> !firrtl.uint<8>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %tmp_b, %tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: @reg_cst_prop3
// CHECK-NEXT:   %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %out_b, %c0_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop3(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %tmp_a = firrtl.reg %clock {name = "tmp_a"} : (!firrtl.clock) -> !firrtl.uint<8>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = firrtl.xor %tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  firrtl.connect %out_b, %xor : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: @pcon
// CHECK-NEXT:   %0 = firrtl.bits %in 4 to 0 : (!firrtl.uint<9>) -> !firrtl.uint<5>
// CHECK-NEXT:   firrtl.connect %out, %0 : !firrtl.uint<5>, !firrtl.uint<5>
// CHECK-NEXT:  }
firrtl.module @pcon(in %in: !firrtl.uint<9>, out %out: !firrtl.uint<5>) {
  firrtl.partialconnect %out, %in : !firrtl.uint<5>, !firrtl.uint<9>
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

// CHECK-LABEL: @wire_cst_prop1
// CHECK-NEXT:   %c10_ui9 = firrtl.constant 10 : !firrtl.uint<9>
// CHECK-NEXT:   firrtl.connect %out_b, %c10_ui9 : !firrtl.uint<9>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %tmp_a = firrtl.wire : !firrtl.uint<8>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = firrtl.add %tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %xor : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @wire_port_prop1
// CHECK-NEXT:   firrtl.connect %out_b, %in_a : !firrtl.uint<9>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_port_prop1(in %in_a: !firrtl.uint<9>, out %out_b: !firrtl.uint<9>) {
  %tmp = firrtl.wire : !firrtl.uint<9>
  firrtl.connect %tmp, %in_a : !firrtl.uint<9>, !firrtl.uint<9>

  firrtl.connect %out_b, %tmp : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @LEQWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.geq %a, %c42_ui
firrtl.module @LEQWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = firrtl.constant 42 : !firrtl.uint
  %1 = firrtl.leq %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1

  %1 = firrtl.lt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y1, %c0_ui1

  %2 = firrtl.geq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1

  %3 = firrtl.gt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1

  %4 = firrtl.eq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c1_ui1

  %5 = firrtl.neq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1

  // b <= 3 -> 1
  // b <= 4 -> 1
  %2 = firrtl.leq %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.leq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1

  // b <= -5 -> 0
  // b <= -6 -> 0
  %4 = firrtl.leq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.leq %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1

  // b < 4 -> 1
  // b < 5 -> 1
  %2 = firrtl.lt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.lt %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1

  // b < -4 -> 0
  // b < -5 -> 0
  %4 = firrtl.lt %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.lt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
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
  // CHECK-NEXT: firrtl.connect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c0_ui1

  // b >= 4 -> 0
  // b >= 5 -> 0
  %2 = firrtl.geq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.geq %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1

  // b >= -4 -> 1
  // b >= -5 -> 1
  %4 = firrtl.geq %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.geq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c1_ui1
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
  // CHECK-NEXT: firrtl.connect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c0_ui1

  // b > 3 -> 0
  // b > 4 -> 0
  %2 = firrtl.gt %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.gt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1

  // b > -5 -> 1
  // b > -6 -> 1
  %4 = firrtl.gt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.gt %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c1_ui1
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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y11, %c1_ui1
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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y11, %c1_ui1
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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y8, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y11, %c1_ui1
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
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y7, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y8, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y10, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y11, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y12, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y13, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y14, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y15, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y16, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y17, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y18, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y19, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y20, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y21, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y22, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y23, %c1_ui1
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
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant

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
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y2, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y4, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y6, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y8, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y9, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y10, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y11, %c0_ui1
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
  %4 = firrtl.lt %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %5 = firrtl.lt %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %6 = firrtl.lt %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %7 = firrtl.lt %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %8 = firrtl.geq %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %9 = firrtl.geq %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %10 = firrtl.geq %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %11 = firrtl.geq %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %12 = firrtl.gt %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %13 = firrtl.gt %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %14 = firrtl.gt %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %15 = firrtl.gt %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>

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
  // CHECK-NEXT: firrtl.connect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y6, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y7, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y8, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y9, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y10, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y11, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y12, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y13, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y14, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y15, %c1_ui1
}

// CHECK-LABEL: @add_cst_prop1
// CHECK-NEXT:   %c11_ui9 = firrtl.constant 11 : !firrtl.uint<9>
// CHECK-NEXT:   firrtl.connect %out_b, %c11_ui9 : !firrtl.uint<9>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %c6_ui7 = firrtl.constant 6 : !firrtl.uint<7>
  %tmp_a = firrtl.wire : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.add %tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %add : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @add_cst_prop2
// CHECK-NEXT:   %c-1_si9 = firrtl.constant -1 : !firrtl.sint<9>
// CHECK-NEXT:   firrtl.connect %out_b, %c-1_si9 : !firrtl.sint<9>, !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop2(out %out_b: !firrtl.sint<9>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.add %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  firrtl.connect %out_b, %add : !firrtl.sint<9>, !firrtl.sint<9>
}

// CHECK-LABEL: @add_cst_prop3
// CHECK-NEXT:   %c-2_si4 = firrtl.constant -2 : !firrtl.sint<4>
// CHECK-NEXT:   firrtl.connect %out_b, %c-2_si4 : !firrtl.sint<4>, !firrtl.sint<4>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop3(out %out_b: !firrtl.sint<4>) {
  %c1_si2 = firrtl.constant -1 : !firrtl.sint<2>
  %tmp_a = firrtl.wire : !firrtl.sint<2>
  %c1_si3 = firrtl.constant -1 : !firrtl.sint<3>
  firrtl.connect %tmp_a, %c1_si2 : !firrtl.sint<2>, !firrtl.sint<2>
  %add = firrtl.add %tmp_a, %c1_si3 : (!firrtl.sint<2>, !firrtl.sint<3>) -> !firrtl.sint<4>
  firrtl.connect %out_b, %add : !firrtl.sint<4>, !firrtl.sint<4>
}

// CHECK-LABEL: @sub_cst_prop1
// CHECK-NEXT:      %c1_ui9 = firrtl.constant 1 : !firrtl.uint<9>
// CHECK-NEXT:      firrtl.connect %out_b, %c1_ui9 : !firrtl.uint<9>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %c6_ui7 = firrtl.constant 6 : !firrtl.uint<7>
  %tmp_a = firrtl.wire : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.sub %tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %add : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @sub_cst_prop2
// CHECK-NEXT:      %c-11_si9 = firrtl.constant -11 : !firrtl.sint<9>
// CHECK-NEXT:      firrtl.connect %out_b, %c-11_si9 : !firrtl.sint<9>, !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop2(out %out_b: !firrtl.sint<9>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.sub %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  firrtl.connect %out_b, %add : !firrtl.sint<9>, !firrtl.sint<9>
}

// CHECK-LABEL: @mul_cst_prop1
// CHECK-NEXT:      %c30_ui15 = firrtl.constant 30 : !firrtl.uint<15>
// CHECK-NEXT:      firrtl.connect %out_b, %c30_ui15 : !firrtl.uint<15>, !firrtl.uint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop1(out %out_b: !firrtl.uint<15>) {
  %c6_ui7 = firrtl.constant 6 : !firrtl.uint<7>
  %tmp_a = firrtl.wire : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.mul %tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<15>
  firrtl.connect %out_b, %add : !firrtl.uint<15>, !firrtl.uint<15>
}

// CHECK-LABEL: @mul_cst_prop2
// CHECK-NEXT:      %c-30_si15 = firrtl.constant -30 : !firrtl.sint<15>
// CHECK-NEXT:      firrtl.connect %out_b, %c-30_si15 : !firrtl.sint<15>, !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop2(out %out_b: !firrtl.sint<15>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant 5 : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.mul %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  firrtl.connect %out_b, %add : !firrtl.sint<15>, !firrtl.sint<15>
}

// CHECK-LABEL: @mul_cst_prop3
// CHECK-NEXT:      %c30_si15 = firrtl.constant 30 : !firrtl.sint<15>
// CHECK-NEXT:      firrtl.connect %out_b, %c30_si15 : !firrtl.sint<15>, !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop3(out %out_b: !firrtl.sint<15>) {
  %c6_ui7 = firrtl.constant -6 : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant -5 : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.mul %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  firrtl.connect %out_b, %add : !firrtl.sint<15>, !firrtl.sint<15>
}

// CHECK-LABEL: firrtl.module @MuxInvalidOpt
firrtl.module @MuxInvalidOpt(in %cond: !firrtl.uint<1>, in %data: !firrtl.uint<4>, out %out1: !firrtl.uint<4>, out %out2: !firrtl.uint<4>, out %out3: !firrtl.uint<4>, out %out4: !firrtl.uint<4>) {

  // We can optimize out these mux's since the invalid value can take on any input.
  %tmp1 = firrtl.invalidvalue : !firrtl.uint<4>
  %a = firrtl.mux(%cond, %data, %tmp1) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out1, %data
  firrtl.connect %out1, %a : !firrtl.uint<4>, !firrtl.uint<4>

  %b = firrtl.mux(%cond, %tmp1, %data) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out2, %data
  firrtl.connect %out2, %b : !firrtl.uint<4>, !firrtl.uint<4>

  %false = firrtl.constant 0 : !firrtl.uint<1>
  %c = firrtl.mux(%false, %data, %tmp1) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out3, %data
  firrtl.connect %out3, %c : !firrtl.uint<4>, !firrtl.uint<4>

  %true = firrtl.constant 1 : !firrtl.uint<1>
  %d = firrtl.mux(%false, %tmp1, %data) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out4, %data
  firrtl.connect %out4, %d : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @MuxCanon
firrtl.module @MuxCanon(in %c1: !firrtl.uint<1>, in %c2: !firrtl.uint<1>, in %d1: !firrtl.uint<5>, in %d2: !firrtl.uint<5>, in %d3: !firrtl.uint<5>, out %foo: !firrtl.uint<5>, out %foo2: !firrtl.uint<5>) {
  %0 = firrtl.mux(%c1, %d2, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %1 = firrtl.mux(%c1, %d1, %0) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %2 = firrtl.mux(%c1, %0, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  firrtl.connect %foo, %1 : !firrtl.uint<5>, !firrtl.uint<5>
  firrtl.connect %foo2, %2 : !firrtl.uint<5>, !firrtl.uint<5>
  // CHECK: firrtl.mux(%c1, %d1, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  // CHECK: firrtl.mux(%c1, %d2, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
}

// CHECK-LABEL: firrtl.module @EmptyNode
firrtl.module @EmptyNode(in %d1: !firrtl.uint<5>, out %foo: !firrtl.uint<5>, out %foo2: !firrtl.uint<5>) {
  %bar0 = firrtl.node %d1 : !firrtl.uint<5>
  %bar1 = firrtl.node %d1 : !firrtl.uint<5>
  %bar2 = firrtl.node %d1 {annotations = [{extrastuff = "n1"}]} : !firrtl.uint<5>
  firrtl.connect %foo, %bar1 : !firrtl.uint<5>, !firrtl.uint<5>
  firrtl.connect %foo2, %bar2 : !firrtl.uint<5>, !firrtl.uint<5>
}
// CHECK-NEXT: %bar2 = firrtl.node %d1 {annotations = [{extrastuff = "n1"}]}
// CHECK-NEXT: firrtl.connect %foo, %d1
// CHECK-NEXT: firrtl.connect %foo2, %bar2

// COM: https://github.com/llvm/circt/issues/929
// CHECK-LABEL: firrtl.module @MuxInvalidTypeOpt
firrtl.module @MuxInvalidTypeOpt(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<4>) {
  %c7_ui4 = firrtl.constant 7 : !firrtl.uint<4>
  %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
  %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
  %0 = firrtl.mux (%in, %c7_ui4, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  %1 = firrtl.mux (%in, %c1_ui2, %0) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}
// CHECK: firrtl.mux(%in, %c7_ui4, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
// CHECK: firrtl.mux(%in, %c1_ui2, %0) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

// CHECK-LABEL: firrtl.module @issue1100
// CHECK: firrtl.connect %tmp62, %c1_ui1
  firrtl.module @issue1100(out %tmp62: !firrtl.uint<1>) {
    %c-1_si2 = firrtl.constant -1 : !firrtl.sint<2>
    %0 = firrtl.orr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    firrtl.connect %tmp62, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @issue1101
  // CHECK: firrtl.connect %y, %c-7_si4
  firrtl.module @issue1101(out %y: !firrtl.sint<4>) {
    %c9_si10 = firrtl.constant 9 : !firrtl.sint<10>
    firrtl.partialconnect %y, %c9_si10 : !firrtl.sint<4>, !firrtl.sint<10>
  }

firrtl.module @zeroWidthMem(in %clock: !firrtl.clock) {
  %_M__T_10, %_M__T_11, %_M__T_18 = firrtl.mem Undefined  {depth = 8 : i64, name = "_M", portNames = ["_T_10", "_T_11", "_T_18"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>, !firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>, !firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: flip<bundle<id: uint<4>>>>>
  %0 = firrtl.subfield %_M__T_18("addr") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: flip<bundle<id: uint<4>>>>>) -> !firrtl.uint<3>
  %invalid_ui3 = firrtl.invalidvalue : !firrtl.uint<3>
  firrtl.connect %0, %invalid_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
  %1 = firrtl.subfield %_M__T_18("en") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: flip<bundle<id: uint<4>>>>>) -> !firrtl.uint<1>
  %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
  firrtl.connect %1, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  %2 = firrtl.subfield %_M__T_18("clk") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: flip<bundle<id: uint<4>>>>>) -> !firrtl.clock
  %invalid_clock = firrtl.invalidvalue : !firrtl.clock
  firrtl.connect %2, %invalid_clock : !firrtl.clock, !firrtl.clock
  %3 = firrtl.subfield %_M__T_18("data") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: flip<bundle<id: uint<4>>>>>) -> !firrtl.bundle<id: uint<4>>
  %4 = firrtl.subfield %3("id") : (!firrtl.bundle<id: uint<4>>) -> !firrtl.uint<4>
  %5 = firrtl.subfield %_M__T_10("addr") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.uint<3>
  %invalid_ui3_0 = firrtl.invalidvalue : !firrtl.uint<3>
  firrtl.connect %5, %invalid_ui3_0 : !firrtl.uint<3>, !firrtl.uint<3>
  %6 = firrtl.subfield %_M__T_10("en") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.uint<1>
  %invalid_ui1_1 = firrtl.invalidvalue : !firrtl.uint<1>
  firrtl.connect %6, %invalid_ui1_1 : !firrtl.uint<1>, !firrtl.uint<1>
  %7 = firrtl.subfield %_M__T_10("clk") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.clock
  %invalid_clock_2 = firrtl.invalidvalue : !firrtl.clock
  firrtl.connect %7, %invalid_clock_2 : !firrtl.clock, !firrtl.clock
  %8 = firrtl.subfield %_M__T_10("data") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.bundle<id: uint<4>>
  %9 = firrtl.subfield %8("id") : (!firrtl.bundle<id: uint<4>>) -> !firrtl.uint<4>
  %invalid_ui4 = firrtl.invalidvalue : !firrtl.uint<4>
  firrtl.connect %9, %invalid_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
  %10 = firrtl.subfield %_M__T_10("mask") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.bundle<id: uint<1>>
  %11 = firrtl.subfield %10("id") : (!firrtl.bundle<id: uint<1>>) -> !firrtl.uint<1>
  %invalid_ui1_3 = firrtl.invalidvalue : !firrtl.uint<1>
  firrtl.connect %11, %invalid_ui1_3 : !firrtl.uint<1>, !firrtl.uint<1>
  %12 = firrtl.subfield %_M__T_11("addr") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.uint<3>
  %invalid_ui3_4 = firrtl.invalidvalue : !firrtl.uint<3>
  firrtl.connect %12, %invalid_ui3_4 : !firrtl.uint<3>, !firrtl.uint<3>
  %13 = firrtl.subfield %_M__T_11("en") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.uint<1>
  %invalid_ui1_5 = firrtl.invalidvalue : !firrtl.uint<1>
  firrtl.connect %13, %invalid_ui1_5 : !firrtl.uint<1>, !firrtl.uint<1>
  %14 = firrtl.subfield %_M__T_11("clk") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.clock
  %invalid_clock_6 = firrtl.invalidvalue : !firrtl.clock
  firrtl.connect %14, %invalid_clock_6 : !firrtl.clock, !firrtl.clock
  %15 = firrtl.subfield %_M__T_11("data") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.bundle<id: uint<4>>
  %16 = firrtl.subfield %15("id") : (!firrtl.bundle<id: uint<4>>) -> !firrtl.uint<4>
  %invalid_ui4_7 = firrtl.invalidvalue : !firrtl.uint<4>
  firrtl.connect %16, %invalid_ui4_7 : !firrtl.uint<4>, !firrtl.uint<4>
  // COM: Check that missing is fine %17 = firrtl.subfield %_M__T_11("mask") : (!firrtl.flip<bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>>) -> !firrtl.bundle<id: uint<1>>
}
// CHECK-LABEL: firrtl.module @zeroWidthMem
// CHECK-NEXT:  }
}

