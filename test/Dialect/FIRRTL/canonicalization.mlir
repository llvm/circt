// RUN: circt-opt -simple-canonicalizer %s | FileCheck %s

firrtl.circuit "And" {

// CHECK-LABEL: firrtl.module @Div
firrtl.module @Div(%a: !firrtl.uint<4>,
                  %b: !firrtl.flip<uint<4>>,
                  %c: !firrtl.sint<4>,
                  %d: !firrtl.flip<sint<5>>,
                  %e: !firrtl.uint,
                  %f: !firrtl.flip<uint>,
                  %g: !firrtl.sint,
                  %h: !firrtl.flip<sint>) {

  // CHECK-DAG: [[ONE_i4:%.+]] = firrtl.constant(1 : i4) : !firrtl.uint<4>
  // CHECK-DAG: [[ONE_s5:%.+]] = firrtl.constant(1 : i5) : !firrtl.sint<5>
  // CHECK-DAG: [[ONE_i2:%.+]] = firrtl.constant(1 : i2) : !firrtl.uint
  // CHECK-DAG: [[ONE_s2:%.+]] = firrtl.constant(1 : i2) : !firrtl.sint

  // COM: Check that 'div(a, a) -> 1' works for known UInt widths
  // CHECK: firrtl.connect %b, [[ONE_i4]]
  %0 = firrtl.div %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %b, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // COM: Check that 'div(c, c) -> 1' works for known SInt widths
  // CHECK: firrtl.connect %d, [[ONE_s5]] : !firrtl.flip<sint<5>>, !firrtl.sint<5>
  %1 = firrtl.div %c, %c : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.sint<5>
  firrtl.connect %d, %1 : !firrtl.flip<sint<5>>, !firrtl.sint<5>

  // COM: Check that 'div(e, e) -> 1' works for unknown UInt widths
  // CHECK: firrtl.connect %f, [[ONE_i2]]
  %2 = firrtl.div %e, %e : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
  firrtl.connect %f, %2 : !firrtl.flip<uint>, !firrtl.uint

  // COM: Check that 'div(g, g) -> 1' works for unknown SInt widths
  // CHECK: firrtl.connect %h, [[ONE_s2]]
  %3 = firrtl.div %g, %g : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
  firrtl.connect %h, %3 : !firrtl.flip<sint>, !firrtl.sint

  // COM: Check that 'div(a, 1) -> a' for known UInt widths
  // CHECK: firrtl.connect %b, %a
  %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
  %4 = firrtl.div %a, %c1_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %b, %4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

}

// CHECK-LABEL: firrtl.module @And
firrtl.module @And(%in: !firrtl.uint<4>,
                   %sin: !firrtl.sint<4>,
                   %out: !firrtl.flip<uint<4>>) {
  // CHECK: firrtl.connect %out, %c1_ui4
  %c1_ui4 = firrtl.constant(1 : ui4) : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant(3 : ui4) : !firrtl.uint<4>
  %0 = firrtl.and %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %c15_ui4 = firrtl.constant(15 : ui4) : !firrtl.uint<4>
  %1 = firrtl.and %in, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c0_ui4
  %c1_ui0 = firrtl.constant(0 : ui4) : !firrtl.uint<4>
  %2 = firrtl.and %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %3 = firrtl.and %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // Mixed type inputs - the constant is zero extended, not sign extended, so it
  // cannot be folded!

  // CHECK: firrtl.and %in, %c3_ui2
  // CHECK-NEXT: firrtl.connect %out,
  %c3_ui2 = firrtl.constant(3 : ui2) : !firrtl.uint<2>
  %4 = firrtl.and %in, %c3_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %out, %4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.connect %out, %c1_ui4
  %c1_si4 = firrtl.constant(1 : si4) : !firrtl.sint<4>
  %5 = firrtl.and %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: [[AND:%.+]] = firrtl.and %sin, %sin
  // CHECK-NEXT: firrtl.connect %out, [[AND]]
  %6 = firrtl.and %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Or
firrtl.module @Or(%in: !firrtl.uint<4>,
                  %sin: !firrtl.sint<4>,
                  %out: !firrtl.flip<uint<4>>) {
  // CHECK: firrtl.connect %out, %c7_ui4
  %c4_ui4 = firrtl.constant(4 : ui4) : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant(3 : ui4) : !firrtl.uint<4>
  %0 = firrtl.or %c3_ui4, %c4_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c15_ui4
  %c1_ui15 = firrtl.constant(15 : ui4) : !firrtl.uint<4>
  %1 = firrtl.or %in, %c1_ui15 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %c1_ui0 = firrtl.constant(0 : ui4) : !firrtl.uint<4>
  %2 = firrtl.or %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %3 = firrtl.or %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.connect %out, %c1_ui4
  %c1_si4 = firrtl.constant(1 : si4) : !firrtl.sint<4>
  %5 = firrtl.or %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: [[OR:%.+]] = firrtl.or %sin, %sin
  // CHECK-NEXT: firrtl.connect %out, [[OR]]
  %6 = firrtl.or %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Xor
firrtl.module @Xor(%in: !firrtl.uint<4>,
                   %sin: !firrtl.sint<4>,
                   %out: !firrtl.flip<uint<4>>) {
  // CHECK: firrtl.connect %out, %c2_ui4
  %c1_ui4 = firrtl.constant(1 : ui4) : !firrtl.uint<4>
  %c3_ui4 = firrtl.constant(3 : ui4) : !firrtl.uint<4>
  %0 = firrtl.xor %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %in
  %c1_ui0 = firrtl.constant(0 : ui4) : !firrtl.uint<4>
  %2 = firrtl.xor %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c0_ui4
  %3 = firrtl.xor %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %3 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: firrtl.connect %out, %c0_ui4
  %6 = firrtl.xor %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %6 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @EQ
firrtl.module @EQ(%in: !firrtl.uint<1>,
                   %out: !firrtl.flip<uint<1>>) {
  // CHECK: firrtl.connect %out, %in
  %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
  %0 = firrtl.eq %in, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // Issue #368: https://github.com/llvm/circt/issues/368
  %c3_ui2 = firrtl.constant(3 : ui2) : !firrtl.uint<2>
  %1 = firrtl.eq %in, %c3_ui2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<1>
  firrtl.connect %out, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK: firrtl.eq %in, %c3_ui2
  // CHECK: firrtl.connect
}

// CHECK-LABEL: firrtl.module @NEQ
firrtl.module @NEQ(%in: !firrtl.uint<1>,
                   %out: !firrtl.flip<uint<1>>) {
  // CHECK: firrtl.connect %out, %in
  %c1_ui0 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
  %0 = firrtl.neq %in, %c1_ui0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Cat
firrtl.module @Cat(%in4: !firrtl.uint<4>,
                   %out4: !firrtl.flip<uint<4>>) {

  // CHECK: firrtl.connect %out4, %in4
  %0 = firrtl.bits %in4 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %1 = firrtl.bits %in4 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %2 = firrtl.cat %0, %1 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %out4, %2 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Bits
firrtl.module @Bits(%in1: !firrtl.uint<1>,
                    %in4: !firrtl.uint<4>,
                    %out1: !firrtl.flip<uint<1>>,
                    %out2: !firrtl.flip<uint<2>>,
                    %out4: !firrtl.flip<uint<4>>) {
  // CHECK: firrtl.connect %out1, %in1
  %0 = firrtl.bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out4, %in4
  %1 = firrtl.bits %in4 3 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out4, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out2, %c1_ui2
  %c7_ui4 = firrtl.constant(10 : ui4) : !firrtl.uint<4>
  %2 = firrtl.bits %c7_ui4 2 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  firrtl.connect %out2, %2 : !firrtl.flip<uint<2>>, !firrtl.uint<2>


  // CHECK: firrtl.bits %in4 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %out1, %
  %3 = firrtl.bits %in4 3 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  %4 = firrtl.bits %3 1 to 1 : (!firrtl.uint<3>) -> !firrtl.uint<1>
  firrtl.connect %out1, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out1, %in1
  %5 = firrtl.bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Head
firrtl.module @Head(%in4u: !firrtl.uint<4>,
                   %out1u: !firrtl.flip<uint<1>>,
                   %out3u: !firrtl.flip<uint<3>>) {
  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 3
  // CHECK-NEXT: firrtl.connect %out1u, [[BITS]]
  %0 = firrtl.head %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 1
  // CHECK-NEXT: firrtl.connect %out3u, [[BITS]]
  %1 = firrtl.head %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %1 : !firrtl.flip<uint<3>>, !firrtl.uint<3>
}

// CHECK-LABEL: firrtl.module @Mux
firrtl.module @Mux(%in: !firrtl.uint<4>,
                   %cond: !firrtl.uint<1>,
                   %out: !firrtl.flip<uint<4>>,
                   %out1: !firrtl.flip<uint<1>>) {
  // CHECK: firrtl.connect %out, %in
  %0 = firrtl.mux (%cond, %in, %in) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out, %c7_ui4
  %c7_ui4 = firrtl.constant(7 : ui4) : !firrtl.uint<4>
  %c1_ui0 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
  %2 = firrtl.mux (%c1_ui0, %in, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %2 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %out1, %cond
  %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
  %3 = firrtl.mux (%cond, %c1_ui1, %c1_ui0) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Pad
firrtl.module @Pad(%in1u: !firrtl.uint<1>,
                   %out1u: !firrtl.flip<uint<1>>,
                   %outu: !firrtl.flip<uint<4>>,
                   %outs: !firrtl.flip<sint<4>>) {
  // CHECK: firrtl.connect %out1u, %in1u
  %0 = firrtl.pad %in1u, 1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: firrtl.connect %outu, %c1_ui4
  %c1_ui0 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
  %1 = firrtl.pad %c1_ui0, 4 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  firrtl.connect %outu, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  // CHECK: firrtl.connect %outs, %c-1_si4
  %c1_si1 = firrtl.constant(-1 : si1) : !firrtl.sint<1>
  %2 = firrtl.pad %c1_si1, 4 : (!firrtl.sint<1>) -> !firrtl.sint<4>
  firrtl.connect %outs, %2 : !firrtl.flip<sint<4>>, !firrtl.sint<4>
}

// CHECK-LABEL: firrtl.module @Shl
firrtl.module @Shl(%in1u: !firrtl.uint<1>,
                   %out1u: !firrtl.flip<uint<1>>,
                   %outu: !firrtl.flip<uint<4>>) {
  // CHECK: firrtl.connect %out1u, %in1u
  %0 = firrtl.shl %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: firrtl.connect %outu, %c8_ui4
  %c1_ui0 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
  %1 = firrtl.shl %c1_ui0, 3 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  firrtl.connect %outu, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @Shr
firrtl.module @Shr(%in1u: !firrtl.uint<1>,
                   %in4u: !firrtl.uint<4>,
                   %in1s: !firrtl.sint<1>,
                   %in4s: !firrtl.sint<4>,
                   %out1s: !firrtl.flip<sint<1>>,
                   %out1u: !firrtl.flip<uint<1>>,
                   %outu: !firrtl.flip<uint<4>>) {
  // CHECK: firrtl.connect %out1u, %in1u
  %0 = firrtl.shr %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out1u, %c0_ui1
  %1 = firrtl.shr %in4u, 4 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out1u, %c0_ui1
  %2 = firrtl.shr %in4u, 5 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.connect %out1s, [[CAST]]
  %3 = firrtl.shr %in4s, 3 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %3 : !firrtl.flip<sint<1>>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.connect %out1s, [[CAST]]
  %4 = firrtl.shr %in4s, 4 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %4 : !firrtl.flip<sint<1>>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = firrtl.asSInt [[BITS]]
  // CHECK-NEXT: firrtl.connect %out1s, [[CAST]]
  %5 = firrtl.shr %in4s, 5 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %5 : !firrtl.flip<sint<1>>, !firrtl.sint<1>

  // CHECK: firrtl.connect %out1u, %c1_ui1
  %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>
  %6 = firrtl.shr %c12_ui4, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 3 to 3
  // CHECK-NEXT: firrtl.connect %out1u, [[BITS]]
  %7 = firrtl.shr %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // Issue #313: https://github.com/llvm/circt/issues/313
  // CHECK: firrtl.connect %out1s, %in1s : !firrtl.flip<sint<1>>, !firrtl.sint<1>
  %8 = firrtl.shr %in1s, 42 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  firrtl.connect %out1s, %8 : !firrtl.flip<sint<1>>, !firrtl.sint<1>
}

// CHECK-LABEL: firrtl.module @Tail
firrtl.module @Tail(%in4u: !firrtl.uint<4>,
                   %out1u: !firrtl.flip<uint<1>>,
                   %out3u: !firrtl.flip<uint<3>>) {
  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 0 to 0
  // CHECK-NEXT: firrtl.connect %out1u, [[BITS]]
  %0 = firrtl.tail %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.connect %out1u, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = firrtl.bits %in4u 2 to 0
  // CHECK-NEXT: firrtl.connect %out3u, [[BITS]]
  %1 = firrtl.tail %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  firrtl.connect %out3u, %1 : !firrtl.flip<uint<3>>, !firrtl.uint<3>
}

// CHECK-LABEL: firrtl.module @issue326
firrtl.module @issue326(%tmp57: !firrtl.flip<sint<1>>) {
  %c29_si7 = firrtl.constant(29 : si7) : !firrtl.sint<7>
  %0 = firrtl.shr %c29_si7, 47 : (!firrtl.sint<7>) -> !firrtl.sint<1>
   // CHECK: c0_si1 = firrtl.constant(false) : !firrtl.sint<1>
   firrtl.connect %tmp57, %0 : !firrtl.flip<sint<1>>, !firrtl.sint<1>
}

// CHECK-LABEL: firrtl.module @issue331
firrtl.module @issue331(%tmp81: !firrtl.flip<sint<1>>) {
  // CHECK: %c-1_si1 = firrtl.constant(true) : !firrtl.sint<1>
  %c-1_si1 = firrtl.constant(-1 : si1) : !firrtl.sint<1>
  %0 = firrtl.shr %c-1_si1, 3 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  firrtl.connect %tmp81, %0 : !firrtl.flip<sint<1>>, !firrtl.sint<1>
}

// CHECK-LABEL: firrtl.module @issue432
firrtl.module @issue432(%tmp8: !firrtl.flip<uint<10>>) {
  %c130_si10 = firrtl.constant(130 : si10) : !firrtl.sint<10>
  %0 = firrtl.tail %c130_si10, 0 : (!firrtl.sint<10>) -> !firrtl.uint<10>
  firrtl.connect %tmp8, %0 : !firrtl.flip<uint<10>>, !firrtl.uint<10>
  // CHECK-NEXT: %c130_ui10 = firrtl.constant(130 : i10) : !firrtl.uint<10>
  // CHECK-NEXT: firrtl.connect %tmp8, %c130_ui10
}

// CHECK-LABEL: firrtl.module @issue437
firrtl.module @issue437(%tmp19: !firrtl.flip<uint<1>>) {
  // CHECK-NEXT: %c1_ui1 = firrtl.constant(true) : !firrtl.uint<1>
  %c-1_si1 = firrtl.constant(-1 : si1) : !firrtl.sint<1>
  %0 = firrtl.bits %c-1_si1 0 to 0 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  firrtl.connect %tmp19, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @issue446
// CHECK-NEXT: firrtl.xor %inp_1, %inp_1
firrtl.module @issue446(%inp_1: !firrtl.sint<0>, %tmp10: !firrtl.flip<uint<1>>) {
  %0 = firrtl.xor %inp_1, %inp_1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<0>
  firrtl.connect %tmp10, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<0>
}

// CHECK-LABEL: firrtl.module @xorUnsized
// CHECK-NEXT: %c0_ui = firrtl.constant(false) : !firrtl.uint
firrtl.module @xorUnsized(%inp_1: !firrtl.sint, %tmp10: !firrtl.flip<uint>) {
  %0 = firrtl.xor %inp_1, %inp_1 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
  firrtl.connect %tmp10, %0 : !firrtl.flip<uint>, !firrtl.uint
}

// https://github.com/llvm/circt/issues/516
// CHECK-LABEL: @issue516
// CHECK-NEXT: firrtl.div
firrtl.module @issue516(%inp_0: !firrtl.uint<0>, %tmp3: !firrtl.flip<uint<0>>) {
  %0 = firrtl.div %inp_0, %inp_0 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  firrtl.connect %tmp3, %0 : !firrtl.flip<uint<0>>, !firrtl.uint<0>
}

// https://github.com/llvm/circt/issues/591
// CHECK-LABEL: @reg_cst_prop1
// CHECK-NEXT:   %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %out_b, %c5_ui8 : !firrtl.flip<uint<8>>, !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop1(%clock: !firrtl.clock, %out_b: !firrtl.flip<uint<8>>) {
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  %tmp_a = firrtl.reg %clock {name = "tmp_a"} : (!firrtl.clock) -> !firrtl.uint<8>
  %tmp_b = firrtl.reg %clock {name = "tmp_b"} : (!firrtl.clock) -> !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %tmp_b, %tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %out_b, %tmp_b : !firrtl.flip<uint<8>>, !firrtl.uint<8>
}

// CHECK-LABEL: @reg_cst_prop2
// CHECK-NEXT:   %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %out_b, %c5_ui8 : !firrtl.flip<uint<8>>, !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop2(%clock: !firrtl.clock, %out_b: !firrtl.flip<uint<8>>) {
  %tmp_b = firrtl.reg %clock {name = "tmp_b"} : (!firrtl.clock) -> !firrtl.uint<8>
  firrtl.connect %out_b, %tmp_b : !firrtl.flip<uint<8>>, !firrtl.uint<8>

  %tmp_a = firrtl.reg %clock {name = "tmp_a"} : (!firrtl.clock) -> !firrtl.uint<8>
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %tmp_b, %tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: @reg_cst_prop3
// CHECK-NEXT:   %c0_ui8 = firrtl.constant(0 : i8) : !firrtl.uint<8>
// CHECK-NEXT:   firrtl.connect %out_b, %c0_ui8 : !firrtl.flip<uint<8>>, !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop3(%clock: !firrtl.clock, %out_b: !firrtl.flip<uint<8>>) {
  %tmp_a = firrtl.reg %clock {name = "tmp_a"} : (!firrtl.clock) -> !firrtl.uint<8>
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = firrtl.xor %tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  firrtl.connect %out_b, %xor : !firrtl.flip<uint<8>>, !firrtl.uint<8>
}

// CHECK-LABEL: @pcon
// CHECK-NEXT:   %0 = firrtl.bits %in 4 to 0 : (!firrtl.uint<9>) -> !firrtl.uint<5>
// CHECK-NEXT:   firrtl.connect %out, %0 : !firrtl.flip<uint<5>>, !firrtl.uint<5>
// CHECK-NEXT:  }
firrtl.module @pcon(%in: !firrtl.uint<9>, %out: !firrtl.flip<uint<5>>) {
  firrtl.partialconnect %out, %in : !firrtl.flip<uint<5>>, !firrtl.uint<9>
}

// https://github.com/llvm/circt/issues/788

// CHECK-LABEL: @AttachMerge
firrtl.module @AttachMerge(%a: !firrtl.analog<1>, %b: !firrtl.analog<1>,
                           %c: !firrtl.analog<1>) {
  // CHECK-NEXT: firrtl.attach %c, %b, %a :
  // CHECK-NEXT: }
  firrtl.attach %b, %a : !firrtl.analog<1>, !firrtl.analog<1>
  firrtl.attach %c, %b : !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachDeadWire
firrtl.module @AttachDeadWire(%a: !firrtl.analog<1>, %b: !firrtl.analog<1>) {
  // CHECK-NEXT: firrtl.attach %a, %b :
  // CHECK-NEXT: }
  %c = firrtl.wire  : !firrtl.analog<1>
  firrtl.attach %a, %b, %c : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachOpts
firrtl.module @AttachOpts(%a: !firrtl.analog<1>) {
  // CHECK-NEXT: }
  %b = firrtl.wire  : !firrtl.analog<1>
  firrtl.attach %b, %a : !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @wire_cst_prop1
// CHECK-NEXT:   %c10_ui9 = firrtl.constant(10 : i9) : !firrtl.uint<9>
// CHECK-NEXT:   firrtl.connect %out_b, %c10_ui9 : !firrtl.flip<uint<9>>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_cst_prop1(%out_b: !firrtl.flip<uint<9>>) {
  %tmp_a = firrtl.wire : !firrtl.uint<8>
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = firrtl.add %tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %xor : !firrtl.flip<uint<9>>, !firrtl.uint<9>
}

// CHECK-LABEL: @wire_port_prop1
// CHECK-NEXT:   firrtl.connect %out_b, %in_a : !firrtl.flip<uint<9>>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_port_prop1(%in_a: !firrtl.uint<9>, %out_b: !firrtl.flip<uint<9>>) {
  %tmp = firrtl.wire : !firrtl.uint<9>
  firrtl.connect %tmp, %in_a : !firrtl.uint<9>, !firrtl.uint<9>

  firrtl.connect %out_b, %tmp : !firrtl.flip<uint<9>>, !firrtl.uint<9>
}

// CHECK-LABEL: @LEQWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.geq %a, %c42_ui
firrtl.module @LEQWithConstLHS(%a: !firrtl.uint, %b: !firrtl.flip<uint<1>>) {
  %0 = firrtl.constant(42) : !firrtl.uint
  %1 = firrtl.leq %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: @LTWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.gt %a, %c42_ui
firrtl.module @LTWithConstLHS(%a: !firrtl.uint, %b: !firrtl.flip<uint<1>>) {
  %0 = firrtl.constant(42) : !firrtl.uint
  %1 = firrtl.lt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: @GEQWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.leq %a, %c42_ui
firrtl.module @GEQWithConstLHS(%a: !firrtl.uint, %b: !firrtl.flip<uint<1>>) {
  %0 = firrtl.constant(42) : !firrtl.uint
  %1 = firrtl.geq %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: @GTWithConstLHS
// CHECK-NEXT: %c42_ui = firrtl.constant
// CHECK-NEXT: %0 = firrtl.lt %a, %c42_ui
firrtl.module @GTWithConstLHS(%a: !firrtl.uint, %b: !firrtl.flip<uint<1>>) {
  %0 = firrtl.constant(42) : !firrtl.uint
  %1 = firrtl.gt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %b, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

// CHECK-LABEL: @CompareWithSelf
firrtl.module @CompareWithSelf(
  %a: !firrtl.uint,
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant

  %0 = firrtl.leq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1

  %1 = firrtl.lt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y1, %c0_ui1

  %2 = firrtl.geq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1

  %3 = firrtl.gt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1

  %4 = firrtl.eq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c1_ui1

  %5 = firrtl.neq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
}

// CHECK-LABEL: @LEQOutsideBounds
firrtl.module @LEQOutsideBounds(
  %a: !firrtl.uint<3>,
  %b: !firrtl.sint<3>,
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm5_si = firrtl.constant(-5) : !firrtl.sint
  %cm6_si = firrtl.constant(-6) : !firrtl.sint
  %c3_si = firrtl.constant(3) : !firrtl.sint
  %c4_si = firrtl.constant(4) : !firrtl.sint
  %c7_ui = firrtl.constant(7) : !firrtl.uint
  %c8_ui = firrtl.constant(8) : !firrtl.uint

  // a <= 7 -> 1
  // a <= 8 -> 1
  %0 = firrtl.leq %a, %c7_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.leq %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1

  // b <= 3 -> 1
  // b <= 4 -> 1
  %2 = firrtl.leq %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.leq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1

  // b <= -5 -> 0
  // b <= -6 -> 0
  %4 = firrtl.leq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.leq %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
}

// CHECK-LABEL: @LTOutsideBounds
firrtl.module @LTOutsideBounds(
  %a: !firrtl.uint<3>,
  %b: !firrtl.sint<3>,
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm4_si = firrtl.constant(-4) : !firrtl.sint
  %cm5_si = firrtl.constant(-5) : !firrtl.sint
  %c4_si = firrtl.constant(4) : !firrtl.sint
  %c5_si = firrtl.constant(5) : !firrtl.sint
  %c8_ui = firrtl.constant(8) : !firrtl.uint
  %c9_ui = firrtl.constant(9) : !firrtl.uint

  // a < 8 -> 1
  // a < 9 -> 1
  %0 = firrtl.lt %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.lt %a, %c9_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y0, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c1_ui1

  // b < 4 -> 1
  // b < 5 -> 1
  %2 = firrtl.lt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.lt %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c1_ui1

  // b < -4 -> 0
  // b < -5 -> 0
  %4 = firrtl.lt %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.lt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c0_ui1
}

// CHECK-LABEL: @GEQOutsideBounds
firrtl.module @GEQOutsideBounds(
  %a: !firrtl.uint<3>,
  %b: !firrtl.sint<3>,
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm4_si = firrtl.constant(-4) : !firrtl.sint
  %cm5_si = firrtl.constant(-5) : !firrtl.sint
  %c4_si = firrtl.constant(4) : !firrtl.sint
  %c5_si = firrtl.constant(5) : !firrtl.sint
  %c8_ui = firrtl.constant(8) : !firrtl.uint
  %c9_ui = firrtl.constant(9) : !firrtl.uint

  // a >= 8 -> 0
  // a >= 9 -> 0
  %0 = firrtl.geq %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.geq %a, %c9_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c0_ui1

  // b >= 4 -> 0
  // b >= 5 -> 0
  %2 = firrtl.geq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.geq %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1

  // b >= -4 -> 1
  // b >= -5 -> 1
  %4 = firrtl.geq %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.geq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c1_ui1
}

// CHECK-LABEL: @GTOutsideBounds
firrtl.module @GTOutsideBounds(
  %a: !firrtl.uint<3>,
  %b: !firrtl.sint<3>,
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %cm5_si = firrtl.constant(-5) : !firrtl.sint
  %cm6_si = firrtl.constant(-6) : !firrtl.sint
  %c3_si = firrtl.constant(3) : !firrtl.sint
  %c4_si = firrtl.constant(4) : !firrtl.sint
  %c7_ui = firrtl.constant(7) : !firrtl.uint
  %c8_ui = firrtl.constant(8) : !firrtl.uint

  // a > 7 -> 0
  // a > 8 -> 0
  %0 = firrtl.gt %a, %c7_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = firrtl.gt %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y0, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y1, %c0_ui1

  // b > 3 -> 0
  // b > 4 -> 0
  %2 = firrtl.gt %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = firrtl.gt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y2, %c0_ui1
  // CHECK-NEXT: firrtl.connect %y3, %c0_ui1

  // b > -5 -> 1
  // b > -6 -> 1
  %4 = firrtl.gt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = firrtl.gt %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %y4, %c1_ui1
  // CHECK-NEXT: firrtl.connect %y5, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfDifferentWidths
firrtl.module @ComparisonOfDifferentWidths(
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>,
  %y6: !firrtl.flip<uint<1>>,
  %y7: !firrtl.flip<uint<1>>,
  %y8: !firrtl.flip<uint<1>>,
  %y9: !firrtl.flip<uint<1>>,
  %y10: !firrtl.flip<uint<1>>,
  %y11: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c3_si3 = firrtl.constant(3 : i3) : !firrtl.sint<3>
  %c4_si4 = firrtl.constant(4 : i4) : !firrtl.sint<4>
  %c3_ui2 = firrtl.constant(3 : i2) : !firrtl.uint<2>
  %c4_ui3 = firrtl.constant(4 : i3) : !firrtl.uint<3>

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

  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
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
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>,
  %y6: !firrtl.flip<uint<1>>,
  %y7: !firrtl.flip<uint<1>>,
  %y8: !firrtl.flip<uint<1>>,
  %y9: !firrtl.flip<uint<1>>,
  %y10: !firrtl.flip<uint<1>>,
  %y11: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c3_si = firrtl.constant(3) : !firrtl.sint
  %c4_si4 = firrtl.constant(4 : i4) : !firrtl.sint<4>
  %c3_ui = firrtl.constant(3) : !firrtl.uint
  %c4_ui3 = firrtl.constant(4 : i3) : !firrtl.uint<3>

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

  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
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
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>,
  %y6: !firrtl.flip<uint<1>>,
  %y7: !firrtl.flip<uint<1>>,
  %y8: !firrtl.flip<uint<1>>,
  %y9: !firrtl.flip<uint<1>>,
  %y10: !firrtl.flip<uint<1>>,
  %y11: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c0_si = firrtl.constant(0) : !firrtl.sint
  %c4_si = firrtl.constant(4) : !firrtl.sint
  %c0_ui = firrtl.constant(0) : !firrtl.uint
  %c4_ui = firrtl.constant(4) : !firrtl.uint

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

  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
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
  %xu: !firrtl.uint<0>,
  %xs: !firrtl.sint<0>,
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>,
  %y6: !firrtl.flip<uint<1>>,
  %y7: !firrtl.flip<uint<1>>,
  %y8: !firrtl.flip<uint<1>>,
  %y9: !firrtl.flip<uint<1>>,
  %y10: !firrtl.flip<uint<1>>,
  %y11: !firrtl.flip<uint<1>>,
  %y12: !firrtl.flip<uint<1>>,
  %y13: !firrtl.flip<uint<1>>,
  %y14: !firrtl.flip<uint<1>>,
  %y15: !firrtl.flip<uint<1>>,
  %y16: !firrtl.flip<uint<1>>,
  %y17: !firrtl.flip<uint<1>>,
  %y18: !firrtl.flip<uint<1>>,
  %y19: !firrtl.flip<uint<1>>,
  %y20: !firrtl.flip<uint<1>>,
  %y21: !firrtl.flip<uint<1>>,
  %y22: !firrtl.flip<uint<1>>,
  %y23: !firrtl.flip<uint<1>>
) {
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  // CHECK-NEXT: [[_:.+]] = firrtl.constant
  %c0_si4 = firrtl.constant(0 : i4) : !firrtl.sint<4>
  %c0_ui4 = firrtl.constant(0 : i4) : !firrtl.uint<4>
  %c4_si4 = firrtl.constant(4 : i4) : !firrtl.sint<4>
  %c4_ui4 = firrtl.constant(4 : i4) : !firrtl.uint<4>

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

  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y12, %12 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y13, %13 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y14, %14 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y15, %15 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y16, %16 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y17, %17 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y18, %18 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y19, %19 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y20, %20 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y21, %21 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y22, %22 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y23, %23 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
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
  %xu0: !firrtl.uint<0>,
  %xu1: !firrtl.uint<0>,
  %xs0: !firrtl.sint<0>,
  %xs1: !firrtl.sint<0>,
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>,
  %y6: !firrtl.flip<uint<1>>,
  %y7: !firrtl.flip<uint<1>>,
  %y8: !firrtl.flip<uint<1>>,
  %y9: !firrtl.flip<uint<1>>,
  %y10: !firrtl.flip<uint<1>>,
  %y11: !firrtl.flip<uint<1>>,
  %y12: !firrtl.flip<uint<1>>,
  %y13: !firrtl.flip<uint<1>>,
  %y14: !firrtl.flip<uint<1>>,
  %y15: !firrtl.flip<uint<1>>,
  %y16: !firrtl.flip<uint<1>>,
  %y17: !firrtl.flip<uint<1>>,
  %y18: !firrtl.flip<uint<1>>,
  %y19: !firrtl.flip<uint<1>>,
  %y20: !firrtl.flip<uint<1>>,
  %y21: !firrtl.flip<uint<1>>,
  %y22: !firrtl.flip<uint<1>>,
  %y23: !firrtl.flip<uint<1>>
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

  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
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
  %y0: !firrtl.flip<uint<1>>,
  %y1: !firrtl.flip<uint<1>>,
  %y2: !firrtl.flip<uint<1>>,
  %y3: !firrtl.flip<uint<1>>,
  %y4: !firrtl.flip<uint<1>>,
  %y5: !firrtl.flip<uint<1>>,
  %y6: !firrtl.flip<uint<1>>,
  %y7: !firrtl.flip<uint<1>>,
  %y8: !firrtl.flip<uint<1>>,
  %y9: !firrtl.flip<uint<1>>,
  %y10: !firrtl.flip<uint<1>>,
  %y11: !firrtl.flip<uint<1>>,
  %y12: !firrtl.flip<uint<1>>,
  %y13: !firrtl.flip<uint<1>>,
  %y14: !firrtl.flip<uint<1>>,
  %y15: !firrtl.flip<uint<1>>,
  %y16: !firrtl.flip<uint<1>>,
  %y17: !firrtl.flip<uint<1>>,
  %y18: !firrtl.flip<uint<1>>,
  %y19: !firrtl.flip<uint<1>>,
  %y20: !firrtl.flip<uint<1>>,
  %y21: !firrtl.flip<uint<1>>,
  %y22: !firrtl.flip<uint<1>>,
  %y23: !firrtl.flip<uint<1>>
) {
  %c2_si4 = firrtl.constant(2 : i4) : !firrtl.sint<4>
  %c-3_si3 = firrtl.constant(-3 : i3) : !firrtl.sint<3>
  %c2_ui4 = firrtl.constant(2 : i4) : !firrtl.uint<4>
  %c5_ui3 = firrtl.constant(5 : i3) : !firrtl.uint<3>

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

  firrtl.connect %y0, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y1, %1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y2, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y3, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y4, %4 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y5, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y6, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y7, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y8, %8 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y9, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y10, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y11, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y12, %12 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y13, %13 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y14, %14 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  firrtl.connect %y15, %15 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
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
// CHECK-NEXT:   %c11_ui9 = firrtl.constant(11 : i9) : !firrtl.uint<9>
// CHECK-NEXT:   firrtl.connect %out_b, %c11_ui9 : !firrtl.flip<uint<9>>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop1(%out_b: !firrtl.flip<uint<9>>) {
  %c6_ui7 = firrtl.constant(6 : ui7) : !firrtl.uint<7>
  %tmp_a = firrtl.wire : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.add %tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %add : !firrtl.flip<uint<9>>, !firrtl.uint<9>
}

// CHECK-LABEL: @add_cst_prop2
// CHECK-NEXT:   %c-1_si9 = firrtl.constant(-1 : i9) : !firrtl.sint<9>
// CHECK-NEXT:   firrtl.connect %out_b, %c-1_si9 : !firrtl.flip<sint<9>>, !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop2(%out_b: !firrtl.flip<sint<9>>) {
  %c6_ui7 = firrtl.constant(-6 : i7) : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant(5 : i8) : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.add %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  firrtl.connect %out_b, %add : !firrtl.flip<sint<9>>, !firrtl.sint<9>
}

// CHECK-LABEL: @add_cst_prop3
// CHECK-NEXT:   %c-2_si4 = firrtl.constant(-2 : i4) : !firrtl.sint<4>
// CHECK-NEXT:   firrtl.connect %out_b, %c-2_si4 : !firrtl.flip<sint<4>>, !firrtl.sint<4>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop3(%out_b: !firrtl.flip<sint<4>>) {
  %c1_si2 = firrtl.constant(-1 : i2) : !firrtl.sint<2>
  %tmp_a = firrtl.wire : !firrtl.sint<2>
  %c1_si3 = firrtl.constant(-1 : i3) : !firrtl.sint<3>
  firrtl.connect %tmp_a, %c1_si2 : !firrtl.sint<2>, !firrtl.sint<2>
  %add = firrtl.add %tmp_a, %c1_si3 : (!firrtl.sint<2>, !firrtl.sint<3>) -> !firrtl.sint<4>
  firrtl.connect %out_b, %add : !firrtl.flip<sint<4>>, !firrtl.sint<4>
}

// CHECK-LABEL: @sub_cst_prop1
// CHECK-NEXT:      %c1_ui9 = firrtl.constant(1 : i9) : !firrtl.uint<9>
// CHECK-NEXT:      firrtl.connect %out_b, %c1_ui9 : !firrtl.flip<uint<9>>, !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop1(%out_b: !firrtl.flip<uint<9>>) {
  %c6_ui7 = firrtl.constant(6 : ui7) : !firrtl.uint<7>
  %tmp_a = firrtl.wire : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.sub %tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  firrtl.connect %out_b, %add : !firrtl.flip<uint<9>>, !firrtl.uint<9>
}

// CHECK-LABEL: @sub_cst_prop2
// CHECK-NEXT:      %c-11_si9 = firrtl.constant(-11 : i9) : !firrtl.sint<9>
// CHECK-NEXT:      firrtl.connect %out_b, %c-11_si9 : !firrtl.flip<sint<9>>, !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop2(%out_b: !firrtl.flip<sint<9>>) {
  %c6_ui7 = firrtl.constant(-6 : i7) : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant(5 : i8) : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.sub %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  firrtl.connect %out_b, %add : !firrtl.flip<sint<9>>, !firrtl.sint<9>
}

// CHECK-LABEL: @mul_cst_prop1
// CHECK-NEXT:      %c30_ui15 = firrtl.constant(30 : i15) : !firrtl.uint<15>
// CHECK-NEXT:      firrtl.connect %out_b, %c30_ui15 : !firrtl.flip<uint<15>>, !firrtl.uint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop1(%out_b: !firrtl.flip<uint<15>>) {
  %c6_ui7 = firrtl.constant(6 : ui7) : !firrtl.uint<7>
  %tmp_a = firrtl.wire : !firrtl.uint<7>
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = firrtl.mul %tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<15>
  firrtl.connect %out_b, %add : !firrtl.flip<uint<15>>, !firrtl.uint<15>
}

// CHECK-LABEL: @mul_cst_prop2
// CHECK-NEXT:      %c-30_si15 = firrtl.constant(-30 : i15) : !firrtl.sint<15>
// CHECK-NEXT:      firrtl.connect %out_b, %c-30_si15 : !firrtl.flip<sint<15>>, !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop2(%out_b: !firrtl.flip<sint<15>>) {
  %c6_ui7 = firrtl.constant(-6 : i7) : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant(5 : i8) : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.mul %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  firrtl.connect %out_b, %add : !firrtl.flip<sint<15>>, !firrtl.sint<15>
}

// CHECK-LABEL: @mul_cst_prop3
// CHECK-NEXT:      %c30_si15 = firrtl.constant(30 : i15) : !firrtl.sint<15>
// CHECK-NEXT:      firrtl.connect %out_b, %c30_si15 : !firrtl.flip<sint<15>>, !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop3(%out_b: !firrtl.flip<sint<15>>) {
  %c6_ui7 = firrtl.constant(-6 : i7) : !firrtl.sint<7>
  %tmp_a = firrtl.wire : !firrtl.sint<7>
  %c5_ui8 = firrtl.constant(-5 : i8) : !firrtl.sint<8>
  firrtl.connect %tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = firrtl.mul %tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  firrtl.connect %out_b, %add : !firrtl.flip<sint<15>>, !firrtl.sint<15>
}

// CHECK-LABEL: firrtl.module @MuxInvalidOpt
firrtl.module @MuxInvalidOpt(%cond: !firrtl.uint<1>, %data: !firrtl.uint<4>, %out1: !firrtl.flip<uint<4>>, %out2: !firrtl.flip<uint<4>>, %out3: !firrtl.flip<uint<4>>, %out4: !firrtl.flip<uint<4>>) {

  // We can optimize out these mux's since the invalid value can take on any input.
  %tmp1 = firrtl.invalidvalue : !firrtl.uint<4>
  %a = firrtl.mux(%cond, %data, %tmp1) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out1, %data
  firrtl.connect %out1, %a : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  %b = firrtl.mux(%cond, %tmp1, %data) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out2, %data
  firrtl.connect %out2, %b : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  %false = firrtl.constant(false) : !firrtl.uint<1>
  %c = firrtl.mux(%false, %data, %tmp1) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out3, %data
  firrtl.connect %out3, %c : !firrtl.flip<uint<4>>, !firrtl.uint<4>

  %true = firrtl.constant(true) : !firrtl.uint<1>
  %d = firrtl.mux(%false, %tmp1, %data) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK:         firrtl.connect %out4, %data
  firrtl.connect %out4, %d : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}

// CHECK-LABEL: firrtl.module @MuxCanon
firrtl.module @MuxCanon(%c1: !firrtl.uint<1>, %c2: !firrtl.uint<1>, %d1: !firrtl.uint<5>, %d2: !firrtl.uint<5>, %d3: !firrtl.uint<5>, %foo: !firrtl.flip<uint<5>>, %foo2: !firrtl.flip<uint<5>>) {
  %0 = firrtl.mux(%c1, %d2, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %1 = firrtl.mux(%c1, %d1, %0) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %2 = firrtl.mux(%c1, %0, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  firrtl.connect %foo, %1 : !firrtl.flip<uint<5>>, !firrtl.uint<5>
  firrtl.connect %foo2, %2 : !firrtl.flip<uint<5>>, !firrtl.uint<5>
  // CHECK: firrtl.mux(%c1, %d1, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5> 
  // CHECK: firrtl.mux(%c1, %d2, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5> 
}

// CHECK-LABEL: firrtl.module @EmptyNode
firrtl.module @EmptyNode(%d1: !firrtl.uint<5>, %foo: !firrtl.flip<uint<5>>, %foo2: !firrtl.flip<uint<5>>) {
  %bar0 = firrtl.node %d1 : !firrtl.uint<5>
  %bar1 = firrtl.node %d1 : !firrtl.uint<5>
  %bar2 = firrtl.node %d1 {annotations = [{extrastuff = "n1"}]} : !firrtl.uint<5>
  firrtl.connect %foo, %bar1 : !firrtl.flip<uint<5>>, !firrtl.uint<5>
  firrtl.connect %foo2, %bar2 : !firrtl.flip<uint<5>>, !firrtl.uint<5>
}
// CHECK-NEXT: %bar2 = firrtl.node %d1 {annotations = [{extrastuff = "n1"}]}
// CHECK-NEXT: firrtl.connect %foo, %d1
// CHECK-NEXT: firrtl.connect %foo2, %bar2

// COM: https://github.com/llvm/circt/issues/929
// CHECK-LABEL: firrtl.module @MuxInvalidTypeOpt
firrtl.module @MuxInvalidTypeOpt(%in : !firrtl.uint<1>, %out : !firrtl.flip<uint<4>>) {
  %c7_ui4 = firrtl.constant(7 : ui4) : !firrtl.uint<4>
  %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
  %c0_ui2 = firrtl.constant(0 : ui2) : !firrtl.uint<2>
  %0 = firrtl.mux (%in, %c7_ui4, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  %1 = firrtl.mux (%in, %c1_ui2, %0) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out, %1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}
// CHECK: firrtl.mux(%in, %c7_ui4, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
// CHECK: firrtl.mux(%in, %c1_ui2, %0) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

}
