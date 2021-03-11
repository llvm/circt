// RUN: circt-opt -canonicalize %s | FileCheck %s

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
firrtl.module @reg_cst_prop1(%clock: !firrtl.clock, %reset: !firrtl.uint<1>, %out_b: !firrtl.flip<uint<8>>) {
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
firrtl.module @reg_cst_prop2(%clock: !firrtl.clock, %reset: !firrtl.uint<1>, %out_b: !firrtl.flip<uint<8>>) {
  %tmp_b = firrtl.reg %clock {name = "tmp_b"} : (!firrtl.clock) -> !firrtl.uint<8>
  firrtl.connect %out_b, %tmp_b : !firrtl.flip<uint<8>>, !firrtl.uint<8>

  %tmp_a = firrtl.reg %clock {name = "tmp_a"} : (!firrtl.clock) -> !firrtl.uint<8>
  %c5_ui8 = firrtl.constant(5 : ui8) : !firrtl.uint<8>
  firrtl.connect %tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  firrtl.connect %tmp_b, %tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
}


}
