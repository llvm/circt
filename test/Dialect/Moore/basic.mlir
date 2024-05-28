// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: moore.module @Empty()
moore.module @Empty() {
  // CHECK: moore.output
}

// CHECK-LABEL: moore.module @Ports
moore.module @Ports(
  // CHECK-SAME: in %a : !moore.string
  in %a : !moore.string,
  // CHECK-SAME: out b : !moore.string
  out b : !moore.string,
  // CHECK-SAME: in %c : !moore.event
  in %c : !moore.event,
  // CHECK-SAME: out d : !moore.event
  out d : !moore.event
) {
  // CHECK: moore.output %a, %c : !moore.string, !moore.event
  moore.output %a, %c : !moore.string, !moore.event
}

// CHECK-LABEL: moore.module @Module
moore.module @Module() {
  // CHECK: moore.instance "empty" @Empty() -> ()
  moore.instance "empty" @Empty() -> ()

  // CHECK: %[[I1_READ:.+]] = moore.read %i1
  // CHECK: %[[I2_READ:.+]] = moore.read %i2 
  // CHECK: moore.instance "ports" @Ports(a: %[[I1_READ]]: !moore.string, c: %[[I2_READ]]: !moore.event) -> (b: !moore.string, d: !moore.event)
  %i1 = moore.variable : <!moore.string>
  %i2 = moore.variable : <!moore.event>
  %5 = moore.read %i1 : !moore.string
  %6 = moore.read %i2 : !moore.event
  %o1, %o2 = moore.instance "ports" @Ports(a: %5: !moore.string, c: %6: !moore.event) -> (b: !moore.string, d: !moore.event)

  // CHECK: %v1 = moore.variable : <i1>
  %v1 = moore.variable : <i1>
  %v2 = moore.variable : <i1>
  // CHECK: %[[TMP1:.+]] = moore.read %v2 : i1
  // CHECK: %[[TMP2:.+]] = moore.variable name "v1" %[[TMP1]] : <i1>
  %0 = moore.read %v2 : i1
  moore.variable name "v1" %0 : <i1>

  // CHECK: %w0 = moore.net wire : <l1>
  %w0 = moore.net wire : <l1>
  // CHECK: %[[W0:.+]] = moore.read %w0 : l1
  %1 = moore.read %w0 : l1
  // CHECK: %w1 = moore.net wire %[[W0]] : <l1>
  %w1 = moore.net wire %1 : <l1>
  // CHECK: %w2 = moore.net uwire %[[W0]] : <l1>
  %w2 = moore.net uwire %1 : <l1>
  // CHECK: %w3 = moore.net tri %[[W0]] : <l1>
  %w3 = moore.net tri %1 : <l1>
  // CHECK: %w4 = moore.net triand %[[W0]] : <l1>
  %w4 = moore.net triand %1 : <l1>
  // CHECK: %w5 = moore.net trior %[[W0]] : <l1>
  %w5 = moore.net trior %1 : <l1>
  // CHECK: %w6 = moore.net wand %[[W0]] : <l1>
  %w6 = moore.net wand %1 : <l1>
  // CHECK: %w7 = moore.net wor %[[W0]] : <l1>
  %w7 = moore.net wor %1 : <l1>
  // CHECK: %w8 = moore.net trireg %[[W0]] : <l1>
  %w8 = moore.net trireg %1 : <l1>
  // CHECK: %w9 = moore.net tri0 %[[W0]] : <l1>
  %w9 = moore.net tri0 %1 : <l1>
  // CHECK: %w10 = moore.net tri1 %[[W0]] : <l1>
  %w10 = moore.net tri1 %1 : <l1>
  // CHECK: %w11 = moore.net supply0 : <l1>
  %w11 = moore.net supply0 : <l1>
  // CHECK: %w12 = moore.net supply1 : <l1>
  %w12 = moore.net supply1 : <l1>

  // CHECK: moore.procedure initial {
  // CHECK: moore.procedure final {
  // CHECK: moore.procedure always {
  // CHECK: moore.procedure always_comb {
  // CHECK: moore.procedure always_latch {
  // CHECK: moore.procedure always_ff {
  moore.procedure initial {}
  moore.procedure final {}
  moore.procedure always {}
  moore.procedure always_comb {}
  moore.procedure always_latch {}
  moore.procedure always_ff {}

  // CHECK: %[[TMP1:.+]] = moore.read %v2 : i1
  // CHECK: moore.assign %v1, %[[TMP1]] : i1
  %2 = moore.read %v2 : i1
  moore.assign %v1, %2 : i1

  moore.procedure always {
    // CHECK: %[[TMP1:.+]] = moore.read %v2 : i1
    // CHECK: moore.blocking_assign %v1, %[[TMP1]] : i1
    %3 = moore.read %v2 : i1
    moore.blocking_assign %v1, %3 : i1
    // CHECK: %[[TMP2:.+]] = moore.read %v2 : i1
    // CHECK: moore.nonblocking_assign %v1, %[[TMP2]] : i1
    %4 = moore.read %v2 : i1
    moore.nonblocking_assign %v1, %4 : i1
    // CHECK: %a = moore.variable : <i32>
    %a = moore.variable : <i32>
  }
}

// CHECK-LABEL: moore.module @Expressions
moore.module @Expressions() {
  %b1 = moore.variable : <i1>
  %l1 = moore.variable : <l1>
  %b5 = moore.variable : <i5>
  %int = moore.variable : <i32>
  %int2 = moore.variable : <i32>
  %integer = moore.variable : <l32>
  %integer2 = moore.variable : <l32>
  %arr = moore.variable : <uarray<2 x uarray<4 x i8>>>

  // CHECK: %[[b1:.+]] = moore.read %b1 : i1
  // CHECK: %[[l1:.+]] = moore.read %l1 : l1
  // CHECK: %[[b5:.+]] = moore.read %b5 : i5
  // CHECK: %[[int:.+]] = moore.read %int : i32
  // CHECK: %[[int2:.+]] = moore.read %int2 : i32
  // CHECK: %[[integer:.+]] = moore.read %integer : l32
  // CHECK: %[[integer2:.+]] = moore.read %integer2 : l32
  // CHECK: %[[arr:.+]] = moore.read %arr : uarray<2 x uarray<4 x i8>>
  %0 = moore.read %b1 : i1
  %1 = moore.read %l1 : l1
  %2 = moore.read %b5 : i5
  %3 = moore.read %int : i32
  %4 = moore.read %int2 : i32
  %5 = moore.read %integer : l32
  %6 = moore.read %integer2 : l32
  %7 = moore.read %arr : uarray<2 x uarray<4 x i8>>

  // CHECK: moore.constant 0 : i32
  moore.constant 0 : i32
  // CHECK: moore.constant -2 : i2
  moore.constant 2 : i2
  // CHECK: moore.constant -2 : i2
  moore.constant -2 : i2

  // CHECK: moore.conversion %[[b5]] : !moore.i5 -> !moore.l5
  moore.conversion %2 : !moore.i5 -> !moore.l5

  // CHECK: moore.neg %[[int]] : i32
  moore.neg %3 : i32
  // CHECK: moore.not %[[int]] : i32
  moore.not %3 : i32

  // CHECK: moore.reduce_and %[[int]] : i32 -> i1
  moore.reduce_and %3 : i32 -> i1
  // CHECK: moore.reduce_or %[[int]] : i32 -> i1
  moore.reduce_or %3 : i32 -> i1
  // CHECK: moore.reduce_xor %[[int]] : i32 -> i1
  moore.reduce_xor %3 : i32 -> i1
  // CHECK: moore.reduce_xor %[[integer]] : l32 -> l1
  moore.reduce_xor %5 : l32 -> l1

  // CHECK: moore.bool_cast %[[int]] : i32 -> i1
  moore.bool_cast %3 : i32 -> i1
  // CHECK: moore.bool_cast %[[integer]] : l32 -> l1
  moore.bool_cast %5 : l32 -> l1

  // CHECK: moore.add %[[int]], %[[int2]] : i32
  moore.add %3, %4 : i32
  // CHECK: moore.sub %[[int]], %[[int2]] : i32
  moore.sub %3, %4 : i32
  // CHECK: moore.mul %[[int]], %[[int2]] : i32
  moore.mul %3, %4 : i32
  // CHECK: moore.divu %[[int]], %[[int2]] : i32
  moore.divu %3, %4 : i32
  // CHECK: moore.divs %[[int]], %[[int2]] : i32
  moore.divs %3, %4 : i32
  // CHECK: moore.modu %[[int]], %[[int2]] : i32
  moore.modu %3, %4 : i32
  // CHECK: moore.mods %[[int]], %[[int2]] : i32
  moore.mods %3, %4 : i32

  // CHECK: moore.and %[[int]], %[[int2]] : i32
  moore.and %3, %4 : i32
  // CHECK: moore.or %[[int]], %[[int2]] : i32
  moore.or %3, %4 : i32
  // CHECK: moore.xor %[[int]], %[[int2]] : i32
  moore.xor %3, %4 : i32

  // CHECK: moore.shl %[[l1]], %[[b1]] : l1, i1
  moore.shl %1, %0 : l1, i1
  // CHECK: moore.shr %[[l1]], %[[b1]] : l1, i1
  moore.shr %1, %0 : l1, i1
  // CHECK: moore.ashr %[[b5]], %[[b1]] : i5, i1
  moore.ashr %2, %0 : i5, i1

  // CHECK: moore.eq %[[int]], %[[int2]] : i32 -> i1
  moore.eq %3, %4 : i32 -> i1
  // CHECK: moore.ne %[[int]], %[[int2]] : i32 -> i1
  moore.ne %3, %4 : i32 -> i1
  // CHECK: moore.ne %[[integer]], %[[integer2]] : l32 -> l1
  moore.ne %5, %6 : l32 -> l1
  // CHECK: moore.case_eq %[[int]], %[[int2]] : i32
  moore.case_eq %3, %4 : i32
  // CHECK: moore.case_ne %[[int]], %[[int2]] : i32
  moore.case_ne %3, %4 : i32
  // CHECK: moore.wildcard_eq %[[int]], %[[int2]] : i32 -> i1
  moore.wildcard_eq %3, %4 : i32 -> i1
  // CHECK: moore.wildcard_ne %[[int]], %[[int2]] : i32 -> i1
  moore.wildcard_ne %3, %4 : i32 -> i1
  // CHECK: moore.wildcard_ne %[[integer]], %[[integer2]] : l32 -> l1
  moore.wildcard_ne %5, %6 : l32 -> l1

  // CHECK: moore.ult %[[int]], %[[int2]] : i32 -> i1
  moore.ult %3, %4 : i32 -> i1
  // CHECK: moore.ule %[[int]], %[[int2]] : i32 -> i1
  moore.ule %3, %4 : i32 -> i1
  // CHECK: moore.ugt %[[int]], %[[int2]] : i32 -> i1
  moore.ugt %3, %4 : i32 -> i1
  // CHECK: moore.uge %[[int]], %[[int2]] : i32 -> i1
  moore.uge %3, %4 : i32 -> i1
  // CHECK: moore.slt %[[int]], %[[int2]] : i32 -> i1
  moore.slt %3, %4 : i32 -> i1
  // CHECK: moore.sle %[[int]], %[[int2]] : i32 -> i1
  moore.sle %3, %4 : i32 -> i1
  // CHECK: moore.sgt %[[int]], %[[int2]] : i32 -> i1
  moore.sgt %3, %4 : i32 -> i1
  // CHECK: moore.sge %[[int]], %[[int2]] : i32 -> i1
  moore.sge %3, %4 : i32 -> i1
  // CHECK: moore.uge %[[integer]], %[[integer2]] : l32 -> l1
  moore.uge %5, %6 : l32 -> l1

  // CHECK: moore.concat %[[b1]] : (!moore.i1) -> i1
  moore.concat %0 : (!moore.i1) -> i1
  // CHECK: moore.concat %[[b5]], %[[b1]] : (!moore.i5, !moore.i1) -> i6
  moore.concat %2, %0 : (!moore.i5, !moore.i1) -> i6
  // CHECK: moore.concat %[[l1]], %[[l1]], %[[l1]] : (!moore.l1, !moore.l1, !moore.l1) -> l3
  moore.concat %1, %1, %1 : (!moore.l1, !moore.l1, !moore.l1) -> l3
  // CHECK: moore.concat_ref %b1 : (!moore.ref<i1>) -> <i1>
  moore.concat_ref %b1 : (!moore.ref<i1>) -> <i1>
  // CHECK: moore.concat_ref %b5, %b1 : (!moore.ref<i5>, !moore.ref<i1>) -> <i6>
  moore.concat_ref %b5, %b1 : (!moore.ref<i5>, !moore.ref<i1>) -> <i6>
  // CHECK: moore.concat_ref %l1, %l1, %l1 : (!moore.ref<l1>, !moore.ref<l1>, !moore.ref<l1>) -> <l3>
  moore.concat_ref %l1, %l1, %l1 : (!moore.ref<l1>, !moore.ref<l1>, !moore.ref<l1>) -> <l3>
  // CHECK: moore.replicate %[[b1]] : i1 -> i4
  moore.replicate %0 : i1 -> i4

  // CHECK: moore.extract %[[b5]] from %[[b1]] : i5, i1 -> i1
  moore.extract %2 from %0 : i5, i1 -> i1
  // CHECK: %[[VAL1:.*]] = moore.constant 0 : i32
  // CHECK: %[[VAL2:.*]] = moore.extract %[[arr]] from %[[VAL1]] : uarray<2 x uarray<4 x i8>>, i32 -> uarray<4 x i8>
  %11 = moore.constant 0 : i32
  %12 = moore.extract %7 from %11 : uarray<2 x uarray<4 x i8>>, i32 -> uarray<4 x i8>
  // CHECK: %[[VAL3:.*]] = moore.constant 3 : i32
  // CHECK: %[[VAL4:.*]] = moore.extract %[[VAL2]] from %[[VAL3]] : uarray<4 x i8>, i32 -> i8
  %13 = moore.constant 3 : i32
  %14 = moore.extract %12 from %13 : uarray<4 x i8>, i32 -> i8
  // CHECK: %[[VAL5:.*]] = moore.constant 2 : i32
  // CHECK: moore.extract %[[VAL4]] from %[[VAL5]] : i8, i32 -> i5
  %15 = moore.constant 2 : i32
  moore.extract %14 from %15 : i8, i32 -> i5

  // CHECK: moore.extract_ref %b5 from %[[b1]] : <i5>, i1 -> <i1>
  moore.extract_ref %b5 from %0 : <i5>, i1 -> <i1>
  // CHECK: %[[TMP1:.+]] = moore.constant 0
  // CHECK: %[[TMP2:.+]] = moore.extract_ref %arr from %[[TMP1]] : <uarray<2 x uarray<4 x i8>>>, i32 -> <uarray<4 x i8>>
  %16 = moore.constant 0 : i32
  %17 = moore.extract_ref %arr from %16 : <uarray<2 x uarray<4 x i8>>>, i32 -> <uarray<4 x i8>>
  // CHECK: %[[TMP3:.+]] = moore.constant 3
  // CHECK: %[[TMP4:.+]] = moore.extract_ref %[[TMP2]] from %[[TMP3]] : <uarray<4 x i8>>, i32 -> <i8>
  %18 = moore.constant 3 : i32
  %19 = moore.extract_ref %17 from %18 : <uarray<4 x i8>>, i32 -> <i8>
  // CHECK: %[[TMP5:.+]] = moore.constant 2
  // CHECK: extract_ref %[[TMP4]] from %[[TMP5]] : <i8>, i32 -> <i4>
  %20 = moore.constant 2 : i32
  moore.extract_ref %19 from %20 : <i8>, i32 -> <i4>

  // CHECK: %[[B1_COND:.+]] = moore.read %b1
  // CHECK: moore.conditional %[[B1_COND]] : i1 -> i32 {
  // CHECK:   %[[INT_READ:.+]] = moore.read %int
  // CHECK:   moore.yield %[[INT_READ]] : i32
  // CHECK: } {
  // CHECK:   %[[INT2_READ:.+]] = moore.read %int2
  // CHECK:   moore.yield %[[INT2_READ]] : i32
  // CHECK: }
  %21 = moore.read %b1 : i1
  moore.conditional %21 : i1 -> i32 {
    %22 = moore.read %int : i32
    moore.yield %22 : i32
  } {
    %22 = moore.read %int2 : i32
    moore.yield %22 : i32
  }
}
