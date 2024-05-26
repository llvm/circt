// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: moore.module @Foo
moore.module @Foo {
  // CHECK: moore.instance "foo" @Foo
  moore.instance "foo" @Foo
  // CHECK: %v1 = moore.variable : i1
  %v1 = moore.variable : i1
  %v2 = moore.variable : i1
  // CHECK: [[TMP:%.+]] = moore.variable name "v1" %v2 : i1
  moore.variable name "v1" %v2 : i1

  // CHECK: %w0 = moore.net wire : l1
  %w0 = moore.net wire : l1
  // CHECK: %w1 = moore.net wire %w0 : l1
  %w1 = moore.net wire %w0 : l1
  // CHECK: %w2 = moore.net uwire %w0 : l1
  %w2 = moore.net uwire %w0 : l1
  // CHECK: %w3 = moore.net tri %w0 : l1
  %w3 = moore.net tri %w0 : l1
  // CHECK: %w4 = moore.net triand %w0 : l1
  %w4 = moore.net triand %w0 : l1
  // CHECK: %w5 = moore.net trior %w0 : l1
  %w5 = moore.net trior %w0 : l1
  // CHECK: %w6 = moore.net wand %w0 : l1
  %w6 = moore.net wand %w0 : l1
  // CHECK: %w7 = moore.net wor %w0 : l1
  %w7 = moore.net wor %w0 : l1
  // CHECK: %w8 = moore.net trireg %w0 : l1
  %w8 = moore.net trireg %w0 : l1
  // CHECK: %w9 = moore.net tri0 %w0 : l1
  %w9 = moore.net tri0 %w0 : l1
  // CHECK: %w10 = moore.net tri1 %w0 : l1
  %w10 = moore.net tri1 %w0 : l1
  // CHECK: %w11 = moore.net supply0 : l1
  %w11 = moore.net supply0 : l1
  // CHECK: %w12 = moore.net supply1 : l1
  %w12 = moore.net supply1 : l1

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

  // CHECK: moore.assign %v1, %v2 : i1
  moore.assign %v1, %v2 : i1

  moore.procedure always {
    // CHECK: moore.blocking_assign %v1, %v2 : i1
    moore.blocking_assign %v1, %v2 : i1
    // CHECK: moore.nonblocking_assign %v1, %v2 : i1
    moore.nonblocking_assign %v1, %v2 : i1
    // CHECK: %a = moore.variable : i32
    %a = moore.variable : i32
  }
}

// CHECK-LABEL: moore.module @Bar
moore.module @Bar {
}

// CHECK-LABEL: moore.module @Expressions
moore.module @Expressions {
  %b1 = moore.variable : i1
  %l1 = moore.variable : l1
  %b5 = moore.variable : i5
  %int = moore.variable : i32
  %int2 = moore.variable : i32
  %integer = moore.variable : l32
  %integer2 = moore.variable : l32
  %arr = moore.variable : uarray<2 x uarray<4 x i8>>

  // CHECK: moore.constant 0 : i32
  moore.constant 0 : i32
  // CHECK: moore.constant -2 : i2
  moore.constant 2 : i2
  // CHECK: moore.constant -2 : i2
  moore.constant -2 : i2

  // CHECK: moore.conversion %b5 : !moore.i5 -> !moore.l5
  moore.conversion %b5 : !moore.i5 -> !moore.l5

  // CHECK: moore.neg %int : i32
  moore.neg %int : i32
  // CHECK: moore.not %int : i32
  moore.not %int : i32

  // CHECK: moore.reduce_and %int : i32 -> i1
  moore.reduce_and %int : i32 -> i1
  // CHECK: moore.reduce_or %int : i32 -> i1
  moore.reduce_or %int : i32 -> i1
  // CHECK: moore.reduce_xor %int : i32 -> i1
  moore.reduce_xor %int : i32 -> i1
  // CHECK: moore.reduce_xor %integer : l32 -> l1
  moore.reduce_xor %integer : l32 -> l1

  // CHECK: moore.bool_cast %int : i32 -> i1
  moore.bool_cast %int : i32 -> i1
  // CHECK: moore.bool_cast %integer : l32 -> l1
  moore.bool_cast %integer : l32 -> l1

  // CHECK: moore.add %int, %int2 : i32
  moore.add %int, %int2 : i32
  // CHECK: moore.sub %int, %int2 : i32
  moore.sub %int, %int2 : i32
  // CHECK: moore.mul %int, %int2 : i32
  moore.mul %int, %int2 : i32
  // CHECK: moore.divu %int, %int2 : i32
  moore.divu %int, %int2 : i32
  // CHECK: moore.divs %int, %int2 : i32
  moore.divs %int, %int2 : i32
  // CHECK: moore.modu %int, %int2 : i32
  moore.modu %int, %int2 : i32
  // CHECK: moore.mods %int, %int2 : i32
  moore.mods %int, %int2 : i32

  // CHECK: moore.and %int, %int2 : i32
  moore.and %int, %int2 : i32
  // CHECK: moore.or %int, %int2 : i32
  moore.or %int, %int2 : i32
  // CHECK: moore.xor %int, %int2 : i32
  moore.xor %int, %int2 : i32

  // CHECK: moore.shl %l1, %b1 : l1, i1
  moore.shl %l1, %b1 : l1, i1
  // CHECK: moore.shr %l1, %b1 : l1, i1
  moore.shr %l1, %b1 : l1, i1
  // CHECK: moore.ashr %b5, %b1 : i5, i1
  moore.ashr %b5, %b1 : i5, i1

  // CHECK: moore.eq %int, %int2 : i32 -> i1
  moore.eq %int, %int2 : i32 -> i1
  // CHECK: moore.ne %int, %int2 : i32 -> i1
  moore.ne %int, %int2 : i32 -> i1
  // CHECK: moore.ne %integer, %integer2 : l32 -> l1
  moore.ne %integer, %integer2 : l32 -> l1
  // CHECK: moore.case_eq %int, %int2 : i32
  moore.case_eq %int, %int2 : i32
  // CHECK: moore.case_ne %int, %int2 : i32
  moore.case_ne %int, %int2 : i32
  // CHECK: moore.wildcard_eq %int, %int2 : i32 -> i1
  moore.wildcard_eq %int, %int2 : i32 -> i1
  // CHECK: moore.wildcard_ne %int, %int2 : i32 -> i1
  moore.wildcard_ne %int, %int2 : i32 -> i1
  // CHECK: moore.wildcard_ne %integer, %integer2 : l32 -> l1
  moore.wildcard_ne %integer, %integer2 : l32 -> l1

  // CHECK: moore.ult %int, %int2 : i32 -> i1
  moore.ult %int, %int2 : i32 -> i1
  // CHECK: moore.ule %int, %int2 : i32 -> i1
  moore.ule %int, %int2 : i32 -> i1
  // CHECK: moore.ugt %int, %int2 : i32 -> i1
  moore.ugt %int, %int2 : i32 -> i1
  // CHECK: moore.uge %int, %int2 : i32 -> i1
  moore.uge %int, %int2 : i32 -> i1
  // CHECK: moore.slt %int, %int2 : i32 -> i1
  moore.slt %int, %int2 : i32 -> i1
  // CHECK: moore.sle %int, %int2 : i32 -> i1
  moore.sle %int, %int2 : i32 -> i1
  // CHECK: moore.sgt %int, %int2 : i32 -> i1
  moore.sgt %int, %int2 : i32 -> i1
  // CHECK: moore.sge %int, %int2 : i32 -> i1
  moore.sge %int, %int2 : i32 -> i1
  // CHECK: moore.uge %integer, %integer2 : l32 -> l1
  moore.uge %integer, %integer2 : l32 -> l1

  // CHECK: moore.concat %b1 : (!moore.i1) -> i1
  moore.concat %b1 : (!moore.i1) -> i1
  // CHECK: moore.concat %b5, %b1 : (!moore.i5, !moore.i1) -> i6
  moore.concat %b5, %b1 : (!moore.i5, !moore.i1) -> i6
  // CHECK: moore.concat %l1, %l1, %l1 : (!moore.l1, !moore.l1, !moore.l1) -> l3
  moore.concat %l1, %l1, %l1 : (!moore.l1, !moore.l1, !moore.l1) -> l3
  // CHECK: moore.replicate %b1 : i1 -> i4
  moore.replicate %b1 : i1 -> i4

  // CHECK: moore.extract %b5 from %b1 : i5, i1 -> i1
  moore.extract %b5 from %b1 : i5, i1 -> i1
  // CHECK: [[VAL1:%.*]] = moore.constant 0 : i32
  // CHECK: [[VAL2:%.*]] = moore.extract %arr from [[VAL1]] : uarray<2 x uarray<4 x i8>>, i32 -> uarray<4 x i8>
  %1 = moore.constant 0 : i32
  %2 = moore.extract %arr from %1 : uarray<2 x uarray<4 x i8>>, i32 -> uarray<4 x i8>
  // CHECK: [[VAL3:%.*]] = moore.constant 3 : i32
  // CHECK: [[VAL4:%.*]] = moore.extract [[VAL2]] from [[VAL3]] : uarray<4 x i8>, i32 -> i8
  %3 = moore.constant 3 : i32
  %4 = moore.extract %2 from %3 : uarray<4 x i8>, i32 -> i8
  // CHECK: [[VAL5:%.*]] = moore.constant 2 : i32
  // CHECK: moore.extract [[VAL4]] from [[VAL5]] : i8, i32 -> i5
  %5 = moore.constant 2 : i32
  moore.extract %4 from %5 : i8, i32 -> i5
}
