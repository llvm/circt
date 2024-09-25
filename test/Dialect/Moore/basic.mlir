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
  %i1 = moore.variable : <string>
  %i2 = moore.variable : <event>
  %5 = moore.read %i1 : <string>
  %6 = moore.read %i2 : <event>
  %o1, %o2 = moore.instance "ports" @Ports(a: %5: !moore.string, c: %6: !moore.event) -> (b: !moore.string, d: !moore.event)

  // CHECK: %v1 = moore.variable : <i1>
  %v1 = moore.variable : <i1>
  %v2 = moore.variable : <i1>
  // CHECK: %[[TMP1:.+]] = moore.read %v2
  // CHECK: %[[TMP2:.+]] = moore.variable name "v1" %[[TMP1]] : <i1>
  %0 = moore.read %v2 : <i1>
  moore.variable name "v1" %0 : <i1>

  // CHECK: %w0 = moore.net wire : <l1>
  %w0 = moore.net wire : <l1>
  // CHECK: %[[W0:.+]] = moore.read %w0
  %1 = moore.read %w0 : <l1>
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
  moore.procedure initial { moore.return }
  moore.procedure final { moore.return }
  moore.procedure always { moore.return }
  moore.procedure always_comb { moore.return }
  moore.procedure always_latch { moore.return }
  moore.procedure always_ff { moore.return }

  // CHECK: %[[TMP1:.+]] = moore.read %v2
  // CHECK: moore.assign %v1, %[[TMP1]] : i1
  %2 = moore.read %v2 : <i1>
  moore.assign %v1, %2 : i1

  moore.procedure always {
    // CHECK: %[[TMP1:.+]] = moore.read %v2
    // CHECK: moore.blocking_assign %v1, %[[TMP1]] : i1
    %3 = moore.read %v2 : <i1>
    moore.blocking_assign %v1, %3 : i1
    // CHECK: %[[TMP2:.+]] = moore.read %v2
    // CHECK: moore.nonblocking_assign %v1, %[[TMP2]] : i1
    %4 = moore.read %v2 : <i1>
    moore.nonblocking_assign %v1, %4 : i1
    // CHECK: %a = moore.variable : <i32>
    %a = moore.variable : <i32>
    moore.return
  }
}

// CHECK-LABEL: moore.module @Expressions
moore.module @Expressions(
  // CHECK-SAME: in [[A:%[^:]+]] : !moore.i32
  // CHECK-SAME: in [[B:%[^:]+]] : !moore.i32
  in %a: !moore.i32,
  in %b: !moore.i32,
  // CHECK-SAME: in [[C:%[^:]+]] : !moore.l32
  // CHECK-SAME: in [[D:%[^:]+]] : !moore.l3
  in %c: !moore.l32,
  in %d: !moore.l32,
  // CHECK-SAME: in [[X:%[^:]+]] : !moore.i1
  in %x: !moore.i1,

  // CHECK-SAME: in [[ARRAY1:%[^:]+]] : !moore.uarray<4 x i8>
  in %array1: !moore.uarray<4 x i8>,
  // CHECK-SAME: in [[ARRAY2:%[^:]+]] : !moore.uarray<2 x uarray<4 x i8>>
  in %array2: !moore.uarray<2 x uarray<4 x i8>>,

  // CHECK-SAME: in [[REF_A:%[^:]+]] : !moore.ref<i32>
  in %refA: !moore.ref<i32>,
  // CHECK-SAME: in [[REF_B:%[^:]+]] : !moore.ref<i32>
  in %refB: !moore.ref<i32>,
  // CHECK-SAME: in [[REF_C:%[^:]+]] : !moore.ref<l32>
  in %refC: !moore.ref<l32>,
  // CHECK-SAME: in [[REF_D:%[^:]+]] : !moore.ref<l32>
  in %refD: !moore.ref<l32>,
  // CHECK-SAME: in [[REF_ARRAY1:%[^:]+]] : !moore.ref<uarray<4 x i8>>
  in %refArray1: !moore.ref<!moore.uarray<4 x i8>>,
  // CHECK-SAME: in [[REF_ARRAY2:%[^:]+]] : !moore.ref<uarray<2 x uarray<4 x i8>>>
  in %refArray2: !moore.ref<uarray<2 x uarray<4 x i8>>>,

  // CHECK-SAME: in [[STRUCT1:%.+]] : !moore.struct<{a: i32, b: i32}>
  in %struct1: !moore.struct<{a: i32, b: i32}>,
  // CHECK-SAME: in [[REF_STRUCT1:%.+]] : !moore.ref<struct<{a: i32, b: i32}>>
  in %refStruct1: !moore.ref<struct<{a: i32, b: i32}>>
) {
  // CHECK: moore.constant 0 : i0
  moore.constant 0 : i0
  // CHECK: moore.constant 0 : i1
  moore.constant 0 : i1
  // CHECK: moore.constant 1 : i1
  moore.constant 1 : i1
  // CHECK: moore.constant 0 : i32
  moore.constant 0 : i32
  // CHECK: moore.constant -2 : i2
  moore.constant 2 : i2
  // CHECK: moore.constant -2 : i2
  moore.constant -2 : i2
  // CHECK: moore.constant 1311768467463790320 : i64
  moore.constant h123456789ABCDEF0 : i64
  // CHECK: moore.constant h123456789ABCDEF0XZ : l72
  moore.constant h123456789ABCDEF0XZ : l72
  // CHECK: moore.constant 10 : i8
  moore.constant b1010 : i8
  // CHECK: moore.constant b1010XZ : l8
  moore.constant b1010XZ : l8

  // CHECK: moore.conversion [[A]] : !moore.i32 -> !moore.l32
  moore.conversion %a : !moore.i32 -> !moore.l32

  // CHECK: moore.neg [[A]] : i32
  moore.neg %a : i32
  // CHECK: moore.not [[A]] : i32
  moore.not %a : i32

  // CHECK: moore.reduce_and [[A]] : i32 -> i1
  moore.reduce_and %a : i32 -> i1
  // CHECK: moore.reduce_or [[A]] : i32 -> i1
  moore.reduce_or %a : i32 -> i1
  // CHECK: moore.reduce_xor [[A]] : i32 -> i1
  moore.reduce_xor %a : i32 -> i1
  // CHECK: moore.reduce_xor [[C]] : l32 -> l1
  moore.reduce_xor %c : l32 -> l1

  // CHECK: moore.bool_cast [[A]] : i32 -> i1
  moore.bool_cast %a : i32 -> i1
  // CHECK: moore.bool_cast [[C]] : l32 -> l1
  moore.bool_cast %c : l32 -> l1

  // CHECK: moore.add [[A]], [[B]] : i32
  moore.add %a, %b : i32
  // CHECK: moore.sub [[A]], [[B]] : i32
  moore.sub %a, %b : i32
  // CHECK: moore.mul [[A]], [[B]] : i32
  moore.mul %a, %b : i32
  // CHECK: moore.divu [[A]], [[B]] : i32
  moore.divu %a, %b : i32
  // CHECK: moore.divs [[A]], [[B]] : i32
  moore.divs %a, %b : i32
  // CHECK: moore.modu [[A]], [[B]] : i32
  moore.modu %a, %b : i32
  // CHECK: moore.mods [[A]], [[B]] : i32
  moore.mods %a, %b : i32

  // CHECK: moore.and [[A]], [[B]] : i32
  moore.and %a, %b : i32
  // CHECK: moore.or [[A]], [[B]] : i32
  moore.or %a, %b : i32
  // CHECK: moore.xor [[A]], [[B]] : i32
  moore.xor %a, %b : i32

  // CHECK: moore.shl [[C]], [[A]] : l32, i32
  moore.shl %c, %a : l32, i32
  // CHECK: moore.shr [[C]], [[A]] : l32, i32
  moore.shr %c, %a : l32, i32
  // CHECK: moore.ashr [[C]], [[A]] : l32, i32
  moore.ashr %c, %a : l32, i32

  // CHECK: moore.eq [[A]], [[B]] : i32 -> i1
  moore.eq %a, %b : i32 -> i1
  // CHECK: moore.ne [[A]], [[B]] : i32 -> i1
  moore.ne %a, %b : i32 -> i1
  // CHECK: moore.ne [[C]], [[D]] : l32 -> l1
  moore.ne %c, %d : l32 -> l1
  // CHECK: moore.case_eq [[A]], [[B]] : i32
  moore.case_eq %a, %b : i32
  // CHECK: moore.case_ne [[A]], [[B]] : i32
  moore.case_ne %a, %b : i32
  // CHECK: moore.wildcard_eq [[A]], [[B]] : i32 -> i1
  moore.wildcard_eq %a, %b : i32 -> i1
  // CHECK: moore.wildcard_ne [[A]], [[B]] : i32 -> i1
  moore.wildcard_ne %a, %b : i32 -> i1
  // CHECK: moore.wildcard_ne [[C]], [[D]] : l32 -> l1
  moore.wildcard_ne %c, %d : l32 -> l1

  // CHECK: moore.ult [[A]], [[B]] : i32 -> i1
  moore.ult %a, %b : i32 -> i1
  // CHECK: moore.ule [[A]], [[B]] : i32 -> i1
  moore.ule %a, %b : i32 -> i1
  // CHECK: moore.ugt [[A]], [[B]] : i32 -> i1
  moore.ugt %a, %b : i32 -> i1
  // CHECK: moore.uge [[A]], [[B]] : i32 -> i1
  moore.uge %a, %b : i32 -> i1
  // CHECK: moore.slt [[A]], [[B]] : i32 -> i1
  moore.slt %a, %b : i32 -> i1
  // CHECK: moore.sle [[A]], [[B]] : i32 -> i1
  moore.sle %a, %b : i32 -> i1
  // CHECK: moore.sgt [[A]], [[B]] : i32 -> i1
  moore.sgt %a, %b : i32 -> i1
  // CHECK: moore.sge [[A]], [[B]] : i32 -> i1
  moore.sge %a, %b : i32 -> i1
  // CHECK: moore.uge [[C]], [[D]] : l32 -> l1
  moore.uge %c, %d : l32 -> l1

  // CHECK: moore.concat [[A]] : (!moore.i32) -> i32
  moore.concat %a : (!moore.i32) -> i32
  // CHECK: moore.concat [[A]], [[B]] : (!moore.i32, !moore.i32) -> i64
  moore.concat %a, %b : (!moore.i32, !moore.i32) -> i64
  // CHECK: moore.concat [[C]], [[D]], [[C]] : (!moore.l32, !moore.l32, !moore.l32) -> l96
  moore.concat %c, %d, %c : (!moore.l32, !moore.l32, !moore.l32) -> l96

  // CHECK: moore.concat_ref [[REF_A]] : (!moore.ref<i32>) -> <i32>
  moore.concat_ref %refA : (!moore.ref<i32>) -> <i32>
  // CHECK: moore.concat_ref [[REF_A]], [[REF_B]] : (!moore.ref<i32>, !moore.ref<i32>) -> <i64>
  moore.concat_ref %refA, %refB : (!moore.ref<i32>, !moore.ref<i32>) -> <i64>
  // CHECK: moore.concat_ref [[REF_C]], [[REF_D]], [[REF_C]] : (!moore.ref<l32>, !moore.ref<l32>, !moore.ref<l32>) -> <l96>
  moore.concat_ref %refC, %refD, %refC : (!moore.ref<l32>, !moore.ref<l32>, !moore.ref<l32>) -> <l96>

  // CHECK: moore.replicate [[X]] : i1 -> i4
  moore.replicate %x : i1 -> i4

  // CHECK: moore.dyn_extract [[A]] from [[B]] : i32, i32 -> i1
  moore.dyn_extract %a from %b : i32, i32 -> i1
  // CHECK: moore.dyn_extract [[ARRAY2]] from [[A]] : uarray<2 x uarray<4 x i8>>, i32 -> uarray<4 x i8>
  moore.dyn_extract %array2 from %a : uarray<2 x uarray<4 x i8>>, i32 -> uarray<4 x i8>
  // CHECK: moore.dyn_extract [[ARRAY1]] from [[A]] : uarray<4 x i8>, i32 -> i8
  moore.dyn_extract %array1 from %a : uarray<4 x i8>, i32 -> i8

  // CHECK: moore.dyn_extract_ref [[REF_A]] from [[B]] : <i32>, i32 -> <i1>
  moore.dyn_extract_ref %refA from %b : <i32>, i32 -> <i1>
  // CHECK: moore.dyn_extract_ref [[REF_ARRAY2]] from [[A]] : <uarray<2 x uarray<4 x i8>>>, i32 -> <uarray<4 x i8>>
  moore.dyn_extract_ref %refArray2 from %a : <uarray<2 x uarray<4 x i8>>>, i32 -> <uarray<4 x i8>>
  // CHECK: moore.dyn_extract_ref [[REF_ARRAY1]] from [[A]] : <uarray<4 x i8>>, i32 -> <i8>
  moore.dyn_extract_ref %refArray1 from %a : <uarray<4 x i8>>, i32 -> <i8>

  // CHECK: moore.conditional [[X]] : i1 -> i32 {
  // CHECK:   moore.yield [[A]] : i32
  // CHECK: } {
  // CHECK:   moore.yield [[B]] : i32
  // CHECK: }
  moore.conditional %x : i1 -> i32 {
    moore.yield %a : i32
  } {
    moore.yield %b : i32
  }

  // CHECK: moore.array_create [[A]], [[B]] : !moore.i32, !moore.i32 -> array<2 x i32>
  moore.array_create %a, %b : !moore.i32, !moore.i32 -> array<2 x i32>

  // CHECK: moore.struct_create [[A]], [[B]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  moore.struct_create %a, %b : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  // CHECK: moore.struct_extract [[STRUCT1]], "a" : struct<{a: i32, b: i32}> -> i32
  moore.struct_extract %struct1, "a" : struct<{a: i32, b: i32}> -> i32
  // CHECK: moore.struct_extract_ref [[REF_STRUCT1]], "a" : <struct<{a: i32, b: i32}>> -> <i32>
  moore.struct_extract_ref %refStruct1, "a" : <struct<{a: i32, b: i32}>> -> <i32>
  // CHECK: moore.struct_inject [[STRUCT1]], "a", [[B]] : struct<{a: i32, b: i32}>, i32
  moore.struct_inject %struct1, "a", %b : struct<{a: i32, b: i32}>, i32

  moore.output
}

// CHECK-LABEL: moore.module @GraphRegion
moore.module @GraphRegion() {
  %1 = moore.add %0, %0 : i32
  %0 = moore.constant 0 : i32
}

// CHECK-LABEL: func.func @WaitEvent
func.func @WaitEvent(%arg0: !moore.i1, %arg1: !moore.i1) {
  // CHECK: moore.wait_event {
  moore.wait_event {
    // CHECK: moore.detect_event any %arg0 : i1
    moore.detect_event any %arg0 : i1
    // CHECK: moore.detect_event posedge %arg0 : i1
    moore.detect_event posedge %arg0 : i1
    // CHECK: moore.detect_event negedge %arg0 : i1
    moore.detect_event negedge %arg0 : i1
    // CHECK: moore.detect_event edge %arg0 : i1
    moore.detect_event edge %arg0 : i1
    // CHECK: moore.detect_event any %arg0 if %arg1 : i1
    moore.detect_event any %arg0 if %arg1 : i1
  }
  // CHECK: }
  return
}

// CHECK-LABEL: func.func @SimulationControlBuiltins
func.func @SimulationControlBuiltins() {
  // CHECK: moore.builtin.stop
  moore.builtin.stop
  // CHECK: moore.builtin.finish 42
  moore.builtin.finish 42
  // CHECK: moore.builtin.finish_message false
  moore.builtin.finish_message false
  // CHECK: moore.unreachable
  moore.unreachable
}
