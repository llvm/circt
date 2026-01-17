// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

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
    // CHECK: %a = moore.variable : <i32>
    %a = moore.variable : <i32>
    moore.return
  }
}

// CHECK-LABEL: moore.module @ContinuousAssignments
moore.module @ContinuousAssignments(
  in %arg0: !moore.ref<i42>,
  in %arg1: !moore.i42,
  in %arg2: !moore.time
) {
  // CHECK: moore.assign %arg0, %arg1 : i42
  moore.assign %arg0, %arg1 : i42
  // CHECK: moore.delayed_assign %arg0, %arg1, %arg2 : i42
  moore.delayed_assign %arg0, %arg1, %arg2 : i42
}

// CHECK-LABEL: func.func @ProceduralAssignments
func.func @ProceduralAssignments(
  %arg0: !moore.ref<i42>,
  %arg1: !moore.i42,
  %arg2: !moore.time
) {
  // CHECK: moore.blocking_assign %arg0, %arg1 : i42
  moore.blocking_assign %arg0, %arg1 : i42
  // CHECK: moore.nonblocking_assign %arg0, %arg1 : i42
  moore.nonblocking_assign %arg0, %arg1 : i42
  // CHECK: moore.delayed_nonblocking_assign %arg0, %arg1, %arg2 : i42
  moore.delayed_nonblocking_assign %arg0, %arg1, %arg2 : i42
  return
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

  // CHECK-SAME: in %s1 : !moore.string
  in %s1 : !moore.string,
  // CHECK-SAME: in %s2 : !moore.string
  in %s2 : !moore.string,

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
  // CHECK: moore.packed_to_sbv [[STRUCT1]] : struct<{a: i32, b: i32}>
  moore.packed_to_sbv %struct1 : struct<{a: i32, b: i32}>
  // CHECK: moore.sbv_to_packed [[C]] : struct<{u: l16, v: l16}>
  moore.sbv_to_packed %c : struct<{u: l16, v: l16}>
  // CHECK: moore.logic_to_int [[C]] : l32
  moore.logic_to_int %c : l32
  // CHECK: moore.int_to_logic [[A]] : i32
  moore.int_to_logic %a : i32
  // CHECK: moore.to_builtin_bool [[X]] : i1
  moore.to_builtin_bool %x : i1

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
  // CHECK: moore.uarray_cmp eq [[ARRAY1]], [[ARRAY1]] : <4 x i8> -> i1
  moore.uarray_cmp eq %array1, %array1 : <4 x i8> -> i1
  // CHECK: moore.ne [[A]], [[B]] : i32 -> i1
  moore.ne %a, %b : i32 -> i1
  // CHECK: moore.ne [[C]], [[D]] : l32 -> l1
  moore.ne %c, %d : l32 -> l1
  // CHECK: moore.uarray_cmp ne [[ARRAY1]], [[ARRAY1]] : <4 x i8> -> i1
  moore.uarray_cmp ne %array1, %array1 : <4 x i8> -> i1
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

  // CHECK: moore.constant_string "Test" : i128
  moore.constant_string "Test" : i128
  // CHECK: moore.constant_string "" : i128
  moore.constant_string "" : i128
  // CHECK: moore.string_cmp eq %s1, %s2 : string -> i1
  moore.string_cmp eq %s1, %s2 : string -> i1
  // CHECK: moore.string_cmp ne %s1, %s2 : string -> i1
  moore.string_cmp ne %s1, %s2 : string -> i1
  // CHECK: moore.string_cmp lt %s1, %s2 : string -> i1
  moore.string_cmp lt %s1, %s2 : string -> i1
  // CHECK: moore.string_cmp le %s1, %s2 : string -> i1
  moore.string_cmp le %s1, %s2 : string -> i1
  // CHECK: moore.string_cmp gt %s1, %s2 : string -> i1
  moore.string_cmp gt %s1, %s2 : string -> i1
  // CHECK: moore.string_cmp ge %s1, %s2 : string -> i1
  moore.string_cmp ge %s1, %s2 : string -> i1

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

// CHECK-LABEL: func.func @WaitDelay
func.func @WaitDelay(%arg0: !moore.time) {
  // CHECK: moore.wait_delay %arg0
  moore.wait_delay %arg0
  return
}

// CHECK-LABEL: func.func @FormatStrings
// CHECK-SAME: %arg0: !moore.format_string
func.func @FormatStrings(%arg0: !moore.format_string, %arg1: !moore.i42, %arg2: !moore.f32, %arg3: !moore.f64) {
  // CHECK: moore.fmt.literal "hello"
  moore.fmt.literal "hello"
  // CHECK: moore.fmt.concat ()
  moore.fmt.concat ()
  // CHECK: moore.fmt.concat (%arg0)
  moore.fmt.concat (%arg0)
  // CHECK: moore.fmt.concat (%arg0, %arg0)
  moore.fmt.concat (%arg0, %arg0)
  // CHECK: moore.fmt.int binary %arg1, align left, pad zero width 42 : i42
  moore.fmt.int binary %arg1, align left, pad zero width 42 : i42
  // CHECK: moore.fmt.int binary %arg1, align right, pad zero width 42 : i42
  moore.fmt.int binary %arg1, align right, pad zero width 42 : i42
  // CHECK: moore.fmt.int binary %arg1, align right, pad space width 42 : i42
  moore.fmt.int binary %arg1, align right, pad space width 42 : i42
  // CHECK: moore.fmt.int octal %arg1, align left, pad zero width 42 : i42
  moore.fmt.int octal %arg1, align left, pad zero width 42 : i42
  // CHECK: moore.fmt.int decimal %arg1, align left, pad zero width 42 signed : i42
  moore.fmt.int decimal %arg1, align left, pad zero width 42 signed : i42
  // CHECK: moore.fmt.int decimal %arg1, align left, pad zero signed : i42
  moore.fmt.int decimal %arg1, align left, pad zero signed : i42
  // CHECK: moore.fmt.int decimal %arg1, align left, pad zero width 42 : i42
  moore.fmt.int decimal %arg1, align left, pad zero width 42 : i42
  // CHECK: moore.fmt.int hex_lower %arg1, align left, pad zero width 42 : i42
  moore.fmt.int hex_lower %arg1, align left, pad zero width 42 : i42
  // CHECK: moore.fmt.int hex_upper %arg1, align left, pad zero width 42 : i42
  moore.fmt.int hex_upper %arg1, align left, pad zero width 42 : i42

  // CHECK: moore.fmt.real float %arg2, align left : f32
  moore.fmt.real float %arg2, align left : f32
  // CHECK: moore.fmt.real exponential %arg3, align left : f64
  moore.fmt.real exponential %arg3, align left : f64
  // CHECK: moore.fmt.real general %arg3, align right fieldWidth 9 fracDigits 8 : f64
  moore.fmt.real general %arg3, align right fieldWidth 9 fracDigits 8 : f64
  // CHECK: moore.fmt.real float %arg2, align right fieldWidth 12 : f32
  moore.fmt.real float %arg2, align right fieldWidth 12 : f32
  // CHECK: moore.fmt.real exponential %arg3, align right fracDigits 5 : f64
  moore.fmt.real exponential %arg3, align right fracDigits 5 : f64
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

// CHECK-LABEL: func.func @SeverityAndDisplayBuiltins
func.func @SeverityAndDisplayBuiltins(%arg0: !moore.format_string) {
  // CHECK: moore.builtin.display %arg0
  moore.builtin.display %arg0
  // CHECK: moore.builtin.severity info %arg0
  moore.builtin.severity info %arg0
  // CHECK: moore.builtin.severity warning %arg0
  moore.builtin.severity warning %arg0
  // CHECK: moore.builtin.severity error %arg0
  moore.builtin.severity error %arg0
  // CHECK: moore.builtin.severity fatal %arg0
  moore.builtin.severity fatal %arg0
  return
}

// CHECK-LABEL: func.func @MathBuiltins
func.func @MathBuiltins(%arg0: !moore.i32, %arg1: !moore.l42) {
  // CHECK: moore.builtin.clog2 %arg0 : i32
  moore.builtin.clog2 %arg0 : i32
  // CHECK: moore.builtin.clog2 %arg1 : l42
  moore.builtin.clog2 %arg1 : l42
  return
}

// CHECK-LABEL: func.func @TimeConversion
func.func @TimeConversion(%arg0: !moore.time, %arg1: !moore.l64) {
  // CHECK: moore.packed_to_sbv %arg0 : time
  moore.packed_to_sbv %arg0 : time
  // CHECK: moore.sbv_to_packed %arg1 : time
  moore.sbv_to_packed %arg1 : time
  // CHECK: moore.time_to_logic %arg0
  moore.time_to_logic %arg0
  // CHECK: moore.logic_to_time %arg1
  moore.logic_to_time %arg1
  return
}

// CHECK-LABEL: func.func @RealConversion32(%arg0: !moore.f32, %arg1: !moore.i42, %arg2: !moore.f64)
func.func @RealConversion32(%arg0: !moore.f32, %arg1: !moore.i42, %arg2: !moore.f64) {
  // CHECK: moore.real_to_int %arg0 : f32 -> i42
  %0 = moore.real_to_int %arg0 : f32 -> i42
  // CHECK: moore.sint_to_real %arg1 : i42 -> f32
  %1 = moore.sint_to_real %arg1 : i42 -> f32
  // CHECK: moore.convert_real %arg0 : f32 -> f64
  %2 = moore.convert_real %arg0 : f32 -> f64
  // CHECK: moore.convert_real %arg2 : f64 -> f32
  %3 = moore.convert_real %arg2 : f64 -> f32
  return
}

// CHECK-LABEL: func.func @RealConversion64(%arg0: !moore.f64, %arg1: !moore.i42)
func.func @RealConversion64(%arg0: !moore.f64, %arg1: !moore.i42) {
  // CHECK: moore.real_to_int %arg0 : f64 -> i42
  %0 = moore.real_to_int %arg0 : f64 -> i42
  // CHECK: moore.uint_to_real %arg1 : i42 -> f64
  %1 = moore.uint_to_real %arg1 : i42 -> f64
  return
}

// CHECK-LABEL: moore.global_variable @GlobalVar1 : !moore.i42
moore.global_variable @GlobalVar1 : !moore.i42

// CHECK: moore.get_global_variable @GlobalVar1 : <i42>
moore.get_global_variable @GlobalVar1 : <i42>

// CHECK-LABEL: moore.global_variable @GlobalVar2 : !moore.i42
moore.global_variable @GlobalVar2 : !moore.i42 init {
  // CHECK-NEXT: moore.constant
  %0 = moore.constant 9001 : i42
  // CHECK-NEXT: moore.yield
  moore.yield %0 : !moore.i42
}

// CHECK: moore.get_global_variable @GlobalVar2 : <i42>
moore.get_global_variable @GlobalVar2 : <i42>

// CHECK-LABEL: func.func @StringConversion
// CHECK-SAME: [[A:%.+]]: !moore.i32
// CHECK-SAME: [[B:%.+]]: !moore.string
func.func @StringConversion(%a: !moore.i32, %b: !moore.string) { 
  // CHECK: moore.int_to_string [[A]] : i32
  moore.int_to_string %a : i32
  // CHECK: moore.string_to_int [[B]] : i32
  moore.string_to_int %b : i32
  return
}

// CHECK-LABEL: func.func @StringOperations
func.func @StringOperations(%arg0 : !moore.string, %arg1 : !moore.string, %arg2 : !moore.string) {
  // CHECK: [[EMPTY:%.+]] = moore.string.concat ()
  %empty = moore.string.concat ()
  // CHECK: moore.string.concat (%arg0)
  %single = moore.string.concat (%arg0)
  // CHECK: [[TWO:%.+]] = moore.string.concat (%arg0, %arg1)
  %two = moore.string.concat (%arg0, %arg1)
  // CHECK: moore.string.concat (%arg0, %arg1, %arg2)
  %three = moore.string.concat (%arg0, %arg1, %arg2)
  // CHECK: [[NESTED:%.+]] = moore.string.concat ([[TWO]], %arg2)
  %nested = moore.string.concat (%two, %arg2)
  // CHECK: moore.string.len %arg0
  %len1 = moore.string.len %arg0
  // CHECK: moore.string.len %arg1
  %len2 = moore.string.len %arg1
  // CHECK: moore.string.len [[EMPTY]]
  %len_empty = moore.string.len %empty
  // CHECK: moore.string.len [[TWO]]
  %len_concat = moore.string.len %two
  // CHECK: moore.string.len [[NESTED]]
  %len_nested = moore.string.len %nested
  
  return
}
