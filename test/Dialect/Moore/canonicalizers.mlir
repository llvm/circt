// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @Casts
func.func @Casts(%arg0: !moore.i1) -> (!moore.i1, !moore.i1) {
  // CHECK-NOT: moore.conversion
  // CHECK-NOT: moore.bool_cast
  %0 = moore.conversion %arg0 : !moore.i1 -> !moore.i1
  %1 = moore.bool_cast %arg0 : !moore.i1 -> !moore.i1
  // CHECK: return %arg0, %arg0
  return %0, %1 : !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @LogicToInt
func.func @LogicToInt(%arg0: !moore.i42) -> (!moore.i42, !moore.i42, !moore.i42) {
  // CHECK-NOT: moore.int_to_logic
  // CHECK-NOT: moore.logic_to_int
  %0 = moore.int_to_logic %arg0 : i42
  %1 = moore.logic_to_int %0 : l42

  // CHECK-DAG: [[TMP2:%.+]] = moore.constant 9001 : i42
  %2 = moore.constant 9001 : l42
  %3 = moore.logic_to_int %2 : l42

  // CHECK-DAG: [[TMP3:%.+]] = moore.constant 36865 : i42
  %4 = moore.constant h9XZ1 : l42
  %5 = moore.logic_to_int %4 : l42

  // CHECK: return %arg0, [[TMP2]], [[TMP3]] :
  return %1, %3, %5 : !moore.i42, !moore.i42, !moore.i42
}

// CHECK-LABEL: func.func @IntToLogic
func.func @IntToLogic(%arg0: !moore.l42) -> (!moore.l42, !moore.l42) {
  // CHECK-DAG: [[TMP1:%.+]] = moore.logic_to_int %arg0
  // CHECK-DAG: [[TMP2:%.+]] = moore.int_to_logic [[TMP1]]
  %0 = moore.logic_to_int %arg0 : l42
  %1 = moore.int_to_logic %0 : i42

  // CHECK-DAG: [[TMP3:%.+]] = moore.constant 9001 : l42
  %2 = moore.constant 9001 : i42
  %3 = moore.int_to_logic %2 : i42

  // CHECK: return [[TMP2]], [[TMP3]] :
  return %1, %3 : !moore.l42, !moore.l42
}

// CHECK-LABEL: func.func @ConstTrunc
func.func @ConstTrunc() -> (!moore.i4, !moore.l4) {
  // CHECK-NEXT: moore.constant 5 : i4
  // CHECK-NEXT: moore.constant bZX01 : l4
  %0 = moore.constant b10100101 : i8
  %1 = moore.constant b10XZZX01 : l8
  %2 = moore.trunc %0 : i8 -> i4
  %3 = moore.trunc %1 : l8 -> l4
  return %2, %3 : !moore.i4, !moore.l4
}

// CHECK-LABEL: func.func @ConstZExt
func.func @ConstZExt() -> (!moore.i8, !moore.i8, !moore.l8, !moore.l8, !moore.l8, !moore.l8) {
  // CHECK-NEXT: moore.constant 5 : i8
  // CHECK-NEXT: moore.constant 13 : i8
  %i0 = moore.constant b0101 : i4
  %i1 = moore.constant b1101 : i4
  %ei0 = moore.zext %i0 : i4 -> i8
  %ei1 = moore.zext %i1 : i4 -> i8

  // CHECK-NEXT: moore.constant 5 : l8
  // CHECK-NEXT: moore.constant 13 : l8
  // CHECK-NEXT: moore.constant bX101 : l8
  // CHECK-NEXT: moore.constant bZ101 : l8
  %l0 = moore.constant b0101 : l4
  %l1 = moore.constant b1101 : l4
  %l2 = moore.constant bX101 : l4
  %l3 = moore.constant bZ101 : l4
  %el0 = moore.zext %l0 : l4 -> l8
  %el1 = moore.zext %l1 : l4 -> l8
  %el2 = moore.zext %l2 : l4 -> l8
  %el3 = moore.zext %l3 : l4 -> l8

  return %ei0, %ei1, %el0, %el1, %el2, %el3 : !moore.i8, !moore.i8, !moore.l8, !moore.l8, !moore.l8, !moore.l8
}

// CHECK-LABEL: func.func @ConstSExt
func.func @ConstSExt() -> (!moore.i8, !moore.i8, !moore.l8, !moore.l8, !moore.l8, !moore.l8) {
  // CHECK-NEXT: moore.constant 5 : i8
  // CHECK-NEXT: moore.constant -3 : i8
  %i0 = moore.constant b0101 : i4
  %i1 = moore.constant b1101 : i4
  %ei0 = moore.sext %i0 : i4 -> i8
  %ei1 = moore.sext %i1 : i4 -> i8

  // CHECK-NEXT: moore.constant 5 : l8
  // CHECK-NEXT: moore.constant -3 : l8
  // CHECK-NEXT: moore.constant bXXXXX101 : l8
  // CHECK-NEXT: moore.constant bZZZZZ101 : l8
  %l0 = moore.constant b0101 : l4
  %l1 = moore.constant b1101 : l4
  %l2 = moore.constant bX101 : l4
  %l3 = moore.constant bZ101 : l4
  %el0 = moore.sext %l0 : l4 -> l8
  %el1 = moore.sext %l1 : l4 -> l8
  %el2 = moore.sext %l2 : l4 -> l8
  %el3 = moore.sext %l3 : l4 -> l8

  return %ei0, %ei1, %el0, %el1, %el2, %el3 : !moore.i8, !moore.i8, !moore.l8, !moore.l8, !moore.l8, !moore.l8
}

// CHECK-LABEL: func.func @ConstTimeToLogic
func.func @ConstTimeToLogic() -> !moore.l64 {
  // CHECK-NEXT: moore.constant 1234 : l64
  %0 = moore.constant_time 1234 fs
  %1 = moore.time_to_logic %0
  return %1 : !moore.l64
}

// CHECK-LABEL: func.func @ConstLogicToTime
func.func @ConstLogicToTime() -> !moore.time {
  // CHECK-NEXT: moore.constant_time 1234 fs
  %0 = moore.constant 1234 : l64
  %1 = moore.logic_to_time %0
  return %1 : !moore.time
}

// CHECK-LABEL: func.func @RedundantTimeLogicConversions
func.func @RedundantTimeLogicConversions(%arg0: !moore.time, %arg1: !moore.l64) -> (!moore.time, !moore.l64) {
  // CHECK-NEXT: return %arg0, %arg1
  %0 = moore.time_to_logic %arg0
  %1 = moore.logic_to_time %0
  %2 = moore.logic_to_time %arg1
  %3 = moore.time_to_logic %2
  return %1, %3 : !moore.time, !moore.l64
}

// CHECK-LABEL: func.func @ConstMul
func.func @ConstMul() -> (!moore.i24, !moore.l24, !moore.l24) {
  %c2_i24 = moore.constant 2 : i24
  %c2_l24 = moore.constant 2 : l24
  %c3_i24 = moore.constant 3 : i24
  %c3_l24 = moore.constant 3 : l24
  %cXZ_l24 = moore.constant bXZ : l24

  // CHECK-DAG: [[V0:%.+]] = moore.constant 6 : i24
  %0 = moore.mul %c2_i24, %c3_i24 : i24
  // CHECK-DAG: [[V1:%.+]] = moore.constant 6 : l24
  %1 = moore.mul %c2_l24, %c3_l24 : l24
  // CHECK-DAG: [[V2:%.+]] = moore.constant hXXXXXX : l24
  %2 = moore.mul %c2_l24, %cXZ_l24 : l24

  // CHECK: return [[V0]], [[V1]], [[V2]]
  return %0, %1, %2 : !moore.i24, !moore.l24, !moore.l24
}

// CHECK-LABEL: func.func @ConstDivU
func.func @ConstDivU() -> (!moore.i24, !moore.l24, !moore.l24, !moore.l24) {
  %c6_i24 = moore.constant 6 : i24
  %c6_l24 = moore.constant 6 : l24
  %c3_i24 = moore.constant 3 : i24
  %c3_l24 = moore.constant 3 : l24
  %cXZ_l24 = moore.constant bXZ : l24
  %c0_l24 = moore.constant 0 : l24

  // CHECK-DAG: [[I2:%.+]] = moore.constant 2 : i24
  // CHECK-DAG: [[L2:%.+]] = moore.constant 2 : l24
  // CHECK-DAG: [[X:%.+]] = moore.constant hXXXXXX : l24
  %0 = moore.divu %c6_i24, %c3_i24 : i24
  %1 = moore.divu %c6_l24, %c3_l24 : l24
  %2 = moore.divu %c6_l24, %cXZ_l24 : l24
  %3 = moore.divu %c6_l24, %c0_l24 : l24

  // CHECK: return [[I2]], [[L2]], [[X]], [[X]]
  return %0, %1, %2, %3 : !moore.i24, !moore.l24, !moore.l24, !moore.l24
}

// CHECK-LABEL: func.func @ConstDivS
func.func @ConstDivS() -> (!moore.i24, !moore.l24, !moore.i24, !moore.l24, !moore.i24, !moore.l24, !moore.i24, !moore.l24, !moore.l24, !moore.l24) {
  %c6_i24 = moore.constant 6 : i24
  %c6_l24 = moore.constant 6 : l24
  %c3_i24 = moore.constant 3 : i24
  %c3_l24 = moore.constant 3 : l24
  %c-6_i24 = moore.constant -6 : i24
  %c-6_l24 = moore.constant -6 : l24
  %c-3_i24 = moore.constant -3 : i24
  %c-3_l24 = moore.constant -3 : l24
  %cXZ_l24 = moore.constant bXZ : l24
  %c0_l24 = moore.constant 0 : l24

  // CHECK-DAG: [[I2:%.+]] = moore.constant 2 : i24
  // CHECK-DAG: [[L2:%.+]] = moore.constant 2 : l24
  // CHECK-DAG: [[IM2:%.+]] = moore.constant -2 : i24
  // CHECK-DAG: [[LM2:%.+]] = moore.constant -2 : l24
  // CHECK-DAG: [[X:%.+]] = moore.constant hXXXXXX : l24
  %0 = moore.divs %c6_i24, %c3_i24 : i24
  %1 = moore.divs %c6_l24, %c3_l24 : l24
  %2 = moore.divs %c6_i24, %c-3_i24 : i24
  %3 = moore.divs %c6_l24, %c-3_l24 : l24
  %4 = moore.divs %c-6_i24, %c3_i24 : i24
  %5 = moore.divs %c-6_l24, %c3_l24 : l24
  %6 = moore.divs %c-6_i24, %c-3_i24 : i24
  %7 = moore.divs %c-6_l24, %c-3_l24 : l24
  %8 = moore.divs %c6_l24, %cXZ_l24 : l24
  %9 = moore.divs %c6_l24, %c0_l24 : l24

  // CHECK: return [[I2]], [[L2]], [[IM2]], [[LM2]], [[IM2]], [[LM2]], [[I2]], [[L2]], [[X]], [[X]]
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9: !moore.i24, !moore.l24, !moore.i24, !moore.l24, !moore.i24, !moore.l24, !moore.i24, !moore.l24, !moore.l24, !moore.l24
}

// CHECK-LABEL: moore.module @OptimizeUniquelyAssignedVars
moore.module @OptimizeUniquelyAssignedVars(in %u: !moore.i42, in %v: !moore.i42, in %w: !moore.i42) {
  // Unique continuous assignments to variables should remove the `ref<T>`
  // indirection and instead directly propagate the assigned value to readers.
  // CHECK-NOT: moore.assign
  // CHECK-NOT: moore.variable
  // CHECK: %a = moore.assigned_variable %u : i42
  // CHECK: dbg.variable "a", %a : !moore.i42
  moore.assign %a, %u : i42
  %a = moore.variable : <i42>
  %3 = moore.read %a : <i42>
  dbg.variable "a", %3 : !moore.i42

  // Continuous assignments to variables should override the initial value.
  // CHECK-NOT: moore.assign
  // CHECK-NOT: moore.constant 9001
  // CHECK-NOT: moore.variable
  // CHECK: %b = moore.assigned_variable %v : i42
  // CHECK: dbg.variable "b", %b : !moore.i42
  moore.assign %b, %v : i42
  %0 = moore.constant 9001 : i42
  %b = moore.variable %0 : <i42>
  %4 = moore.read %b : <i42>
  dbg.variable "b", %4 : !moore.i42

  // Unique continuous assignments to nets should remove the `ref<T>`
  // indirection and instead directly propagate the assigned value to readers.
  // CHECK-NOT: moore.assign
  // CHECK-NOT: moore.net wire
  // CHECK: %c = moore.assigned_variable %w : i42
  // CHECK: dbg.variable "c", %c : !moore.i42
  moore.assign %c, %w : i42
  %c = moore.net wire : <i42>
  %5 = moore.read %c : <i42>
  dbg.variable "c", %5 : !moore.i42

  // Variables without names should not create an `assigned_variable`.
  // CHECK-NOT: moore.assign
  // CHECK-NOT: moore.variable
  // CHECK-NOT: moore.assigned_variable
  // CHECK: dbg.variable "d", %u : !moore.i42
  moore.assign %1, %u : i42
  %1 = moore.variable : <i42>
  %6 = moore.read %1 : <i42>
  dbg.variable "d", %6 : !moore.i42

  // Nets without names should not create an `assigned_variable`.
  // CHECK-NOT: moore.assign
  // CHECK-NOT: moore.net wire
  // CHECK-NOT: moore.assigned_variable
  // CHECK: dbg.variable "e", %v : !moore.i42
  moore.assign %2, %v : i42
  %2 = moore.net wire : <i42>
  %7 = moore.read %2 : <i42>
  dbg.variable "e", %7 : !moore.i42
}

// CHECK-LABEL: moore.module @DontOptimizeVarsWithMultipleAssigns
moore.module @DontOptimizeVarsWithMultipleAssigns() {
  %0 = moore.constant 1337 : i42
  %1 = moore.constant 9001 : i42

  // CHECK: %a = moore.variable
  // CHECK: moore.assign %a
  // CHECK: moore.assign %a
  // CHECK: [[TMP:%.+]] = moore.read %a
  // CHECK: dbg.variable "a", [[TMP]]
  %a = moore.variable : <i42>
  moore.assign %a, %0 : i42
  moore.assign %a, %1 : i42
  %2 = moore.read %a : <i42>
  dbg.variable "a", %2 : !moore.i42

  // CHECK: %b = moore.net
  // CHECK: moore.assign %b
  // CHECK: moore.assign %b
  // CHECK: [[TMP:%.+]] = moore.read %b
  // CHECK: dbg.variable "b", [[TMP]]
  %b = moore.net wire : <i42>
  moore.assign %b, %0 : i42
  moore.assign %b, %1 : i42
  %3 = moore.read %b : <i42>
  dbg.variable "b", %3 : !moore.i42

  // CHECK: %c = moore.net
  // CHECK: moore.assign %c
  // CHECK: [[TMP:%.+]] = moore.read %c
  // CHECK: dbg.variable "c", [[TMP]]
  %c = moore.net wire %0 : <i42>
  moore.assign %c, %1 : i42
  %4 = moore.read %c : <i42>
  dbg.variable "c", %4 : !moore.i42
}

// CHECK-LABEL: moore.module @DontOptimizeVarsWithNonReadUses
moore.module @DontOptimizeVarsWithNonReadUses(in %u: !moore.i42, in %v: !moore.i42) {
  // CHECK: %a = moore.variable
  // CHECK: moore.assign %a, %u
  // CHECK: func.call @useRef(%a)
  // CHECK: [[TMP:%.+]] = moore.read %a
  // CHECK: dbg.variable "a", [[TMP]]
  %a = moore.variable : <i42>
  moore.assign %a, %u : i42
  func.call @useRef(%a) : (!moore.ref<i42>) -> ()
  %2 = moore.read %a : <i42>
  dbg.variable "a", %2 : !moore.i42

  // CHECK: %b = moore.net wire %u
  // CHECK: moore.assign %b, %v
  // CHECK: func.call @useRef(%b)
  // CHECK: [[TMP:%.+]] = moore.read %b
  // CHECK: dbg.variable "b", [[TMP]]
  %b = moore.net wire %u : <i42>
  moore.assign %b, %v : i42
  func.call @useRef(%b) : (!moore.ref<i42>) -> ()
  %3 = moore.read %b : <i42>
  dbg.variable "b", %3 : !moore.i42

  // Unique continuous assigns should be folded into net definitions even if the
  // net has non-read uses.
  // CHECK: %c = moore.net wire %u
  // CHECK-NOT: moore.assign %c
  // CHECK: func.call @useRef(%c)
  %c = moore.net wire : <i42>
  moore.assign %c, %u : i42
  func.call @useRef(%c) : (!moore.ref<i42>) -> ()
}

func.func private @useRef(%arg0: !moore.ref<i42>)

// CHECK-LABEL: moore.module @DropRedundantVars
moore.module @DropRedundantVars(in %a : !moore.i42, out b : !moore.i42, out c : !moore.i42) {
  // CHECK: [[C9001:%.+]] = moore.constant 9001 : i42
  %c9001_i42 = moore.constant 9001 : i42

  // Remove variables that shadow an input port of the same name.
  // CHECK-NOT: moore.assigned_variable
  // CHECK: dbg.variable "a", %a
  %0 = moore.assigned_variable name "a" %a : i42
  dbg.variable "a", %0 : !moore.i42

  // Variables that shadow an input port of a different name should remain.
  // CHECK: %a2 = moore.assigned_variable
  // CHECK: dbg.variable "a2", %a
  %a2 = moore.assigned_variable %a : i42
  dbg.variable "a2", %a2 : !moore.i42

  // Chained variables with the same name should be reduced to just one.
  // CHECK: %v = moore.assigned_variable %a
  // CHECK-NOT: moore.assigned_variable
  // CHECK: dbg.variable "v", %v
  %1 = moore.assigned_variable name "v" %a : i42
  %2 = moore.assigned_variable name "v" %1 : i42
  dbg.variable "v", %2 : !moore.i42

  // Remove variables that shadow an output port of the same name. Variables
  // that shadow an output port of a different name should remain.
  // CHECK-NOT: %b = moore.assigned_variable
  // CHECK: %w = moore.assigned_variable [[C9001]]
  // CHECK: moore.output [[C9001]], %w
  %b = moore.assigned_variable %c9001_i42 : i42
  %w = moore.assigned_variable %c9001_i42 : i42
  moore.output %b, %w : !moore.i42, !moore.i42
}

// CHECK-LABEL: func.func @StructExtractFold1
func.func @StructExtractFold1(%arg0: !moore.struct<{a: i17, b: i42}>, %arg1: !moore.i17) -> (!moore.i17) {
  // CHECK-NEXT: return %arg1 : !moore.i17
  %0 = moore.struct_inject %arg0, "a", %arg1 : struct<{a: i17, b: i42}>, i17
  %1 = moore.struct_extract %0, "a" : struct<{a: i17, b: i42}> -> i17
  return %1 : !moore.i17
}

// CHECK-LABEL: func.func @StructExtractFold2
func.func @StructExtractFold2(%arg0: !moore.i17, %arg1: !moore.i42) -> (!moore.i17, !moore.i42) {
  // CHECK-NEXT: return %arg0, %arg1 : !moore.i17, !moore.i42
  %0 = moore.struct_create %arg0, %arg1 : !moore.i17, !moore.i42 -> struct<{a: i17, b: i42}>
  %1 = moore.struct_extract %0, "a" : struct<{a: i17, b: i42}> -> i17
  %2 = moore.struct_extract %0, "b" : struct<{a: i17, b: i42}> -> i42
  return %1, %2 : !moore.i17, !moore.i42
}

// CHECK-LABEL: func.func @StructInjectFold1
func.func @StructInjectFold1(%arg0: !moore.struct<{a: i32, b: i32}>) -> (!moore.struct<{a: i32, b: i32}>) {
  // CHECK-NEXT: [[C42:%.+]] = moore.constant 42
  // CHECK-NEXT: [[C43:%.+]] = moore.constant 43
  // CHECK-NEXT: [[TMP:%.+]] = moore.struct_create [[C42]], [[C43]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  // CHECK-NEXT: return [[TMP]]
  %0 = moore.constant 42 : i32
  %1 = moore.constant 43 : i32
  %2 = moore.struct_inject %arg0, "a", %1 : struct<{a: i32, b: i32}>, i32
  %3 = moore.struct_inject %2, "b", %1 : struct<{a: i32, b: i32}>, i32
  %4 = moore.struct_inject %3, "a", %0 : struct<{a: i32, b: i32}>, i32
  return %4 : !moore.struct<{a: i32, b: i32}>
}

// CHECK-LABEL: func.func @StructInjectFold2
func.func @StructInjectFold2() -> (!moore.struct<{a: i32, b: i32}>) {
  // CHECK-NEXT: [[C42:%.+]] = moore.constant 42
  // CHECK-NEXT: [[C43:%.+]] = moore.constant 43
  // CHECK-NEXT: [[TMP:%.+]] = moore.struct_create [[C42]], [[C43]] : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  // CHECK-NEXT: return [[TMP]]
  %0 = moore.constant 42 : i32
  %1 = moore.constant 43 : i32
  %2 = moore.struct_create %0, %0 : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>
  %3 = moore.struct_inject %2, "b", %1 : struct<{a: i32, b: i32}>, i32
  return %3 : !moore.struct<{a: i32, b: i32}>
}

// CHECK-LABEL: func.func @StructInjectFold3
func.func @StructInjectFold3(%arg0: !moore.struct<{a: i32, b: i32}>) -> (!moore.struct<{a: i32, b: i32}>) {
  // CHECK-NEXT: [[C43:%.+]] = moore.constant 43
  // CHECK-NEXT: [[TMP:%.+]] = moore.struct_inject %arg0, "a", [[C43]] : struct<{a: i32, b: i32}>, i32
  // CHECK-NEXT: return [[TMP]]
  %0 = moore.constant 42 : i32
  %1 = moore.constant 43 : i32
  %2 = moore.struct_inject %arg0, "a", %0 : struct<{a: i32, b: i32}>, i32
  %3 = moore.struct_inject %2, "a", %1 : struct<{a: i32, b: i32}>, i32
  return %3 : !moore.struct<{a: i32, b: i32}>
}

// CHECK-LABEL: func.func @ConvertConstantTwoToFourValued
func.func @ConvertConstantTwoToFourValued() -> (!moore.l42) {
  // CHECK: [[TMP:%.+]] = moore.constant 9001 : l42
  // CHECK-NOT: moore.conversion
  // CHECK: return [[TMP]] :
  %0 = moore.constant 9001 : i42
  %1 = moore.conversion %0 : !moore.i42 -> !moore.l42
  return %1 : !moore.l42
}

// CHECK-LABEL: func.func @ConvertConstantFourToTwoValued
func.func @ConvertConstantFourToTwoValued() -> (!moore.i42) {
  // CHECK: [[TMP:%.+]] = moore.constant 8 : i42
  // CHECK-NOT: moore.conversion
  // CHECK: return [[TMP]] :
  %0 = moore.constant b1XZ0 : l42
  %1 = moore.conversion %0 : !moore.l42 -> !moore.i42
  return %1 : !moore.i42
}

// CHECK-LABEL: func @Pow
func.func @Pow(%arg0 : !moore.l32) -> (!moore.l32, !moore.l32, !moore.l32, !moore.l32, !moore.l32, !moore.l32) {
  // CHECK-NEXT: [[V0:%.+]] = moore.constant 1 : l32
  // CHECK-NEXT: [[V1:%.+]] = moore.constant 0 : l32
  %0 = moore.constant 0 : l32
  %1 = moore.constant 1 : l32
  %2 = moore.constant 2 : l32

  %3 = moore.pows %1, %arg0 : l32
  %4 = moore.pows %arg0, %0 : l32

  %5 = moore.powu %1, %arg0 : l32
  %6 = moore.powu %arg0, %0 : l32

  // CHECK-NEXT: [[V2:%.+]] = moore.shl [[V0]], %arg0
  // CHECK-NEXT: [[V3:%.+]] = moore.slt %arg0, [[V1]]
  // CHECK-NEXT: [[V4:%.+]] = moore.conditional [[V3]]
  // CHECK-NEXT:   moore.yield [[V1]]
  // CHECK-NEXT: } {
  // CHECK-NEXT:   moore.yield [[V2]]
  // CHECK-NEXT: }
  %7 = moore.pows %2, %arg0 : l32

  // CHECK-NEXT: [[V5:%.+]] = moore.shl [[V0]], %arg0
  %8 = moore.powu %2, %arg0 : l32

  // CHECK-NEXT: return [[V0]], [[V0]], [[V0]], [[V0]], [[V4]], [[V5]] :
  return %3, %4, %5, %6, %7, %8 : !moore.l32, !moore.l32, !moore.l32, !moore.l32, !moore.l32, !moore.l32
}

// CHECK-LABEL: func.func @MoveInitialOutOfSSAVariable
func.func @MoveInitialOutOfSSAVariable() {
  // CHECK: [[TMP:%.+]] = moore.constant 9001
  %0 = moore.constant 9001 : i42
  // CHECK: [[VAR:%.+]] = moore.variable : <i42>
  // CHECK-NEXT: moore.blocking_assign [[VAR]], [[TMP]]
  %1 = moore.variable %0 : <i42>
  func.call @useRef(%1) : (!moore.ref<i42>) -> ()
  return
}

// CHECK-LABEL: @sub
func.func @sub(%arg0: !moore.i32) -> !moore.i32 {
  %0 = moore.constant 0 : !moore.i32
  %1 = moore.sub %arg0, %0 : !moore.i32
  // CHECK: return %arg0 :
  return %1 : !moore.i32
}

// CHECK-LABEL: @convertReal
func.func @convertReal(%arg0: !moore.f32) -> !moore.f32 {
  // CHECK-NOT: moore.convert_real
  %0 = moore.convert_real %arg0 : !moore.f32 -> !moore.f32
  return %0 : !moore.f32
}
