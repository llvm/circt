// RUN: circt-opt %s --arc-canonicalizer | FileCheck %s

//===----------------------------------------------------------------------===//
// Remove Passthrough calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @passthoughChecks
hw.module @passthoughChecks(%in0: i1, %in1: i1) -> (out0: i1, out1: i1, out2: i1, out3: i1, out4: i1, out5: i1, out6: i1, out7: i1, out8: i1, out9: i1) {
  %0:2 = arc.call @passthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %1:2 = arc.call @noPassthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %2:2 = arc.state @passthrough(%in0, %in1) lat 0 : (i1, i1) -> (i1, i1)
  %3:2 = arc.state @noPassthrough(%in0, %in1) lat 0 : (i1, i1) -> (i1, i1)
  %4:2 = arc.state @passthrough(%in0, %in1) clock %in0 lat 1 : (i1, i1) -> (i1, i1)
  hw.output %0#0, %0#1, %1#0, %1#1, %2#0, %2#1, %3#0, %3#1, %4#0, %4#1 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  // CHECK-NEXT: [[V0:%.+]]:2 = arc.call @noPassthrough(%in0, %in1) :
  // CHECK-NEXT: [[V1:%.+]]:2 = arc.state @noPassthrough(%in0, %in1) lat 0 :
  // CHECK-NEXT: [[V2:%.+]]:2 = arc.state @passthrough(%in0, %in1) clock %in0 lat 1 :
  // CHECK-NEXT: hw.output %in0, %in1, [[V0]]#0, [[V0]]#1, %in0, %in1, [[V1]]#0, [[V1]]#1, [[V2]]#0, [[V2]]#1 :
}
arc.define @passthrough(%arg0: i1, %arg1: i1) -> (i1, i1) {
  arc.output %arg0, %arg1 : i1, i1
}
arc.define @noPassthrough(%arg0: i1, %arg1: i1) -> (i1, i1) {
  arc.output %arg1, %arg0 : i1, i1
}

//===----------------------------------------------------------------------===//
// MemoryWritePortOp canonicalizer
//===----------------------------------------------------------------------===//

arc.define @memArcFalse(%arg0: i1, %arg1: i32) -> (i1, i32, i1) {
  %false = hw.constant false
  arc.output %arg0, %arg1, %false : i1, i32, i1
}
arc.define @memArcTrue(%arg0: i1, %arg1: i32) -> (i1, i32, i1) {
  %true = hw.constant true
  arc.output %arg0, %arg1, %true : i1, i32, i1
}

// CHECK-LABEL: hw.module @memoryWritePortCanonicalizations
hw.module @memoryWritePortCanonicalizations(%clk: i1, %addr: i1, %data: i32) {
  // CHECK-NEXT: [[MEM:%.+]] = arc.memory <2 x i32, i1>
  %mem = arc.memory <2 x i32, i1>
  arc.memory_write_port %mem, @memArcFalse(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk lat 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk lat 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable lat 1 : <2 x i32, i1>, i1, i32
  %0:3 = arc.state @memArcTrue(%addr, %data) lat 0 : (i1, i32) -> (i1, i32, i1)
  // CHECK-NEXT: hw.output
  hw.output
}

//===----------------------------------------------------------------------===//
// RemoveUnusedArcs
//===----------------------------------------------------------------------===//

// CHECK-NOT: arc.define @unusedArcIsDeleted
arc.define @unusedArcIsDeleted(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arc.call @nestedUnused(%arg0, %arg1) : (i32, i32) -> i32
  arc.output %0 : i32
}
// CHECK-NOT: arc.define @nestedUnused
arc.define @nestedUnused(%arg0: i32, %arg1: i32) -> i32 {
  %0 = comb.add %arg0, %arg1 : i32
  arc.output %0 : i32
}

//===----------------------------------------------------------------------===//
// ICMPCanonicalizer
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @icmpEqCanonicalizer
hw.module @icmpEqCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1, out2: i1, out3: i1) {
  // CHECK: [[V0:%.+]] = comb.and bin %arg0, %arg1, %arg2, %arg3 : i1
  %c-1_i4 = hw.constant -1 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin eq %0, %c-1_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.or bin %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor bin [[V1]], %true : i1
  %c0_i4 = hw.constant 0 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp bin eq %2, %c0_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.and bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp bin eq [[V3]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp bin eq %4, %c-1_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.or bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp bin eq [[V5]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp bin eq %6, %c0_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @icmpNeCanonicalizer
hw.module @icmpNeCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1, out2: i1, out3: i1) {
  // CHECK: [[V0:%.+]] = comb.or bin %arg0, %arg1, %arg2, %arg3 : i1
  %c0_i4 = hw.constant 0 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin ne %0, %c0_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.and bin %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor bin [[V1]], %true : i1
  %c-1_i4 = hw.constant -1 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp bin ne %2, %c-1_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.or bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp bin ne [[V3]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp bin ne %4, %c0_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.and bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp bin ne [[V5]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp bin ne %6, %c-1_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// RemoveUnusedArcArguments
//===----------------------------------------------------------------------===//

// COM: this has to be before @OneOfThreeUsed to check that arguments that
// COM: become unused during the process are removed as well.
// CHECK: arc.define @NestedCall(%arg0: i1) -> i1 {
arc.define @NestedCall(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  // CHECK: arc.call @OneOfThreeUsed(%arg0) : (i1) -> i1
  %0 = arc.call @OneOfThreeUsed(%arg0, %arg1, %arg2) : (i1, i1, i1) -> i1
  arc.output %0 : i1
}

// CHECK-LABEL: arc.define @OneOfThreeUsed(%arg0: i1)
arc.define @OneOfThreeUsed(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  %true = hw.constant true
  %0 = comb.xor %arg1, %true : i1
  // CHECK: arc.output {{%[0-9]+}} :
  arc.output %0 : i1
}

// CHECK: @test1
hw.module @test1 (%arg0: i1, %arg1: i1, %arg2: i1, %clock: i1) -> (out0: i1, out1: i1) {
  // CHECK-NEXT: arc.state @OneOfThreeUsed(%arg1) clock %clock lat 1 : (i1) -> i1
  %0 = arc.state @OneOfThreeUsed(%arg0, %arg1, %arg2) clock %clock lat 1 : (i1, i1, i1) -> i1
  // CHECK-NEXT: arc.state @NestedCall(%arg1)
  %1 = arc.state @NestedCall(%arg0, %arg1, %arg2) clock %clock lat 1 : (i1, i1, i1) -> i1
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: arc.define @NoArgsToRemove()
arc.define @NoArgsToRemove() -> i1 {
  %0 = hw.constant 0 : i1
  arc.output %0 : i1
}

// CHECK: @test2
hw.module @test2 () -> (out: i1) {
  // CHECK-NEXT: arc.state @NoArgsToRemove() lat 0 : () -> i1
  %0 = arc.state @NoArgsToRemove() lat 0 : () -> i1
  hw.output %0 : i1
}

//===----------------------------------------------------------------------===//
// SinkArcInputs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @SinkSameConstantsArc(%arg0: i4)
arc.define @SinkSameConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: %c2_i4 = hw.constant 2
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %c2_i4
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Foo
arc.define @Foo(%arg0: i4) -> i4 {
  // CHECK-NOT: hw.constant
  %k1 = hw.constant 2 : i4
  // CHECK: {{%.+}} = arc.call @SinkSameConstantsArc(%arg0)
  %0 = arc.call @SinkSameConstantsArc(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @SinkSameConstants
hw.module @SinkSameConstants(%x: i4) -> (out0: i4, out1: i4, out2: i4) {
  // CHECK-NOT: hw.constant
  // CHECK-NEXT: %0 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: %1 = arc.state @SinkSameConstantsArc(%x)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @SinkSameConstantsArc(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @SinkSameConstantsArc(%x, %k2) lat 0 : (i4, i4) -> i4
  %2 = arc.call @Foo(%x) : (i4) -> i4
  hw.output %0, %1, %2 : i4, i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-LABEL: hw.module @DontSinkDifferentConstants
hw.module @DontSinkDifferentConstants(%x: i4) -> (out0: i4, out1: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4)
  // CHECK-NEXT: hw.output
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc(%x, %c2_i4) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc(%x, %c3_i4) lat 0 : (i4, i4) -> i4
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4)
arc.define @DontSinkDifferentConstantsArc1(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: [[TMP:%.+]] = comb.add %arg0, %arg1
  // CHECK-NEXT: arc.output [[TMP]]
  %0 = comb.add %arg0, %arg1 : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK: arc.define @Bar
arc.define @Bar(%arg0: i4) -> i4 {
  // CHECK: %c1_i4 = hw.constant 1
  %k1 = hw.constant 1 : i4
  // CHECK: {{%.+}} = arc.call @DontSinkDifferentConstantsArc1(%arg0, %c1_i4)
  %0 = arc.call @DontSinkDifferentConstantsArc1(%arg0, %k1) : (i4, i4) -> i4
  arc.output %0 : i4
}

// CHECK: hw.module @DontSinkDifferentConstants1
hw.module @DontSinkDifferentConstants1(%x: i4) -> (out0: i4, out1: i4, out2: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.state @DontSinkDifferentConstantsArc1(%x, %k1) lat 0 : (i4, i4) -> i4
  %1 = arc.state @DontSinkDifferentConstantsArc1(%x, %k2) lat 0 : (i4, i4) -> i4
  %2 = arc.call @Bar(%x) : (i4) -> i4
  hw.output %0, %1, %2 : i4, i4, i4
}
// CHECK-NEXT: }

// arc.define @TLB_1_arc_191_split_1(%arg0: i2, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i1, %arg10: i1, %arg11: i1, %arg12: i1, %arg13: i1, %arg14: i1, %arg15: i1, %arg16: i1, %arg17: i1, %arg18: i1, %arg19: i1, %arg20: i1, %arg21: i1, %arg22: i1, %arg23: i1, %arg24: i1, %arg25: i1, %arg26: i1, %arg27: i1, %arg28: i1, %arg29: i1, %arg30: i1, %arg31: i1, %arg32: i1, %arg33: i1, %arg34: i1, %arg35: i1, %arg36: i1, %arg37: i1, %arg38: i1, %arg39: i1, %arg40: i1, %arg41: i1, %arg42: i1, %arg43: i1, %arg44: i1, %arg45: i1, %arg46: i1, %arg47: i1, %arg48: i1, %arg49: i1, %arg50: i1, %arg51: i1, %arg52: i1, %arg53: i1, %arg54: i1, %arg55: i1, %arg56: i1, %arg57: i1, %arg58: i1, %arg59: i1, %arg60: i1, %arg61: i1, %arg62: i1, %arg63: i1, %arg64: i1, %arg65: i1, %arg66: i1, %arg67: i1, %arg68: i1, %arg69: i1, %arg70: i1, %arg71: i1, %arg72: i1, %arg73: i1, %arg74: i1, %arg75: i1, %arg76: i1, %arg77: i1, %arg78: i1, %arg79: i1) -> i1 {
//   %true = hw.constant true
//   %c-1_i13 = hw.constant -1 : i13
//   %c0_i13 = hw.constant 0 : i13
//   %0 = comb.extract %arg0 from 0 : (i2) -> i1
//   %1 = comb.replicate %0 : (i1) -> i13
//   %2 = comb.concat %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
//   %3 = comb.xor %1, %2 : i13
//   %4 = comb.concat %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
//   %5 = comb.and bin %3, %4 : i13
//   %6 = comb.xor %5, %c-1_i13 : i13
//   %7 = comb.xor %arg27, %true : i1
//   %8 = comb.xor %arg28, %true : i1
//   %9 = comb.xor %arg29, %true : i1
//   %10 = comb.xor %arg30, %true : i1
//   %11 = comb.xor %arg31, %true : i1
//   %12 = comb.xor %arg32, %true : i1
//   %13 = comb.xor %arg33, %true : i1
//   %14 = comb.xor %arg34, %true : i1
//   %15 = comb.xor %arg35, %true : i1
//   %16 = comb.xor %arg36, %true : i1
//   %17 = comb.xor %arg37, %true : i1
//   %18 = comb.xor %arg38, %true : i1
//   %19 = comb.xor %arg39, %true : i1
//   %20 = comb.concat %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
//   %21 = comb.and %6, %20 : i13
//   %22 = comb.concat %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
//   %23 = comb.or %21, %22 : i13
//   %24 = comb.xor %arg53, %true : i1
//   %25 = comb.xor %arg54, %true : i1
//   %26 = comb.xor %arg55, %true : i1
//   %27 = comb.xor %arg56, %true : i1
//   %28 = comb.xor %arg57, %true : i1
//   %29 = comb.xor %arg58, %true : i1
//   %30 = comb.xor %arg59, %true : i1
//   %31 = comb.xor %arg60, %true : i1
//   %32 = comb.xor %arg61, %true : i1
//   %33 = comb.xor %arg62, %true : i1
//   %34 = comb.xor %arg63, %true : i1
//   %35 = comb.xor %arg64, %true : i1
//   %36 = comb.xor %arg65, %true : i1
//   %37 = comb.concat %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
//   %38 = comb.concat %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
//   %39 = comb.and %23, %37, %38 : i13
//   %40 = comb.icmp bin ne %39, %c0_i13 : i13
//   %41 = comb.or bin %arg79, %40 : i1
//   arc.output %41 : i1
// }

// CHECK-LABEL: func.func @concatPullUp
func.func @concatPullUp(%arg0: i2, %arg1: i2, %arg2: i2, %arg3: i2, %arg4: i2, %arg5: i2) -> (i8, i8, i8) {
  %c-1_i2 = hw.constant -1 : i2
  %c1_i2 = hw.constant 1 : i2
  // CHECK-NEXT: %c-1_i8 = hw.constant -1 : i8
  // CHECK-NEXT: [[V0:%.+]] = comb.concat %arg0, %arg1, %arg2, %arg3 : i2, i2, i2, i2
  // CHECK-NEXT: [[V1:%.+]] = comb.xor [[V0]], %c-1_i8 : i8
  %0 = comb.xor %arg0, %c-1_i2 : i2
  %1 = comb.xor %arg1, %c-1_i2 : i2
  %2 = comb.xor %arg2, %c-1_i2 : i2
  %3 = comb.xor %arg3, %c-1_i2 : i2
  %4 = comb.concat %0, %1, %2, %3 : i2, i2, i2, i2

  %5 = comb.xor %arg4, %c-1_i2 : i2
  %6 = comb.xor %arg5, %c-1_i2 : i2
  %7 = comb.concat %arg0, %5, %6, %arg3 : i2, i2, i2, i2

  %8 = comb.and %arg0, %c1_i2 : i2
  %9 = comb.and %arg1, %c1_i2 : i2
  %10 = comb.and %arg2, %c-1_i2 : i2
  %11 = comb.and %arg3, %c-1_i2 : i2
  %12 = comb.concat %8, %9, %10, %11 : i2, i2, i2, i2

  // CHECK-NEXT: return [[V1]] : i8
  return %4, %7, %12 : i8, i8, i8
}
