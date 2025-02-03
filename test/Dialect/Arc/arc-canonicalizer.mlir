// RUN: circt-opt %s --arc-canonicalizer | FileCheck %s

//===----------------------------------------------------------------------===//
// Remove Passthrough calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @passthoughChecks
hw.module @passthoughChecks(in %clock: !seq.clock, in %in0: i1, in %in1: i1, out out0: i1, out out1: i1, out out2: i1, out out3: i1, out out4: i1, out out5: i1) {
  %0:2 = arc.call @passthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %1:2 = arc.call @noPassthrough(%in0, %in1) : (i1, i1) -> (i1, i1)
  %2:2 = arc.state @passthrough(%in0, %in1) clock %clock latency 1 : (i1, i1) -> (i1, i1)
  hw.output %0#0, %0#1, %1#0, %1#1, %2#0, %2#1 : i1, i1, i1, i1, i1, i1
  // CHECK-NEXT: [[V0:%.+]]:2 = arc.call @noPassthrough(%in0, %in1) :
  // CHECK-NEXT: [[V2:%.+]]:2 = arc.state @passthrough(%in0, %in1) clock %clock latency 1 :
  // CHECK-NEXT: hw.output %in0, %in1, [[V0]]#0, [[V0]]#1, [[V2]]#0, [[V2]]#1 :
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
hw.module @memoryWritePortCanonicalizations(in %clk: !seq.clock, in %addr: i1, in %data: i32) {
  // CHECK-NEXT: [[MEM:%.+]] = arc.memory <2 x i32, i1>
  %mem = arc.memory <2 x i32, i1>
  arc.memory_write_port %mem, @memArcFalse(%addr, %data) clock %clk enable latency 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk latency 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable latency 1 : <2 x i32, i1>, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @memArcTrue_0(%addr, %data) clock %clk latency 1 :
  arc.memory_write_port %mem, @memArcTrue(%addr, %data) clock %clk enable latency 1 : <2 x i32, i1>, i1, i32
  // COM: trivially dead operation, requires listener callback to keep symbol cache up-to-date
  %0:3 = arc.call @memArcTrue(%addr, %data) : (i1, i32) -> (i1, i32, i1)
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
hw.module @icmpEqCanonicalizer(in %arg0: i1, in %arg1: i1, in %arg2: i1, in %arg3: i1, in %arg4: i4, in %arg5: i4, in %arg6: i4, in %arg7: i4, out out0: i1, out out1: i1, out out2: i1, out out3: i1) {
  // CHECK: [[V0:%.+]] = comb.and %arg0, %arg1, %arg2, %arg3 : i1
  %c-1_i4 = hw.constant -1 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp eq %0, %c-1_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.or %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor [[V1]], %true : i1
  %c0_i4 = hw.constant 0 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp eq %2, %c0_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.and %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp eq [[V3]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp eq %4, %c-1_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.or %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp eq [[V5]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp eq %6, %c0_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @icmpNeCanonicalizer
hw.module @icmpNeCanonicalizer(in %arg0: i1, in %arg1: i1, in %arg2: i1, in %arg3: i1, in %arg4: i4, in %arg5: i4, in %arg6: i4, in %arg7: i4, out out0: i1, out out1: i1, out out2: i1, out out3: i1) {
  // CHECK: [[V0:%.+]] = comb.or %arg0, %arg1, %arg2, %arg3 : i1
  %c0_i4 = hw.constant 0 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp ne %0, %c0_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.and %arg0, %arg1, %arg2, %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.xor [[V1]], %true : i1
  %c-1_i4 = hw.constant -1 : i4
  %2 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %3 = comb.icmp ne %2, %c-1_i4 : i4

  // CHECK-NEXT: [[V3:%.+]] = comb.or %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp ne [[V3]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %4 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %5 = comb.icmp ne %4, %c0_i16 : i16

  // CHECK-NEXT: [[V5:%.+]] = comb.and %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.icmp ne [[V5]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %6 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %7 = comb.icmp ne %6, %c-1_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]], [[V4]], [[V6]] :
  hw.output %1, %3, %5, %7 : i1, i1, i1, i1
}

//===----------------------------------------------------------------------===//
// CompRegCanonicalizer
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @HoistCompRegReset
// CHECK-SAME: in %[[CLOCK:[^ ]*]] : !seq.clock, in %[[INPUT:[^ ]*]] : i32, in %[[RESET:[^ ]*]] : i1, in %[[RESET_VALUE:[^ ]*]] : i32
hw.module @HoistCompRegReset(in %clock: !seq.clock, in %input: i32, in %reset: i1, in %resetValue: i32, out out: i32) {
  // CHECK: %[[NEW_INPUT:.*]] = comb.mux %[[RESET]], %[[RESET_VALUE]], %[[INPUT]] : i32
  // CHECK: %[[REG:.*]] = seq.compreg %[[NEW_INPUT]], %[[CLOCK]] : i32
  %reg = seq.compreg %input, %clock reset %reset, %resetValue : i32

  // CHECK: hw.output %[[REG]] : i32
  hw.output %reg : i32
}

// CHECK-LABEL: hw.module @NoHoistCompRegZeroResetValue
// CHECK-SAME: in %[[CLOCK:[^ ]*]] : !seq.clock, in %[[INPUT:[^ ]*]] : i32, in %[[RESET:[^ ]*]] : i1
hw.module @NoHoistCompRegZeroResetValue(in %clock: !seq.clock, in %input: i32, in %reset: i1, out out: i32) {
  // CHECK: %[[ZERO:.*]] = hw.constant 0 : i32
  %zero = hw.constant 0 : i32

  // CHECK: %[[REG:.*]] = seq.compreg %[[INPUT]], %[[CLOCK]] reset %[[RESET]], %[[ZERO]] : i32
  %reg = seq.compreg %input, %clock reset %reset, %zero : i32

  // CHECK: hw.output %[[REG]] : i32
  hw.output %reg : i32
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
hw.module @test1 (in %arg0: i1, in %arg1: i1, in %arg2: i1, in %clock: !seq.clock, out out0: i1, out out1: i1) {
  // CHECK-NEXT: arc.state @OneOfThreeUsed(%arg1) clock %clock latency 1 : (i1) -> i1
  %0 = arc.state @OneOfThreeUsed(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> i1
  // CHECK-NEXT: arc.state @NestedCall(%arg1)
  %1 = arc.state @NestedCall(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> i1
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: arc.define @NoArgsToRemove()
arc.define @NoArgsToRemove() -> i1 {
  %0 = hw.constant 0 : i1
  arc.output %0 : i1
}

// CHECK: @test2
hw.module @test2 (out out: i1) {
  // CHECK-NEXT: arc.call @NoArgsToRemove() : () -> i1
  %0 = arc.call @NoArgsToRemove() : () -> i1
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
hw.module @SinkSameConstants(in %x: i4, out out0: i4, out out1: i4, out out2: i4) {
  // CHECK-NOT: hw.constant
  // CHECK-NEXT: %0 = arc.call @SinkSameConstantsArc(%x)
  // CHECK-NEXT: %1 = arc.call @SinkSameConstantsArc(%x)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.call @SinkSameConstantsArc(%x, %k1) : (i4, i4) -> i4
  %1 = arc.call @SinkSameConstantsArc(%x, %k2) : (i4, i4) -> i4
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
hw.module @DontSinkDifferentConstants(in %x: i4, out out0: i4, out out1: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT: %0 = arc.call @DontSinkDifferentConstantsArc(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.call @DontSinkDifferentConstantsArc(%x, %c3_i4)
  // CHECK-NEXT: hw.output
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %0 = arc.call @DontSinkDifferentConstantsArc(%x, %c2_i4) : (i4, i4) -> i4
  %1 = arc.call @DontSinkDifferentConstantsArc(%x, %c3_i4) : (i4, i4) -> i4
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
hw.module @DontSinkDifferentConstants1(in %x: i4, out out0: i4, out out1: i4, out out2: i4) {
  // CHECK-NEXT: %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT: %0 = arc.call @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: %1 = arc.call @DontSinkDifferentConstantsArc1(%x, %c2_i4)
  // CHECK-NEXT: arc.call
  // CHECK-NEXT: hw.output
  %k1 = hw.constant 2 : i4
  %k2 = hw.constant 2 : i4
  %0 = arc.call @DontSinkDifferentConstantsArc1(%x, %k1) : (i4, i4) -> i4
  %1 = arc.call @DontSinkDifferentConstantsArc1(%x, %k2) : (i4, i4) -> i4
  %2 = arc.call @Bar(%x) : (i4) -> i4
  hw.output %0, %1, %2 : i4, i4, i4
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//
// MergeVectorizeOps
//===----------------------------------------------------------------------===//

hw.module @VecOpTest(in %b: i8, in %e: i8, in %h: i8, in %k: i8, in %c: i8, in %f: i8,
in %i: i8, in %l: i8, in %n: i8, in %p: i8, in %r: i8, in %t: i8, in %en: i1,
in %clock: !seq.clock, in %o: i8, in %v: i8, in %q: i8, in %s: i8) {
  %R:4 = arc.vectorize(%b, %e, %h, %k), (%c, %f, %i, %l) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.add %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %L:4 = arc.vectorize(%R#0, %R#1, %R#2, %R#3), (%n, %p, %r, %t): (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.and %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %C:4 = arc.vectorize(%L#0, %L#1, %L#2, %L#3), (%o, %v, %q, %s) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0 : i8, %arg1: i8):
      %1692 = arc.call @Just_A_Dummy_Func(%arg0, %arg1) : (i8, i8) -> i8
      arc.vectorize.return %1692 : i8
  }
  %4 = arc.state @FooMux(%en, %C#0, %4) clock %clock latency 1 : (i1, i8, i8) -> i8
}

arc.define @FooMux(%arg0: i1, %arg1: i8, %arg2: i8) -> i8 {
  %0 = comb.mux %arg0, %arg1, %arg2 : i8
  arc.output %0 : i8
}
arc.define @Just_A_Dummy_Func(%arg0: i8, %arg1: i8) -> i8 {
  %0 = comb.or %arg0, %arg1: i8
  arc.output %0 : i8
}

// CHECK-LABEL: hw.module @VecOpTest(in %b : i8, in %e : i8, in %h : i8, in %k : i8, in %c : i8, in %f : i8, in %i : i8, in %l : i8, in %n : i8, in %p : i8, in %r : i8, in %t : i8, in %en : i1, in %clock : !seq.clock, in %o : i8, in %v : i8, in %q : i8, in %s : i8) {
// CHECK-NEXT:    [[VEC:%.+]]:4 = arc.vectorize (%b, %e, %h, %k), (%c, %f, %i, %l), (%n, %p, %r, %t), (%o, %v, %q, %s) : (i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
// CHECK-NEXT:    ^[[BLOCK:[[:alnum:]]+]](%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8):
// CHECK-NEXT:      [[ADD0:%.+]] = comb.add %arg0, %arg1 : i8
// CHECK-NEXT:      [[AND:%.+]] = comb.and [[ADD0]], %arg2 : i8
// CHECK-NEXT:      [[CALL:%.+]] = arc.call @Just_A_Dummy_Func([[AND]], %arg3) : (i8, i8) -> i8
// CHECK-NEXT:      arc.vectorize.return [[CALL]] : i8
// CHECK-NEXT:    }
// CHECK-NEXT:    [[STATE:%.+]] = arc.state @FooMux(%en, [[VEC]]#0, [[STATE]]) clock %clock latency 1 : (i1, i8, i8) -> i8
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }

hw.module @Test_2_in_1(in %b: i8, in %e: i8, in %h: i8, in %k: i8, in %c: i8, in %f: i8,
in %i: i8, in %l: i8, in %n: i8, in %p: i8, in %r: i8, in %t: i8, in %en: i1,
in %clock: !seq.clock, in %o: i8, in %v: i8, in %q: i8, in %s: i8) {
  %R:4 = arc.vectorize(%b, %e, %h, %k), (%c, %f, %i, %l) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.add %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %L:4 = arc.vectorize(%o, %v, %q, %s), (%n, %p, %r, %t): (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.and %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %C:4 = arc.vectorize(%R#0, %R#1, %R#2, %R#3), (%L#0, %L#1, %L#2, %L#3) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0 : i8, %arg1: i8):
      %1692 = arc.call @Just_A_Dummy_Func(%arg0, %arg1) : (i8, i8) -> i8
      arc.vectorize.return %1692 : i8
  }
  %4 = arc.state @FooMux(%en, %C#0, %4) clock %clock latency 1 : (i1, i8, i8) -> i8
}

// CHECK-LABEL: hw.module @Test_2_in_1(in %b : i8, in %e : i8, in %h : i8, in %k : i8, in %c : i8, in %f : i8, in %i : i8, in %l : i8, in %n : i8, in %p : i8, in %r : i8, in %t : i8, in %en : i1, in %clock : !seq.clock, in %o : i8, in %v : i8, in %q : i8, in %s : i8) {
// CHECK-NEXT:    [[VEC:%.+]]:4 = arc.vectorize (%b, %e, %h, %k), (%c, %f, %i, %l), (%o, %v, %q, %s), (%n, %p, %r, %t) : (i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
// CHECK-NEXT    ^[[BLOCK:[[:alnum:]]+]](%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8):
// CHECK-NEXT      [[AND:%.+]] = comb.and %arg2, %arg3 : i8
// CHECK-NEXT      [[ADD:%.+]] = comb.add %arg0, %arg1 : i8
// CHECK-NEXT      [[CALL:%.+]] = arc.call @Just_A_Dummy_Func([[ADD]], [[AND]]) : (i8, i8) -> i8
// CHECK-NEXT      arc.vectorize.return [[CALL]] : i8
// CHECK-NEXT    }
// CHECK-NEXT    [[STATE:%.+]] = arc.state @FooMux(%en, [[VEC]]#0, [[STATE:%.+]]) clock %clock latency 1 : (i1, i8, i8) -> i8
// CHECK-NEXT    hw.output
// CHECK-NEXT  }


hw.module @More_Than_One_Use(in %b: i8, in %e: i8, in %h: i8, in %k: i8, in %c: i8, in %f: i8,
in %i: i8, in %l: i8, in %n: i8, in %p: i8, in %r: i8, in %t: i8, in %en: i1,
in %clock: !seq.clock) {
  %R:4 = arc.vectorize(%b, %e, %h, %k), (%c, %f, %i, %l) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.add %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %L:4 = arc.vectorize(%R#0, %R#1, %R#2, %R#3), (%n, %p, %r, %t): (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.and %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %C:4 = arc.vectorize(%L#0, %L#1, %L#2, %L#3), (%R#0, %R#1, %R#2, %R#3) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0 : i8, %arg1: i8):
      %1692 = arc.call @Just_A_Dummy_Func(%arg0, %arg1) : (i8, i8) -> i8
      arc.vectorize.return %1692 : i8
  }
  %4 = arc.state @FooMux(%en, %C#0, %4) clock %clock latency 1 : (i1, i8, i8) -> i8
}

// CHECK-LABEL: hw.module @More_Than_One_Use(in %b : i8, in %e : i8, in %h : i8, in %k : i8, in %c : i8, in %f : i8, in %i : i8, in %l : i8, in %n : i8, in %p : i8, in %r : i8, in %t : i8, in %en : i1, in %clock : !seq.clock) {
// CHECK-NEXT:    [[VEC:%.+]]:4 = arc.vectorize (%b, %e, %h, %k), (%c, %f, %i, %l), (%n, %p, %r, %t) : (i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
// CHECK-NEXT:    ^[[BLOCK:[[:alnum:]]+]](%arg0: i8, %arg1: i8, %arg2: i8):
// CHECK-NEXT:      [[ADD:%.+]] = comb.add %arg0, %arg1 : i8
// CHECK-NEXT:      [[AND:%.+]] = comb.and [[ADD]], %arg2 : i8
// CHECK-NEXT:      [[RET:%.+]] = arc.call @Just_A_Dummy_Func([[AND]], [[ADD]]) : (i8, i8) -> i8
// CHECK-NEXT:      arc.vectorize.return [[RET]] : i8
// CHECK-NEXT:    }
// CHECK-NEXT:    [[STATE:%.+]] = arc.state @FooMux(%en, [[VEC]]#0, [[STATE]]) clock %clock latency 1 : (i1, i8, i8) -> i8
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }

arc.define @TLMonitor_14_arc(%arg0: i3) -> i3 {
  arc.output %arg0 : i3
}

hw.module private @Self_Use(in %clock : !seq.clock) {
  %0:2 = arc.vectorize (%clock, %clock), (%0#0, %0#1) : (!seq.clock, !seq.clock, i3, i3) -> (i3, i3) {
    ^bb0(%arg0: !seq.clock, %arg1: i3):
      %1 = arc.state @TLMonitor_14_arc(%arg1) clock %arg0 latency 1 : (i3) -> i3
      arc.vectorize.return %1 : i3
  }
  hw.output
}

// CHECK-LABEL: hw.module private @Self_Use(in %clock : !seq.clock) {
// CHECK-NEXT:    [[VEC:%.+]]:2 = arc.vectorize (%clock, %clock), ([[VEC:%.+]]#0, [[VEC:%.+]]#1) : (!seq.clock, !seq.clock, i3, i3) -> (i3, i3) {
// CHECK-NEXT:    ^[[BLOCK:[[:alnum:]]+]](%arg0: !seq.clock, %arg1: i3):
// CHECK-NEXT:      [[RET:%.+]] = arc.state @TLMonitor_14_arc(%arg1) clock %arg0 latency 1 : (i3) -> i3
// CHECK-NEXT:      arc.vectorize.return [[RET]] : i3
// CHECK-NEXT:    }
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }

hw.module @Needs_Shuffle(in %b: i8, in %e: i8, in %h: i8, in %k: i8, in %c: i8, in %f: i8,
in %i: i8, in %l: i8, in %n: i8, in %p: i8, in %r: i8, in %t: i8, in %en: i1,
in %clock: !seq.clock, in %o: i8, in %v: i8, in %q: i8, in %s: i8) {
  %R:4 = arc.vectorize(%b, %e, %h, %k), (%c, %f, %i, %l) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.add %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %L:4 = arc.vectorize(%R#1, %R#0, %R#2, %R#3), (%n, %p, %r, %t): (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.and %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %C:4 = arc.vectorize(%L#1, %L#0, %L#2, %L#3), (%o, %v, %q, %s) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0 : i8, %arg1: i8):
      %1692 = arc.call @Just_A_Dummy_Func(%arg0, %arg1) : (i8, i8) -> i8
      arc.vectorize.return %1692 : i8
  }
  %4 = arc.state @FooMux(%en, %C#0, %4) clock %clock latency 1 : (i1, i8, i8) -> i8
}

// CHECK-LABEL: hw.module @Needs_Shuffle(in %b : i8, in %e : i8, in %h : i8, in %k : i8, in %c : i8, in %f : i8, in %i : i8, in %l : i8, in %n : i8, in %p : i8, in %r : i8, in %t : i8, in %en : i1, in %clock : !seq.clock, in %o : i8, in %v : i8, in %q : i8, in %s : i8) {
// CHECK-NEXT:    [[VEC:%.+]]:4 = arc.vectorize (%b, %e, %h, %k), (%c, %f, %i, %l), (%p, %n, %r, %t), (%o, %v, %q, %s) : (i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
// CHECK-NEXT:    ^[[BLOCK:[[:alnum:]]+]](%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8):
// CHECK-NEXT:      [[ADD:%.+]] = comb.add %arg0, %arg1 : i8
// CHECK-NEXT:      [[AND:%.+]] = comb.and [[ADD]], %arg2 : i8
// CHECK-NEXT:      [[CALL:%.+]] = arc.call @Just_A_Dummy_Func([[AND]], %arg3) : (i8, i8) -> i8
// CHECK-NEXT:      arc.vectorize.return [[CALL]] : i8
// CHECK-NEXT:    }
// CHECK-NEXT:   [[STATE:%.+]] = arc.state @FooMux(%en, [[VEC]]#0, [[STATE]]) clock %clock latency 1 : (i1, i8, i8) -> i8
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }

hw.module @Needs_Shuffle_2(in %b: i8, in %e: i8, in %h: i8, in %k: i8, in %c: i8, in %f: i8,
in %i: i8, in %l: i8, in %n: i8, in %p: i8, in %r: i8, in %t: i8, in %en: i1,
in %clock: !seq.clock, in %o: i8, in %v: i8, in %q: i8, in %s: i8) {
  %R:4 = arc.vectorize(%b, %e, %h, %k), (%c, %f, %i, %l) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.add %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %L:4 = arc.vectorize(%R#3, %R#2, %R#1, %R#0), (%n, %p, %r, %t): (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.and %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %C:4 = arc.vectorize(%L#1, %L#0, %L#2, %L#3), (%o, %v, %q, %s) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0 : i8, %arg1: i8):
      %1692 = arc.call @Just_A_Dummy_Func(%arg0, %arg1) : (i8, i8) -> i8
      arc.vectorize.return %1692 : i8
  }
  %4 = arc.state @FooMux(%en, %C#0, %4) clock %clock latency 1 : (i1, i8, i8) -> i8
}

// CHECK-LABEL: hw.module @Needs_Shuffle_2(in %b : i8, in %e : i8, in %h : i8, in %k : i8, in %c : i8, in %f : i8, in %i : i8, in %l : i8, in %n : i8, in %p : i8, in %r : i8, in %t : i8, in %en : i1, in %clock : !seq.clock, in %o : i8, in %v : i8, in %q : i8, in %s : i8) {
// CHECK-NEXT:    [[VEC:%.+]]:4 = arc.vectorize (%h, %k, %e, %b), (%i, %l, %f, %c), (%p, %n, %r, %t), (%o, %v, %q, %s) : (i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
// CHECK-NEXT:    ^[[BLOCK:[[:alnum:]]+]](%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8):
// CHECK-NEXT:      [[ADD:%.+]] = comb.add %arg0, %arg1 : i8
// CHECK-NEXT:      [[AND:%.+]] = comb.and [[ADD]], %arg2 : i8
// CHECK-NEXT:      [[CALL:%.+]] = arc.call @Just_A_Dummy_Func([[AND]], %arg3) : (i8, i8) -> i8
// CHECK-NEXT:      arc.vectorize.return [[CALL]] : i8
// CHECK-NEXT:    }
// CHECK-NEXT:   [[STATE:%.+]] = arc.state @FooMux(%en, [[VEC]]#0, [[STATE]]) clock %clock latency 1 : (i1, i8, i8) -> i8
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }

hw.module @Repeated_input(in %b: i8, in %e: i8, in %h: i8, in %k: i8, in %en: i1, in %clock: !seq.clock) {
  %R:4 = arc.vectorize(%b, %e, %h, %k), (%b, %e, %h, %k) : (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.mul %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %4 = arc.state @FooMux(%en, %R, %4) clock %clock latency 1 : (i1, i8, i8) -> i8
}

// CHECK-LABEL: hw.module @Repeated_input(in %b : i8, in %e : i8, in %h : i8, in %k : i8, in %en : i1, in %clock : !seq.clock) {
// CHECK-NEXT:    [[VEC:%.+]]:4 = arc.vectorize (%b, %e, %h, %k) : (i8, i8, i8, i8) -> (i8, i8, i8, i8) {
// CHECK-NEXT:    ^[[BLOCK:[[:alnum:]]+]](%arg0: i8):
// CHECK-NEXT:      [[MUL:%.+]] = comb.mul %arg0, %arg0 : i8
// CHECK-NEXT:      arc.vectorize.return [[MUL]] : i8
// CHECK-NEXT:    }
// CHECK-NEXT:    [[STATE:%.+]] = arc.state @FooMux(%en, [[VEC]]#0, [[STATE]]) clock %clock latency 1 : (i1, i8, i8) -> i8
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }


hw.module @Repeated_again(in %clock : !seq.clock, in %0 : i1, in %in1: i1, in %in2: i1, out oh: i1) {
  %2:2 = arc.vectorize (%in1, %in2), (%in1, %in2), (%0, %0), (%0, %0) : (i1, i1, i1, i1, i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1):
    %4 = comb.or %arg0, %arg1, %arg2, %arg3 : i1
    arc.vectorize.return %4 : i1
  }
  hw.output %2: i1
}

// CHECK-LABEL: hw.module @Repeated_again(in %clock : !seq.clock, in %0 "" : i1, in %in1 : i1, in %in2 : i1, out oh : i1) {
// CHECK-NEXT:   [[VEC:%.+]]:2 = arc.vectorize (%in1, %in2), (%0, %0) : (i1, i1, i1, i1) -> (i1, i1) {
// CHECK-NEXT:   ^[[BLOCK:[[:alnum:]]+]](%arg0: i1, %arg1: i1):
// CHECK-NEXT:     [[OR:%.+]] = comb.or %arg0, %arg1 : i1
// CHECK-NEXT:     arc.vectorize.return [[OR]] : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output [[VEC]]#0 : i1
// CHECK-NEXT: }

hw.module @Repeated_input_1(in %b: i8, in %e: i8, in %h: i8, in %k: i8, in %en: i1, in %clock: !seq.clock) {
  %R:4 = arc.vectorize(%b, %e, %h, %k), (%b, %e, %h, %k): (i8, i8, i8, i8, i8, i8, i8, i8) -> (i8, i8, i8, i8) {
    ^bb0(%arg0: i8, %arg1: i8):
      %ret = comb.add %arg0, %arg1: i8
      arc.vectorize.return %ret: i8
  }
  %4 = arc.state @FooMux(%en, %R, %4) clock %clock latency 1 : (i1, i8, i8) -> i8
}

// CHECK-LABEL: hw.module @Repeated_input_1(in %b : i8, in %e : i8, in %h : i8, in %k : i8, in %en : i1, in %clock : !seq.clock) {
// CHECK-NEXT:    [[VEC:%.+]]:4 = arc.vectorize (%b, %e, %h, %k) : (i8, i8, i8, i8) -> (i8, i8, i8, i8) {
// CHECK-NEXT:    ^[[BLOCK:[[:alnum:]]+]](%arg0: i8):
// CHECK-NEXT:      [[FALSE:%.+]] = hw.constant false
// CHECK-NEXT:      [[EXTRACT:%.+]] = comb.extract %arg0 from 0 : (i8) -> i7
// CHECK-NEXT:      [[CONCAT:%.+]] = comb.concat [[EXTRACT]], [[FALSE]] : i7, i1
// CHECK-NEXT:      arc.vectorize.return [[CONCAT]] : i8
// CHECK-NEXT:    }
// CHECK-NEXT:    [[STATE:%.+]] = arc.state @FooMux(%en, [[VEC]]#0, [[STATE]]) clock %clock latency 1 : (i1, i8, i8) -> i8
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }
