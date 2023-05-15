// RUN: circt-opt %s --arc-canonicalizer | FileCheck %s

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

// CHECK-LABEL: hw.module @icmpEqCanonicalizer
hw.module @icmpEqCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1) {
  // CHECK-NEXT: %c-1_i4 = hw.constant -1 : i4
  // CHECK-NEXT: [[V0:%.+]] = comb.and bin %arg0, %arg1, %arg2, %arg3 : i1
  %c-1_i4 = hw.constant -1 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin eq %0, %c-1_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.and bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V2:%.+]] = comb.icmp bin eq [[V1]], %c-1_i4 : i4
  %c-1_i16 = hw.constant -1 : i16
  %2 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %3 = comb.icmp bin eq %2, %c-1_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]] :
  hw.output %1, %3 : i1, i1
}

// CHECK-LABEL: hw.module @icmpNeCanonicalizer
hw.module @icmpNeCanonicalizer(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i4, %arg5: i4, %arg6: i4, %arg7: i4) -> (out0: i1, out1: i1) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: [[V0:%.+]] = comb.or bin %arg0, %arg1, %arg2, %arg3 : i1
  %c0_i4 = hw.constant 0 : i4
  %0 = comb.concat %arg0, %arg1, %arg2, %arg3 : i1, i1, i1, i1
  %1 = comb.icmp bin ne %0, %c0_i4 : i4

  // CHECK-NEXT: [[V1:%.+]] = comb.or bin %arg4, %arg5, %arg6, %arg7 : i4
  // CHECK-NEXT: [[V2:%.+]] = comb.icmp bin ne [[V1]], %c0_i4 : i4
  %c0_i16 = hw.constant 0 : i16
  %2 = comb.concat %arg4, %arg5, %arg6, %arg7 : i4, i4, i4, i4
  %3 = comb.icmp bin ne %2, %c0_i16 : i16

  // CHECK-NEXT: hw.output [[V0]], [[V2]] :
  hw.output %1, %3 : i1, i1
}
