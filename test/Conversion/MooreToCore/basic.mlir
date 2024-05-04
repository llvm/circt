// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @FuncArgsAndReturns
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func.func @FuncArgsAndReturns(%arg0: !moore.byte, %arg1: !moore.int, %arg2: !moore.bit) -> !moore.byte {
  // CHECK-NEXT: return %arg0 : i8
  return %arg0 : !moore.byte
}

// CHECK-LABEL: func @ControlFlow
// CHECK-SAME: (%arg0: i32, %arg1: i1)
func.func @ControlFlow(%arg0: !moore.int, %arg1: i1) {
  // CHECK-NEXT:   cf.br ^bb1(%arg0 : i32)
  // CHECK-NEXT: ^bb1(%0: i32):
  // CHECK-NEXT:   cf.cond_br %arg1, ^bb1(%0 : i32), ^bb2(%arg0 : i32)
  // CHECK-NEXT: ^bb2(%1: i32):
  // CHECK-NEXT:   return
  cf.br ^bb1(%arg0: !moore.int)
^bb1(%0: !moore.int):
  cf.cond_br %arg1, ^bb1(%0 : !moore.int), ^bb2(%arg0 : !moore.int)
^bb2(%1: !moore.int):
  return
}

// CHECK-LABEL: func @Calls
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func.func @Calls(%arg0: !moore.byte, %arg1: !moore.int, %arg2: !moore.bit) -> !moore.byte {
  // CHECK-NEXT: %true =
  // CHECK-NEXT: call @ControlFlow(%arg1, %true) : (i32, i1) -> ()
  // CHECK-NEXT: [[TMP:%.+]] = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (i8, i32, i1) -> i8
  // CHECK-NEXT: return [[TMP]] : i8
  %true = hw.constant true
  call @ControlFlow(%arg1, %true) : (!moore.int, i1) -> ()
  %0 = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (!moore.byte, !moore.int, !moore.bit) -> !moore.byte
  return %0 : !moore.byte
}

// CHECK-LABEL: func @UnrealizedConversionCast
func.func @UnrealizedConversionCast(%arg0: !moore.byte) -> !moore.shortint {
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %arg0, %arg0 : i8, i8
  // CHECK-NEXT: return [[TMP]] : i16
  %0 = builtin.unrealized_conversion_cast %arg0 : !moore.byte to i8
  %1 = comb.concat %0, %0 : i8, i8
  %2 = builtin.unrealized_conversion_cast %1 : i16 to !moore.shortint
  return %2 : !moore.shortint
}

// CHECK-LABEL: func @Expressions
func.func @Expressions(%arg0: !moore.bit, %arg1: !moore.logic, %arg2: !moore.packed<range<bit, 5:0>>, %arg3: !moore.packed<range<bit<signed>, 4:0>>, %arg4: !moore.bit<signed>) {
  // CHECK-NEXT: %0 = comb.concat %arg0, %arg0 : i1, i1
  // CHECK-NEXT: %1 = comb.concat %arg1, %arg1 : i1, i1
  moore.concat %arg0, %arg0 : (!moore.bit, !moore.bit) -> !moore.packed<range<bit, 1:0>>
  moore.concat %arg1, %arg1 : (!moore.logic, !moore.logic) -> !moore.packed<range<logic, 1:0>>

  // CHECK-NEXT: comb.replicate %arg0 : (i1) -> i2
  // CHECK-NEXT: comb.replicate %arg1 : (i1) -> i2
  moore.replicate %arg0 : (!moore.bit) -> !moore.packed<range<bit, 1:0>>
  moore.replicate %arg1 : (!moore.logic) -> !moore.packed<range<logic, 1:0>>

  // CHECK-NEXT: %c12_i32 = hw.constant 12 : i32
  // CHECK-NEXT: %c3_i6 = hw.constant 3 : i6
  moore.constant 12 : !moore.int
  moore.constant 3 : !moore.packed<range<bit, 5:0>>

  // CHECK-NEXT: hw.bitcast %arg0 : (i1) -> i1
  moore.conversion %arg0 : !moore.bit -> !moore.logic

  // CHECK-NEXT: [[V0:%.+]] = hw.constant 0 : i5
  // CHECK-NEXT: [[V1:%.+]] = comb.concat [[V0]], %arg0 : i5, i1
  // CHECK-NEXT: comb.shl %arg2, [[V1]] : i6
  moore.shl %arg2, %arg0 : !moore.packed<range<bit, 5:0>>, !moore.bit

  // CHECK-NEXT: [[V2:%.+]] = comb.extract %arg2 from 5 : (i6) -> i1
  // CHECK-NEXT: [[V3:%.+]] = hw.constant false
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp eq [[V2]], [[V3]] : i1
  // CHECK-NEXT: [[V5:%.+]] = comb.extract %arg2 from 0 : (i6) -> i5
  // CHECK-NEXT: [[V6:%.+]] = hw.constant -1 : i5
  // CHECK-NEXT: [[V7:%.+]] = comb.mux [[V4]], [[V5]], [[V6]] : i5
  // CHECK-NEXT: comb.shl %arg3, [[V7]] : i5
  moore.shl %arg3, %arg2 : !moore.packed<range<bit<signed>, 4:0>>, !moore.packed<range<bit, 5:0>>

  // CHECK-NEXT: [[V8:%.+]] = hw.constant 0 : i5
  // CHECK-NEXT: [[V9:%.+]] = comb.concat [[V8]], %arg0 : i5, i1
  // CHECK-NEXT: comb.shru %arg2, [[V9]] : i6
  moore.shr %arg2, %arg0 : !moore.packed<range<bit, 5:0>>, !moore.bit

  // CHECK-NEXT: comb.shrs %arg2, %arg2 : i6
  moore.ashr %arg2, %arg2 : !moore.packed<range<bit, 5:0>>, !moore.packed<range<bit, 5:0>>

  // CHECK-NEXT: [[V10:%.+]] = comb.extract %arg2 from 5 : (i6) -> i1
  // CHECK-NEXT: [[V11:%.+]] = hw.constant false
  // CHECK-NEXT: [[V12:%.+]] = comb.icmp eq [[V10]], [[V11]] : i1
  // CHECK-NEXT: [[V13:%.+]] = comb.extract %arg2 from 0 : (i6) -> i5
  // CHECK-NEXT: [[V14:%.+]] = hw.constant -1 : i5
  // CHECK-NEXT: [[V15:%.+]] = comb.mux [[V12]], [[V13]], [[V14]] : i5
  // CHECK-NEXT: comb.shrs %arg3, [[V15]] : i5
  moore.ashr %arg3, %arg2 : !moore.packed<range<bit<signed>, 4:0>>, !moore.packed<range<bit, 5:0>>

  // CHECK-NEXT: %c2_i32 = hw.constant 2 : i32
  %2 = moore.constant 2 : !moore.int

  // CHECK-NEXT: [[V16:%.+]] = comb.extract %c2_i32 from 6 : (i32) -> i26
  // CHECK-NEXT: %c0_i26 = hw.constant 0 : i26
  // CHECK-NEXT: [[V17:%.+]] = comb.icmp eq [[V16]], %c0_i26 : i26
  // CHECK-NEXT: [[V18:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i6
  // CHECK-NEXT: %c-1_i6 = hw.constant -1 : i6
  // CHECK-NEXT: [[V19:%.+]] = comb.mux [[V17]], [[V18]], %c-1_i6 : i6
  // CHECK-NEXT: [[V20:%.+]] = comb.shru %arg2, [[V19]] : i6
  // CHECK-NEXT: comb.extract [[V20]] from 0 : (i6) -> i2
  moore.extract %arg2 from %2 : !moore.packed<range<bit, 5:0>>, !moore.int -> !moore.packed<range<bit, 3:2>>

  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 6 : (i32) -> i26
  // CHECK-NEXT: %c0_i26_3 = hw.constant 0 : i26
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], %c0_i26_3 : i26
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i6
  // CHECK-NEXT: %c-1_i6_4 = hw.constant -1 : i6
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], %c-1_i6_4 : i6
  // CHECK-NEXT: [[V25:%.+]] = comb.shru %arg2, [[V24]] : i6
  // CHECK-NEXT: comb.extract [[V25]] from 0 : (i6) -> i1
  moore.extract %arg2 from %2 : !moore.packed<range<bit, 5:0>>, !moore.int -> !moore.bit

  // CHECK-NEXT: [[V26:%.+]] = hw.constant -1 : i6
  // CHECK-NEXT: comb.icmp eq %arg2, [[V26]] : i6
  moore.reduce_and %arg2 : !moore.packed<range<bit, 5:0>> -> !moore.bit

  // CHECK-NEXT: [[V27:%.+]] = hw.constant false
  // CHECK-NEXT: comb.icmp ne %arg0, [[V27]] : i1
  moore.reduce_or %arg0 : !moore.bit -> !moore.bit

  // CHECK-NEXT: comb.parity %arg1 : i1
  moore.reduce_xor %arg1 : !moore.logic -> !moore.logic

  // CHECK-NEXT: [[V28:%.+]] = hw.constant 0 : i6
  // CHECK-NEXT: comb.icmp ne %arg2, [[V28]] : i6
  moore.bool_cast %arg2 : !moore.packed<range<bit, 5:0>> -> !moore.bit

  // CHECK-NEXT: [[V29:%.+]] = hw.constant -1 : i6
  // CHECK-NEXT: comb.xor %arg2, [[V29]] : i6
  moore.not %arg2 : !moore.packed<range<bit, 5:0>>

  // CHECK-NEXT: comb.add %arg1, %arg1 : i1
  // CHECK-NEXT: comb.sub %arg1, %arg1 : i1
  // CHECK-NEXT: comb.mul %arg1, %arg1 : i1
  // CHECK-NEXT: comb.divu %arg0, %arg0 : i1
  // CHECK-NEXT: comb.divs %arg4, %arg4 : i1
  // CHECK-NEXT: comb.modu %arg0, %arg0 : i1
  // CHECK-NEXT: comb.mods %arg4, %arg4 : i1
  // CHECK-NEXT: comb.and %arg0, %arg0 : i1
  // CHECK-NEXT: comb.or %arg0, %arg0 : i1
  // CHECK-NEXT: comb.xor %arg0, %arg0 : i1
  moore.add %arg1, %arg1 : !moore.logic
  moore.sub %arg1, %arg1 : !moore.logic
  moore.mul %arg1, %arg1 : !moore.logic
  moore.div %arg0, %arg0 : !moore.bit
  moore.div %arg4, %arg4 : !moore.bit<signed>
  moore.mod %arg0, %arg0 : !moore.bit
  moore.mod %arg4, %arg4 : !moore.bit<signed>
  moore.and %arg0, %arg0 : !moore.bit
  moore.or %arg0, %arg0 : !moore.bit
  moore.xor %arg0, %arg0 : !moore.bit

  // CHECK-NEXT: comb.icmp ult %arg1, %arg1 : i1
  // CHECK-NEXT: comb.icmp ule %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp ugt %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp uge %arg0, %arg0 : i1
  moore.lt %arg1, %arg1 : !moore.logic -> !moore.logic
  moore.le %arg0, %arg0 : !moore.bit -> !moore.bit
  moore.gt %arg0, %arg0 : !moore.bit -> !moore.bit
  moore.ge %arg0, %arg0 : !moore.bit -> !moore.bit

  // CHECK-NEXT: comb.icmp slt %arg4, %arg4 : i1
  // CHECK-NEXT: comb.icmp sle %arg4, %arg4 : i1
  // CHECK-NEXT: comb.icmp sgt %arg4, %arg4 : i1
  // CHECK-NEXT: comb.icmp sge %arg4, %arg4 : i1
  moore.lt %arg4, %arg4 : !moore.bit<signed> -> !moore.bit
  moore.le %arg4, %arg4 : !moore.bit<signed> -> !moore.bit
  moore.gt %arg4, %arg4 : !moore.bit<signed> -> !moore.bit
  moore.ge %arg4, %arg4 : !moore.bit<signed> -> !moore.bit

  // CHECK-NEXT: comb.icmp eq %arg1, %arg1 : i1
  // CHECK-NEXT: comb.icmp ne %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp ceq %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp cne %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp weq %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp wne %arg0, %arg0 : i1
  moore.eq %arg1, %arg1 : !moore.logic -> !moore.logic
  moore.ne %arg0, %arg0 : !moore.bit -> !moore.bit
  moore.case_eq %arg0, %arg0 : !moore.bit 
  moore.case_ne %arg0, %arg0 : !moore.bit
  moore.wildcard_eq %arg0, %arg0 : !moore.bit -> !moore.bit
  moore.wildcard_ne %arg0, %arg0 : !moore.bit -> !moore.bit

  // CHECK-NEXT: return
  return
}
