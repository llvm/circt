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
func.func @Expressions(%arg0: !moore.bit, %arg1: !moore.logic, %arg2: !moore.packed<range<bit, 5:0>>, %arg3: !moore.packed<range<bit<signed>, 4:0>>) {
  // CHECK-NEXT: %0 = comb.concat %arg0, %arg0 : i1, i1
  // CHECK-NEXT: %1 = comb.concat %arg1, %arg1 : i1, i1
  moore.concat %arg0, %arg0 : (!moore.bit, !moore.bit) -> !moore.packed<range<bit, 1:0>>
  moore.concat %arg1, %arg1 : (!moore.logic, !moore.logic) -> !moore.packed<range<logic, 1:0>>

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

  // CHECK-NEXT: return
  return
}
