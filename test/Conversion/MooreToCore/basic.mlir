// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @FuncArgsAndReturns
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func.func @FuncArgsAndReturns(%arg0: !moore.i8, %arg1: !moore.i32, %arg2: !moore.i1) -> !moore.i8 {
  // CHECK-NEXT: return %arg0 : i8
  return %arg0 : !moore.i8
}

// CHECK-LABEL: func @ControlFlow
// CHECK-SAME: (%arg0: i32, %arg1: i1)
func.func @ControlFlow(%arg0: !moore.i32, %arg1: i1) {
  // CHECK-NEXT:   cf.br ^bb1(%arg0 : i32)
  // CHECK-NEXT: ^bb1(%0: i32):
  // CHECK-NEXT:   cf.cond_br %arg1, ^bb1(%0 : i32), ^bb2(%arg0 : i32)
  // CHECK-NEXT: ^bb2(%1: i32):
  // CHECK-NEXT:   return
  cf.br ^bb1(%arg0: !moore.i32)
^bb1(%0: !moore.i32):
  cf.cond_br %arg1, ^bb1(%0 : !moore.i32), ^bb2(%arg0 : !moore.i32)
^bb2(%1: !moore.i32):
  return
}

// CHECK-LABEL: func @Calls
// CHECK-SAME: (%arg0: i8, %arg1: i32, %arg2: i1) -> i8
func.func @Calls(%arg0: !moore.i8, %arg1: !moore.i32, %arg2: !moore.i1) -> !moore.i8 {
  // CHECK-NEXT: %true =
  // CHECK-NEXT: call @ControlFlow(%arg1, %true) : (i32, i1) -> ()
  // CHECK-NEXT: [[TMP:%.+]] = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (i8, i32, i1) -> i8
  // CHECK-NEXT: return [[TMP]] : i8
  %true = hw.constant true
  call @ControlFlow(%arg1, %true) : (!moore.i32, i1) -> ()
  %0 = call @FuncArgsAndReturns(%arg0, %arg1, %arg2) : (!moore.i8, !moore.i32, !moore.i1) -> !moore.i8
  return %0 : !moore.i8
}

// CHECK-LABEL: func @UnrealizedConversionCast
func.func @UnrealizedConversionCast(%arg0: !moore.i8) -> !moore.i16 {
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %arg0, %arg0 : i8, i8
  // CHECK-NEXT: return [[TMP]] : i16
  %0 = builtin.unrealized_conversion_cast %arg0 : !moore.i8 to i8
  %1 = comb.concat %0, %0 : i8, i8
  %2 = builtin.unrealized_conversion_cast %1 : i16 to !moore.i16
  return %2 : !moore.i16
}

// CHECK-LABEL: func @Expressions
func.func @Expressions(%arg0: !moore.i1, %arg1: !moore.l1, %arg2: !moore.i6, %arg3: !moore.i5, %arg4: !moore.i1) {
  // CHECK-NEXT: %0 = comb.concat %arg0, %arg0 : i1, i1
  // CHECK-NEXT: %1 = comb.concat %arg1, %arg1 : i1, i1
  moore.concat %arg0, %arg0 : (!moore.i1, !moore.i1) -> !moore.i2
  moore.concat %arg1, %arg1 : (!moore.l1, !moore.l1) -> !moore.l2

  // CHECK-NEXT: comb.replicate %arg0 : (i1) -> i2
  // CHECK-NEXT: comb.replicate %arg1 : (i1) -> i2
  moore.replicate %arg0 : i1 -> i2
  moore.replicate %arg1 : l1 -> l2

  // CHECK-NEXT: %c12_i32 = hw.constant 12 : i32
  // CHECK-NEXT: %c3_i6 = hw.constant 3 : i6
  moore.constant 12 : !moore.i32
  moore.constant 3 : !moore.i6

  // CHECK-NEXT: hw.bitcast %arg0 : (i1) -> i1
  moore.conversion %arg0 : !moore.i1 -> !moore.l1

  // CHECK-NEXT: [[V0:%.+]] = hw.constant 0 : i5
  // CHECK-NEXT: [[V1:%.+]] = comb.concat [[V0]], %arg0 : i5, i1
  // CHECK-NEXT: comb.shl %arg2, [[V1]] : i6
  moore.shl %arg2, %arg0 : !moore.i6, !moore.i1

  // CHECK-NEXT: [[V2:%.+]] = comb.extract %arg2 from 5 : (i6) -> i1
  // CHECK-NEXT: [[V3:%.+]] = hw.constant false
  // CHECK-NEXT: [[V4:%.+]] = comb.icmp eq [[V2]], [[V3]] : i1
  // CHECK-NEXT: [[V5:%.+]] = comb.extract %arg2 from 0 : (i6) -> i5
  // CHECK-NEXT: [[V6:%.+]] = hw.constant -1 : i5
  // CHECK-NEXT: [[V7:%.+]] = comb.mux [[V4]], [[V5]], [[V6]] : i5
  // CHECK-NEXT: comb.shl %arg3, [[V7]] : i5
  moore.shl %arg3, %arg2 : !moore.i5, !moore.i6

  // CHECK-NEXT: [[V8:%.+]] = hw.constant 0 : i5
  // CHECK-NEXT: [[V9:%.+]] = comb.concat [[V8]], %arg0 : i5, i1
  // CHECK-NEXT: comb.shru %arg2, [[V9]] : i6
  moore.shr %arg2, %arg0 : !moore.i6, !moore.i1

  // CHECK-NEXT: comb.shrs %arg2, %arg2 : i6
  moore.ashr %arg2, %arg2 : !moore.i6, !moore.i6

  // CHECK-NEXT: [[V10:%.+]] = comb.extract %arg2 from 5 : (i6) -> i1
  // CHECK-NEXT: [[V11:%.+]] = hw.constant false
  // CHECK-NEXT: [[V12:%.+]] = comb.icmp eq [[V10]], [[V11]] : i1
  // CHECK-NEXT: [[V13:%.+]] = comb.extract %arg2 from 0 : (i6) -> i5
  // CHECK-NEXT: [[V14:%.+]] = hw.constant -1 : i5
  // CHECK-NEXT: [[V15:%.+]] = comb.mux [[V12]], [[V13]], [[V14]] : i5
  // CHECK-NEXT: comb.shrs %arg3, [[V15]] : i5
  moore.ashr %arg3, %arg2 : !moore.i5, !moore.i6

  // CHECK-NEXT: %c2_i32 = hw.constant 2 : i32
  %2 = moore.constant 2 : !moore.i32

  // CHECK-NEXT: [[V16:%.+]] = comb.extract %c2_i32 from 6 : (i32) -> i26
  // CHECK-NEXT: %c0_i26 = hw.constant 0 : i26
  // CHECK-NEXT: [[V17:%.+]] = comb.icmp eq [[V16]], %c0_i26 : i26
  // CHECK-NEXT: [[V18:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i6
  // CHECK-NEXT: %c-1_i6 = hw.constant -1 : i6
  // CHECK-NEXT: [[V19:%.+]] = comb.mux [[V17]], [[V18]], %c-1_i6 : i6
  // CHECK-NEXT: [[V20:%.+]] = comb.shru %arg2, [[V19]] : i6
  // CHECK-NEXT: comb.extract [[V20]] from 0 : (i6) -> i2
  moore.extract %arg2 from %2 : !moore.i6, !moore.i32 -> !moore.i2

  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 6 : (i32) -> i26
  // CHECK-NEXT: %c0_i26_3 = hw.constant 0 : i26
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], %c0_i26_3 : i26
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i6
  // CHECK-NEXT: %c-1_i6_4 = hw.constant -1 : i6
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], %c-1_i6_4 : i6
  // CHECK-NEXT: [[V25:%.+]] = comb.shru %arg2, [[V24]] : i6
  // CHECK-NEXT: comb.extract [[V25]] from 0 : (i6) -> i1
  moore.extract %arg2 from %2 : !moore.i6, !moore.i32 -> !moore.i1

  // CHECK-NEXT: [[V26:%.+]] = hw.constant -1 : i6
  // CHECK-NEXT: comb.icmp eq %arg2, [[V26]] : i6
  moore.reduce_and %arg2 : !moore.i6 -> !moore.i1

  // CHECK-NEXT: [[V27:%.+]] = hw.constant false
  // CHECK-NEXT: comb.icmp ne %arg0, [[V27]] : i1
  moore.reduce_or %arg0 : !moore.i1 -> !moore.i1

  // CHECK-NEXT: comb.parity %arg1 : i1
  moore.reduce_xor %arg1 : !moore.l1 -> !moore.l1

  // CHECK-NEXT: [[V28:%.+]] = hw.constant 0 : i6
  // CHECK-NEXT: comb.icmp ne %arg2, [[V28]] : i6
  moore.bool_cast %arg2 : !moore.i6 -> !moore.i1

  // CHECK-NEXT: [[V29:%.+]] = hw.constant -1 : i6
  // CHECK-NEXT: comb.xor %arg2, [[V29]] : i6
  moore.not %arg2 : !moore.i6

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
  moore.add %arg1, %arg1 : !moore.l1
  moore.sub %arg1, %arg1 : !moore.l1
  moore.mul %arg1, %arg1 : !moore.l1
  moore.divu %arg0, %arg0 : !moore.i1
  moore.divs %arg4, %arg4 : !moore.i1
  moore.modu %arg0, %arg0 : !moore.i1
  moore.mods %arg4, %arg4 : !moore.i1
  moore.and %arg0, %arg0 : !moore.i1
  moore.or %arg0, %arg0 : !moore.i1
  moore.xor %arg0, %arg0 : !moore.i1

  // CHECK-NEXT: comb.icmp ult %arg1, %arg1 : i1
  // CHECK-NEXT: comb.icmp ule %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp ugt %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp uge %arg0, %arg0 : i1
  moore.ult %arg1, %arg1 : !moore.l1 -> !moore.l1
  moore.ule %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.ugt %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.uge %arg0, %arg0 : !moore.i1 -> !moore.i1

  // CHECK-NEXT: comb.icmp slt %arg4, %arg4 : i1
  // CHECK-NEXT: comb.icmp sle %arg4, %arg4 : i1
  // CHECK-NEXT: comb.icmp sgt %arg4, %arg4 : i1
  // CHECK-NEXT: comb.icmp sge %arg4, %arg4 : i1
  moore.slt %arg4, %arg4 : !moore.i1 -> !moore.i1
  moore.sle %arg4, %arg4 : !moore.i1 -> !moore.i1
  moore.sgt %arg4, %arg4 : !moore.i1 -> !moore.i1
  moore.sge %arg4, %arg4 : !moore.i1 -> !moore.i1

  // CHECK-NEXT: comb.icmp eq %arg1, %arg1 : i1
  // CHECK-NEXT: comb.icmp ne %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp ceq %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp cne %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp weq %arg0, %arg0 : i1
  // CHECK-NEXT: comb.icmp wne %arg0, %arg0 : i1
  moore.eq %arg1, %arg1 : !moore.l1 -> !moore.l1
  moore.ne %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.case_eq %arg0, %arg0 : !moore.i1
  moore.case_ne %arg0, %arg0 : !moore.i1
  moore.wildcard_eq %arg0, %arg0 : !moore.i1 -> !moore.i1
  moore.wildcard_ne %arg0, %arg0 : !moore.i1 -> !moore.i1

  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: hw.module @InstanceNull() {
moore.module @InstanceNull() {

  // CHECK-NEXT: hw.instance "null_instance" @Null() -> ()
  moore.instance "null_instance" @Null() -> ()

  // CHECK-NEXT: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @Null() {
moore.module @Null() {

  // CHECK-NEXT: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @Top(in
// CHECK-SAME: %[[V0:.*]] : i1, in
// CHECK-SAME: %[[V1:.*]] : i1, out out0 : i1) {
moore.module @Top(in %arg0 : !moore.l1, in %arg1 : !moore.l1, out out0 : !moore.l1) {
// CHECK-NEXT: %[[V2:.*]] = hw.instance "inst_0" @SubModule_0(a: %[[V0]]: i1, b: %[[V1]]: i1) -> (c: i1)
  %inst_0.c = moore.instance "inst_0" @SubModule_0(a: %arg0 : !moore.l1, b: %arg1 : !moore.l1) -> (c: !moore.l1)

// CHECK-NEXT: %[[V3:.*]] = hw.instance "inst_1" @SubModule_0(a: %[[V2]]: i1, b: %[[V1]]: i1) -> (c: i1)
  %inst_1.c = moore.instance "inst_1" @SubModule_0(a: %inst_0.c : !moore.l1, b: %arg1 : !moore.l1) -> (c: !moore.l1)

// CHECK-NEXT: hw.output %[[V3]] : i1
  moore.output %inst_1.c : !moore.l1
}

// CHECK-LABEL: hw.module @SubModule_0(in
// CHECK-SAME: %[[V0:.*]] : i1, in
// CHECK-SAME: %[[V1:.*]] : i1, out c : i1) {
moore.module @SubModule_0(in %a : !moore.l1, in %b : !moore.l1, out c : !moore.l1) {
  // CHECK-NEXT: %[[V2:.*]] = comb.and %[[V0]], %[[V1]] : i1
  %0 = moore.and %a, %b : !moore.l1

  // CHECK-NEXT: hw.output %[[V2]] : i1
  moore.output %0 : !moore.l1
}

// CHECK-LABEL: hw.module @ParamTest() {
moore.module @ParamTest(){

  // CHECK-NEXT: [[Pa:%.+]] = hw.constant 1 : i32
  // CHECK-NEXT: %p1 = hw.wire [[Pa]] sym @parameter_p1 : i32
  %0 = moore.constant 1 : l32
  %p1 = moore.named_constant parameter %0 : l32

  // CHECK-NEXT: [[LPa:%.+]] = hw.constant 2 : i32
  // CHECK-NEXT: %lp1 = hw.wire [[LPa]] sym @localparameter_lp1 : i32
  %1 = moore.constant 2 : l32
  %lp1 = moore.named_constant localparam %1 : l32

  // CHECK-NEXT: [[SPa:%.+]] = hw.constant 3 : i32
  // CHECK-NEXT: %sp1 = hw.wire [[SPa]] sym @specparameter_sp1 : i32
  %2 = moore.constant 3 : l32
  %sp1 = moore.named_constant specparam %2 : l32
}
