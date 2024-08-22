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
func.func @Expressions(%arg0: !moore.i1, %arg1: !moore.l1, %arg2: !moore.i6, %arg3: !moore.i5, %arg4: !moore.i1, %arg5: !moore.array<5 x i32>, %arg6: !moore.ref<i1>, %arg7: !moore.ref<!moore.array<5 x i32>>) {
  // CHECK-NEXT: %0 = comb.concat %arg0, %arg0 : i1, i1
  // CHECK-NEXT: %1 = comb.concat %arg1, %arg1 : i1, i1
  moore.concat %arg0, %arg0 : (!moore.i1, !moore.i1) -> !moore.i2
  moore.concat %arg1, %arg1 : (!moore.l1, !moore.l1) -> !moore.l2

  // CHECK-NEXT: comb.replicate %arg0 : (i1) -> i2
  // CHECK-NEXT: comb.replicate %arg1 : (i1) -> i2
  moore.replicate %arg0 : i1 -> i2
  moore.replicate %arg1 : l1 -> l2

  // CHECK-NEXT: %name = hw.wire %arg0 : i1
  %name = moore.assigned_variable %arg0 : !moore.i1

  // CHECK-NEXT: %c12_i32 = hw.constant 12 : i32
  // CHECK-NEXT: %c3_i6 = hw.constant 3 : i6
  moore.constant 12 : !moore.i32
  moore.constant 3 : !moore.i6

  moore.conversion %arg0 : !moore.i1 -> !moore.l1
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 0 : i2 
  // CHECK-NEXT: comb.concat [[V0]], %arg2 : i2, i6 
  moore.conversion %arg2 : !moore.i6 -> !moore.l8
  // CHECK-NEXT: [[V0:%.+]] = comb.extract %arg2 from 4 : (i6) -> i2
  // CHECK-NEXT: [[V1:%.+]] = hw.constant 0 : i2
  // CHECK-NEXT: [[V2:%.+]] = comb.icmp eq [[V0]], [[V1]] : i2
  // CHECK-NEXT: [[V3:%.+]] = comb.extract %arg2 from 0 : (i6) -> i4
  // CHECK-NEXT: [[V4:%.+]] = hw.constant -1 : i4
  // CHECK-NEXT: comb.mux [[V2]], [[V3]], [[V4]] : i4
  moore.conversion %arg2 : !moore.i6 -> !moore.l4

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
  // CHECK-NEXT: %c0_i32 = hw.constant 0 : i32
  %c0 = moore.constant 0 : !moore.i32

  // CHECK-NEXT: comb.extract %arg2 from 2 : (i6) -> i2
  moore.extract %arg2 from 2 : !moore.i6 -> !moore.i2
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: hw.array_slice %arg5[[[V0]]] : (!hw.array<5xi32>) -> !hw.array<2xi32>
  moore.extract %arg5 from 2 : !moore.array<5 x i32> -> !moore.array<2 x i32>
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: hw.array_get %arg5[[[V0]]] : !hw.array<5xi32>
  moore.extract %arg5 from 2 : !moore.array<5 x i32> -> i32

  // CHECK-NEXT: [[V0:%.+]] = hw.constant 0 : i0
  // CHECK-NEXT: llhd.sig.extract %arg6 from [[V0]] : (!hw.inout<i1>) -> !hw.inout<i1>
  moore.extract_ref %arg6 from 0 : !moore.ref<!moore.i1> -> !moore.ref<!moore.i1>
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: llhd.sig.array_slice %arg7 at [[V0]] : (!hw.inout<array<5xi32>>) -> !hw.inout<array<2xi32>>
  moore.extract_ref %arg7 from 2 : !moore.ref<!moore.array<5 x i32>> -> !moore.ref<!moore.array<2 x i32>>
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: llhd.sig.array_get %arg7[[[V0]]] : !hw.inout<array<5xi32>>
  moore.extract_ref %arg7 from 2 : !moore.ref<!moore.array<5 x i32>> -> !moore.ref<i32>

  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 6 : (i32) -> i26
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i26
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i26
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i6
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant -1 : i6
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i6
  // CHECK-NEXT: [[V25:%.+]] = comb.shru %arg2, [[V24]] : i6
  // CHECK-NEXT: comb.extract [[V25]] from 0 : (i6) -> i1
  moore.dyn_extract %arg2 from %2 : !moore.i6, !moore.i32 -> !moore.i1
  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 3 : (i32) -> i29
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i29
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i29
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i3
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant -1 : i3
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i3
  // CHECK-NEXT: hw.array_slice %arg5[[[V24]]] : (!hw.array<5xi32>) -> !hw.array<2xi32>
  moore.dyn_extract %arg5 from %2 : !moore.array<5 x i32>, !moore.i32 -> !moore.array<2 x i32>
  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 3 : (i32) -> i29
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i29
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i29
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i3
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant -1 : i3
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i3
  // CHECK-NEXT: hw.array_get %arg5[[[V24]]] : !hw.array<5xi32>
  moore.dyn_extract %arg5 from %2 : !moore.array<5 x i32>, !moore.i32 -> !moore.i32

  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c0_i32 from 0 : (i32) -> i32
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i32
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i32
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c0_i32 from 0 : (i32) -> i0
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant 0 : i0
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i0
  // CHECK-NEXT: llhd.sig.extract %arg6 from [[V24]] : (!hw.inout<i1>) -> !hw.inout<i1>
  moore.dyn_extract_ref %arg6 from %c0 : !moore.ref<!moore.i1>, !moore.i32 -> !moore.ref<!moore.i1>
  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 3 : (i32) -> i29
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i29
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i29
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i3
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant -1 : i3
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i3
  // CHECK-NEXT: llhd.sig.array_slice %arg7 at [[V24]] : (!hw.inout<array<5xi32>>) -> !hw.inout<array<2xi32>>
  moore.dyn_extract_ref %arg7 from %2 : !moore.ref<!moore.array<5 x i32>>, !moore.i32 -> !moore.ref<!moore.array<2 x i32>>
  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 3 : (i32) -> i29
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i29
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i29
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i3
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant -1 : i3
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i3
  // CHECK-NEXT: llhd.sig.array_get %arg7[[[V24]]] : !hw.inout<array<5xi32>>
  moore.dyn_extract_ref %arg7 from %2 : !moore.ref<!moore.array<5 x i32>>, !moore.i32 -> !moore.ref<!moore.i32>

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

  // CHECK-NEXT: [[ZERO:%.+]] = hw.constant 0 : i6
  // CHECK-NEXT: comb.sub [[ZERO]], %arg2 : i6
  moore.neg %arg2 : !moore.i6

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

  // CHECK-NEXT: [[RES:%.+]] = scf.if %arg0 -> (i6) {
  // CHECK-NEXT:   scf.yield %arg2 : i6
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[TMP:%.+]] = hw.constant 19 : i6
  // CHECK-NEXT:   scf.yield [[TMP]] : i6
  // CHECK-NEXT: }
  // CHECK-NEXT: comb.parity [[RES]] : i6
  %k0 = moore.conditional %arg0 : i1 -> i6 {
    moore.yield %arg2 : i6
  } {
    %0 = moore.constant 19 : i6
    moore.yield %0 : i6
  }
  moore.reduce_xor %k0 : i6 -> i1

  // CHECK-NEXT: [[RES:%.+]] = scf.if %arg1 -> (i6) {
  // CHECK-NEXT:   [[TMP:%.+]] = hw.constant 0 : i6
  // CHECK-NEXT:   scf.yield [[TMP]] : i6
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[TMP:%.+]] = hw.constant 19 : i6
  // CHECK-NEXT:   scf.yield [[TMP]] : i6
  // CHECK-NEXT: }
  // CHECK-NEXT: comb.parity [[RES]] : i6
  %k1 = moore.conditional %arg1 : l1 -> l6 {
    %0 = moore.constant bXXXXXX : l6
    moore.yield %0 : l6
  } {
    %0 = moore.constant 19 : l6
    moore.yield %0 : l6
  }
  moore.reduce_xor %k1 : l6 -> l1

  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @AdvancedConversion
func.func @AdvancedConversion(%arg0: !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>) -> (!moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>, !moore.i320) {
  // CHECK: [[V0:%.+]] = hw.constant 3978585893941511189997889893581765703992223160870725712510875979948892565035285336817671 : i320
  %0 = moore.constant 3978585893941511189997889893581765703992223160870725712510875979948892565035285336817671 : i320
  // CHECK: [[V1:%.+]] = hw.bitcast [[V0]] : (i320) -> !hw.array<5xstruct<exp_bits: i32, man_bits: i32>> 
  %1 = moore.conversion %0 : !moore.i320 -> !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>
  // CHECK: [[V2:%.+]] = hw.bitcast %arg0 : (!hw.array<5xstruct<exp_bits: i32, man_bits: i32>>) -> i320
  %2 = moore.conversion %arg0 : !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>> -> !moore.i320  
  // CHECK: return [[V1]], [[V2]]
  return %1, %2 : !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>, !moore.i320
}

// CHECK-LABEL: func @Statements
func.func @Statements(%arg0: !moore.i42) {
  // CHECK: %x = llhd.sig
  %x = moore.variable : <i42>
  // CHECK: [[TMP:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %x, %arg0 after [[TMP]] : !hw.inout<i42>
  moore.blocking_assign %x, %arg0 : i42
  // CHECK: [[TMP:%.+]] = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.drv %x, %arg0 after [[TMP]] : !hw.inout<i42>
  moore.nonblocking_assign %x, %arg0 : i42
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

// CHECK-LABEL: hw.module private @Null() {
moore.module private @Null() {

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

// CHECK-LABEL: hw.module private @SubModule_0(in
// CHECK-SAME: %[[V0:.*]] : i1, in
// CHECK-SAME: %[[V1:.*]] : i1, out c : i1) {
moore.module private @SubModule_0(in %a : !moore.l1, in %b : !moore.l1, out c : !moore.l1) {
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

moore.module @Variable() {
  // CHECK: [[TMP0:%.+]] = hw.constant 0 : i32
  // CHECK: %a = llhd.sig [[TMP0]] : i32
  %a = moore.variable : <i32>

  // CHECK: [[TMP1:%.+]] = hw.constant 0 : i8
  // CHECK: %b1 = llhd.sig [[TMP1]] : i8
  %b1 = moore.variable : <i8>

  // CHECK: [[PRB:%.+]] = llhd.prb %b1 : !hw.inout<i8>
  %0 = moore.read %b1 : <i8>
  // CHECK: %b2 = llhd.sig [[PRB]] : i8
  %b2 = moore.variable %0 : <i8>

  // CHECK: %true = hw.constant true
  %1 = moore.constant 1 : l1
  // CHECK: %l = llhd.sig %true : i1
  %l = moore.variable %1 : <l1>
  // CHECK: [[TMP:%.+]] = hw.constant 0 : i19
  // CHECK: %m = llhd.sig [[TMP]] : i19
  %m = moore.variable : <l19>

  // CHECK: [[TMP2:%.+]] = hw.constant 10 : i32
  %3 = moore.constant 10 : i32
  
  // CHECK: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %a, [[TMP2]] after [[TIME]] : !hw.inout<i32>
  moore.assign %a, %3 : i32

  // CHECK: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @Struct
moore.module @Struct(in %a : !moore.i32, in %b : !moore.i32, in %arg0 : !moore.struct<{exp_bits: i32, man_bits: i32}>, in %arg1 : !moore.ref<!moore.struct<{exp_bits: i32, man_bits: i32}>>, out a : !moore.i32, out b : !moore.struct<{exp_bits: i32, man_bits: i32}>, out c : !moore.struct<{exp_bits: i32, man_bits: i32}>) {
  // CHECK: hw.struct_extract %arg0["exp_bits"] : !hw.struct<exp_bits: i32, man_bits: i32>
  %0 = moore.struct_extract %arg0, "exp_bits" : !moore.struct<{exp_bits: i32, man_bits: i32}> -> i32

  // CHECK: llhd.sig.struct_extract %arg1["exp_bits"] : !hw.inout<struct<exp_bits: i32, man_bits: i32>>
  %ref = moore.struct_extract_ref %arg1, "exp_bits" : <!moore.struct<{exp_bits: i32, man_bits: i32}>> -> <i32>
  moore.assign %ref, %0 : !moore.i32
  
  // CHECK: [[C0:%.+]] = hw.constant 0 : i64
  // CHECK: [[INIT:%.+]] = hw.bitcast [[C0]] : (i64) -> !hw.struct<exp_bits: i32, man_bits: i32>
  // CHECK: llhd.sig [[INIT]] : !hw.struct<exp_bits: i32, man_bits: i32>
  // CHECK: llhd.sig %arg0 : !hw.struct<exp_bits: i32, man_bits: i32>
  %1 = moore.variable : <struct<{exp_bits: i32, man_bits: i32}>>
  %2 = moore.variable %arg0 : <struct<{exp_bits: i32, man_bits: i32}>>

  %3 = moore.read %1 : <struct<{exp_bits: i32, man_bits: i32}>>
  %4 = moore.read %2 : <struct<{exp_bits: i32, man_bits: i32}>>

  // CHECK; hw.struct_create %a, %b : !hw.struct<a: i32, b: i32>
  moore.struct_create %a, %b : !moore.i32, !moore.i32 -> struct<{a: i32, b: i32}>

  moore.output %0, %3, %4 : !moore.i32, !moore.struct<{exp_bits: i32, man_bits: i32}>, !moore.struct<{exp_bits: i32, man_bits: i32}>
}

// CHECK-LABEL: func.func @CaseXZ(
func.func @CaseXZ(%arg0: !moore.l8, %arg1: !moore.l8) {
  // CHECK: hw.constant -124 : i8
  // CHECK: hw.constant -120 : i8
  %0 = moore.constant b10XX01ZZ : l8
  %1 = moore.constant b1XX01ZZ0 : l8

  // CHECK: comb.icmp ceq %arg0, %arg1 : i8
  moore.casez_eq %arg0, %arg1 : l8
  // CHECK: [[MASK:%.+]] = hw.constant -7 : i8
  // CHECK: [[TMP1:%.+]] = comb.and %arg0, [[MASK]]
  // CHECK: [[TMP2:%.+]] = hw.constant -120 : i8
  // CHECK: comb.icmp ceq [[TMP1]], [[TMP2]] : i8
  moore.casez_eq %arg0, %1 : l8
  // CHECK: [[MASK:%.+]] = hw.constant -4 : i8
  // CHECK: [[TMP1:%.+]] = comb.and %arg1, [[MASK]]
  // CHECK: [[TMP2:%.+]] = hw.constant -124 : i8
  // CHECK: comb.icmp ceq [[TMP1]], [[TMP2]] : i8
  moore.casez_eq %0, %arg1 : l8
  // CHECK: [[TMP1:%.+]] = hw.constant -128 : i8
  // CHECK: [[TMP2:%.+]] = hw.constant -120 : i8
  // CHECK: comb.icmp ceq [[TMP1]], [[TMP2]] : i8
  moore.casez_eq %0, %1 : l8

  // CHECK: comb.icmp ceq %arg0, %arg1 : i8
  moore.casexz_eq %arg0, %arg1 : l8
  // CHECK: [[MASK:%.+]] = hw.constant -103 : i8
  // CHECK: [[TMP1:%.+]] = comb.and %arg0, [[MASK]]
  // CHECK: [[TMP2:%.+]] = hw.constant -120 : i8
  // CHECK: comb.icmp ceq [[TMP1]], [[TMP2]] : i8
  moore.casexz_eq %arg0, %1 : l8
  // CHECK: [[MASK:%.+]] = hw.constant -52 : i8
  // CHECK: [[TMP1:%.+]] = comb.and %arg1, [[MASK]]
  // CHECK: [[TMP2:%.+]] = hw.constant -124 : i8
  // CHECK: comb.icmp ceq [[TMP1]], [[TMP2]] : i8
  moore.casexz_eq %0, %arg1 : l8
  // CHECK: [[TMP1:%.+]] = hw.constant -128 : i8
  // CHECK: [[TMP2:%.+]] = hw.constant -120 : i8
  // CHECK: comb.icmp ceq [[TMP1]], [[TMP2]] : i8
  moore.casexz_eq %0, %1 : l8

  return
}

// CHECK-LABEL: hw.module @Procedures
moore.module @Procedures() {
  // CHECK: llhd.process {
  // CHECK:   func.call @dummyA()
  // CHECK:   llhd.halt
  // CHECK: }
  moore.procedure initial {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // CHECK: llhd.final {
  // CHECK:   func.call @dummyA()
  // CHECK:   llhd.halt
  // CHECK: }
  moore.procedure final {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // CHECK: llhd.process {
  // CHECK:   cf.br ^[[BB:.+]]
  // CHECK: ^[[BB]]:
  // CHECK:   func.call @dummyA()
  // CHECK:   cf.br ^[[BB]]
  // CHECK: }
  moore.procedure always {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // CHECK: llhd.process {
  // CHECK:   cf.br ^[[BB:.+]]
  // CHECK: ^[[BB]]:
  // CHECK:   func.call @dummyA()
  // CHECK:   cf.br ^[[BB]]
  // CHECK: }
  moore.procedure always_ff {
    func.call @dummyA() : () -> ()
    moore.return
  }

  // TODO: moore.procedure always_comb
  // TODO: moore.procedure always_latch
}

func.func private @dummyA() -> ()
func.func private @dummyB() -> ()
func.func private @dummyC() -> ()

// CHECK-LABEL: hw.module @WaitEvent
moore.module @WaitEvent() {
  // CHECK: %a = llhd.sig
  // CHECK: [[PRB_A6:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A5:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A4:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A3:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A2:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A1:%.+]] = llhd.prb %a
  // CHECK: [[PRB_A0:%.+]] = llhd.prb %a
  // CHECK: %b = llhd.sig
  // CHECK: [[PRB_B2:%.+]] = llhd.prb %b
  // CHECK: [[PRB_B1:%.+]] = llhd.prb %b
  // CHECK: [[PRB_B0:%.+]] = llhd.prb %b
  // CHECK: %c = llhd.sig
  // CHECK: [[PRB_C:%.+]] = llhd.prb %c
  %a = moore.variable : <i1>
  %b = moore.variable : <i1>
  %c = moore.variable : <i1>

  // CHECK: llhd.process {
  // CHECK:   func.call @dummyA()
  // CHECK:   cf.br ^[[WAIT:.+]]
  // CHECK: ^[[WAIT]]:
  // CHECK:   func.call @dummyB()
  // CHECK:   llhd.wait ^[[CHECK:.+]]
  // CHECK: ^[[CHECK]]:
  // CHECK:   func.call @dummyB()
  // CHECK:   cf.br ^[[WAIT:.+]]
  // CHECK: ^[[RESUME:.+]]:
  // CHECK:   func.call @dummyC()
  // CHECK:   llhd.prb %a
  // CHECK:   llhd.halt
  // CHECK: }
  moore.procedure initial {
    func.call @dummyA() : () -> ()
    moore.wait_event {
      func.call @dummyB() : () -> ()
    }
    func.call @dummyC() : () -> ()
    moore.read %a : <i1>
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE:%.+]] = llhd.prb %a
    // CHECK:   llhd.wait ([[PRB_A0]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER:%.+]] = llhd.prb %a
    // CHECK:   [[TMP:%.+]] = comb.icmp bin ne [[BEFORE]], [[AFTER]]
    // CHECK:   cf.cond_br [[TMP]]
    moore.wait_event {
      %0 = moore.read %a : <i1>
      moore.detect_event any %0 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE_A:%.+]] = llhd.prb %a
    // CHECK:   [[BEFORE_B:%.+]] = llhd.prb %b
    // CHECK:   llhd.wait ([[PRB_A1]], [[PRB_B0]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER_A:%.+]] = llhd.prb %a
    // CHECK:   [[AFTER_B:%.+]] = llhd.prb %b
    // CHECK:   [[TMP1:%.+]] = comb.icmp bin ne [[BEFORE_A]], [[AFTER_A]]
    // CHECK:   [[TMP2:%.+]] = comb.and bin [[TMP1]], [[AFTER_B]]
    // CHECK:   cf.cond_br [[TMP2]]
    moore.wait_event {
      %0 = moore.read %a : <i1>
      %1 = moore.read %b : <i1>
      moore.detect_event any %0 if %1 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE_A:%.+]] = llhd.prb %a
    // CHECK:   [[BEFORE_B:%.+]] = llhd.prb %b
    // CHECK:   [[BEFORE_C:%.+]] = llhd.prb %c
    // CHECK:   llhd.wait ([[PRB_A2]], [[PRB_B1]], [[PRB_C]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER_A:%.+]] = llhd.prb %a
    // CHECK:   [[AFTER_B:%.+]] = llhd.prb %b
    // CHECK:   [[AFTER_C:%.+]] = llhd.prb %c
    // CHECK:   [[TMP1:%.+]] = comb.icmp bin ne [[BEFORE_A]], [[AFTER_A]]
    // CHECK:   [[TMP2:%.+]] = comb.icmp bin ne [[BEFORE_B]], [[AFTER_B]]
    // CHECK:   [[TMP3:%.+]] = comb.icmp bin ne [[BEFORE_C]], [[AFTER_C]]
    // CHECK:   [[TMP4:%.+]] = comb.or bin [[TMP1]], [[TMP2]], [[TMP3]]
    // CHECK:   cf.cond_br [[TMP4]]
    moore.wait_event {
      %0 = moore.read %a : <i1>
      %1 = moore.read %b : <i1>
      %2 = moore.read %c : <i1>
      moore.detect_event any %0 : i1
      moore.detect_event any %1 : i1
      moore.detect_event any %2 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE:%.+]] = llhd.prb %a
    // CHECK:   llhd.wait ([[PRB_A3]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER:%.+]] = llhd.prb %a
    // CHECK:   [[TRUE:%.+]] = hw.constant true
    // CHECK:   [[TMP1:%.+]] = comb.xor bin [[BEFORE]], [[TRUE]]
    // CHECK:   [[TMP2:%.+]] = comb.and bin [[TMP1]], [[AFTER]]
    // CHECK:   cf.cond_br [[TMP2]]
    moore.wait_event {
      %0 = moore.read %a : <i1>
      moore.detect_event posedge %0 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE:%.+]] = llhd.prb %a
    // CHECK:   llhd.wait ([[PRB_A4]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER:%.+]] = llhd.prb %a
    // CHECK:   [[TRUE:%.+]] = hw.constant true
    // CHECK:   [[TMP1:%.+]] = comb.xor bin [[AFTER]], [[TRUE]]
    // CHECK:   [[TMP2:%.+]] = comb.and bin [[BEFORE]], [[TMP1]]
    // CHECK:   cf.cond_br [[TMP2]]
    moore.wait_event {
      %0 = moore.read %a : <i1>
      moore.detect_event negedge %0 : i1
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE:%.+]] = llhd.prb %a
    // CHECK:   llhd.wait ([[PRB_A5]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER:%.+]] = llhd.prb %a
    // CHECK:   [[TRUE:%.+]] = hw.constant true
    // CHECK:   [[TMP1:%.+]] = comb.xor bin [[BEFORE]], [[TRUE]]
    // CHECK:   [[TMP2:%.+]] = comb.and bin [[TMP1]], [[AFTER]]
    // CHECK:   [[TMP3:%.+]] = comb.xor bin [[AFTER]], [[TRUE]]
    // CHECK:   [[TMP4:%.+]] = comb.and bin [[BEFORE]], [[TMP3]]
    // CHECK:   [[TMP5:%.+]] = comb.or bin [[TMP2]], [[TMP4]]
    // CHECK:   cf.cond_br [[TMP5]]
    moore.wait_event {
      %0 = moore.read %a : <i1>
      moore.detect_event edge %0 : i1
    }
    moore.return
  }

  // CHECK: [[PRB_A:%.+]] = llhd.prb %a
  // CHECK: [[PRB_B:%.+]] = llhd.prb %b
  // CHECK: llhd.process {
  %cond = moore.constant 0 : i1
  moore.procedure always_comb {
    // CHECK:   cf.br ^[[BB1:.+]]
    // CHECK: ^[[BB1]]:
    // CHECK:   llhd.prb %a
    // CHECK:   llhd.prb %b
    // CHECK:   cf.br ^[[BB2:.+]]
    // CHECK: ^[[BB2]]:
    // CHECK:   llhd.wait ([[PRB_A]], [[PRB_B]] : {{.*}}), ^[[BB1]]
    %1 = moore.conditional %cond : i1 -> i1 {
      %2 = moore.read %a : <i1>
      moore.yield %2 : !moore.i1
    } {
      %3 = moore.read %b : <i1>
      moore.yield %3 : !moore.i1
    }
    moore.return
  }

  // CHECK: [[PRB_D:%.+]] = llhd.prb %d
  // CHECK: llhd.process {
  // CHECK:   cf.br ^[[BB1:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK:   llhd.prb %d
  // CHECK:   cf.br ^[[BB2:.+]]
  // CHECK: ^[[BB2]]:
  // CHECK:   llhd.wait ([[PRB_D]] : {{.*}}), ^[[BB1]]
  moore.procedure always_latch {
    %3 = moore.read %d : <i1>
    moore.return
  }

  // CHECK: %d = llhd.sig %false
  %d = moore.variable : <i1>

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: llhd.wait ([[PRB_A6]], [[PRB_B2]] :
    moore.wait_event {
      %0 = moore.constant 0 : i1
      %1 = moore.conditional %0 : i1 -> i1 {
        %2 = moore.read %a : <i1>
        moore.yield %2 : !moore.i1
      } {
        %3 = moore.read %b : <i1>
        moore.yield %3 : !moore.i1
      }
      moore.detect_event any %1 : i1
    }
    moore.return
  }
}

// Just check that block without predecessors are handled without crashing
// CHECK-LABEL: @NoPredecessorBlockErasure
moore.module @NoPredecessorBlockErasure(in %clk_i : !moore.l1, in %raddr_i : !moore.array<2 x l5>, out rdata_o : !moore.array<2 x l32>, in %waddr_i : !moore.array<1 x l5>, in %wdata_i : !moore.array<1 x l32>, in %we_i : !moore.l1) {
  %0 = moore.constant 0 : l32
  %1 = moore.constant 1 : i32
  %2 = moore.constant 0 : i32
  %rdata_o = moore.variable : <array<2 x l32>>
  %mem = moore.variable : <array<32 x l32>>
  moore.procedure always_ff {
    cf.br ^bb1(%2 : !moore.i32)
  ^bb1(%4: !moore.i32):  // 2 preds: ^bb0, ^bb8
    moore.return
  ^bb2:  // no predecessors
    cf.br ^bb3(%2 : !moore.i32)
  ^bb3(%5: !moore.i32):  // 2 preds: ^bb2, ^bb6
    cf.br ^bb8
  ^bb4:  // no predecessors
    cf.br ^bb6
  ^bb5:  // no predecessors
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %6 = moore.add %5, %1 : i32
    cf.br ^bb3(%6 : !moore.i32)
  ^bb7:  // no predecessors
    %7 = moore.extract_ref %mem from 0 : <array<32 x l32>> -> <l32>
    moore.nonblocking_assign %7, %0 : l32
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb3, ^bb7
    %8 = moore.add %4, %1 : i32
    cf.br ^bb1(%8 : !moore.i32)
  }
  %3 = moore.read %rdata_o : <array<2 x l32>>
  moore.output %3 : !moore.array<2 x l32>
}
