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
  // CHECK-NEXT: ^bb1([[TMP:%.+]]: i32):
  // CHECK-NEXT:   cf.cond_br %arg1, ^bb1([[TMP]] : i32), ^bb2(%arg0 : i32)
  // CHECK-NEXT: ^bb2([[TMP:%.+]]: i32):
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
  // CHECK-NEXT: comb.concat %arg0, %arg0 : i1, i1
  // CHECK-NEXT: comb.concat %arg1, %arg1 : i1, i1
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

  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i2
  // CHECK-NEXT: [[C1:%.+]] = hw.constant 0 : i2
  // CHECK-NEXT: comb.concat [[C0]], %arg2, [[C1]] : i2, i6, i2
  moore.extract %arg2 from -2 : !moore.i6 -> !moore.i10

  // CHECK-NEXT: [[V0:%.+]] = comb.extract %arg2 from 4 : (i6) -> i2
  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i2
  // CHECK-NEXT: comb.concat [[V0]], [[C0]] : i2, i2
  moore.extract %arg2 from 4 : !moore.i6 -> !moore.i4

  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i2
  // CHECK-NEXT: [[V0:%.+]] = comb.extract %arg2 from 0 : (i6) -> i2
  // CHECK-NEXT: comb.concat [[C0]], [[V0]] : i2, i2
  moore.extract %arg2 from -2 : !moore.i6 -> !moore.i4

  // CHECK-NEXT: hw.constant 0 : i4
  moore.extract %arg2 from -6 : !moore.i6 -> !moore.i4

  // CHECK-NEXT: hw.constant 0 : i4
  moore.extract %arg2 from 6 : !moore.i6 -> !moore.i4

  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i64
  // CHECK-NEXT: [[V0:%..+]] = hw.bitcast [[C0]] : (i64) -> !hw.array<2xi32>
  // CHECK-NEXT: hw.constant 0 : i3
  // CHECK-NEXT: [[C1:%.+]] = hw.constant 0 : i64
  // CHECK-NEXT: [[V1:%.+]] = hw.bitcast [[C1]] : (i64) -> !hw.array<2xi32>
  // CHECK-NEXT: hw.array_concat [[V0]], %arg5, [[V1]] : !hw.array<2xi32>, !hw.array<5xi32>, !hw.array<2xi32>
  moore.extract %arg5 from -2 : !moore.array<5 x i32> -> !moore.array<9 x i32>

  // CHECK-NEXT: [[IDX:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: [[V0:%.+]] = hw.array_slice %arg5[[[IDX]]] : (!hw.array<5xi32>) -> !hw.array<3xi32>
  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i32
  // CHECK-NEXT: [[V1:%.+]] = hw.bitcast [[C0]] : (i32) -> !hw.array<1xi32>
  // CHECK-NEXT: hw.array_concat [[V0]], [[V1]] : !hw.array<3xi32>, !hw.array<1xi32>
  moore.extract %arg5 from 2 : !moore.array<5 x i32> -> !moore.array<4 x i32>

  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i32
  // CHECK-NEXT: [[V0:%.+]] = hw.bitcast [[C0]] : (i32) -> !hw.array<1xi32>
  // CHECK-NEXT: [[IDX:%.+]] = hw.constant 0 : i3
  // CHECK-NEXT: [[V1:%.+]] = hw.array_slice %arg5[[[IDX]]] : (!hw.array<5xi32>) -> !hw.array<1xi32>
  // CHECK-NEXT: hw.array_concat [[V0]], [[V1]] : !hw.array<1xi32>, !hw.array<1xi32>
  moore.extract %arg5 from -1 : !moore.array<5 x i32> -> !moore.array<2 x i32>

  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i64
  // CHECK-NEXT: hw.bitcast [[C0]] : (i64) -> !hw.array<2xi32>
  moore.extract %arg5 from -2 : !moore.array<5 x i32> -> !moore.array<2 x i32>

  // CHECK-NEXT: [[C0:%.+]] = hw.constant 0 : i64
  // CHECK-NEXT: hw.bitcast [[C0]] : (i64) -> !hw.array<2xi32>
  moore.extract %arg5 from 5 : !moore.array<5 x i32> -> !moore.array<2 x i32>

  // CHECK-NEXT: hw.constant 0 : i32
  moore.extract %arg5 from -2 : !moore.array<5 x i32> -> i32
  // CHECK-NEXT: hw.constant 0 : i32
  moore.extract %arg5 from 6 : !moore.array<5 x i32> -> i32

  // CHECK-NEXT: [[V0:%.+]] = hw.constant 0 : i0
  // CHECK-NEXT: llhd.sig.extract %arg6 from [[V0]] : <i1> -> <i1>
  moore.extract_ref %arg6 from 0 : !moore.ref<!moore.i1> -> !moore.ref<!moore.i1>
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: llhd.sig.array_slice %arg7 at [[V0]] : <!hw.array<5xi32>> -> <!hw.array<2xi32>>
  moore.extract_ref %arg7 from 2 : !moore.ref<!moore.array<5 x i32>> -> !moore.ref<!moore.array<2 x i32>>
  // CHECK-NEXT: [[V0:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: llhd.sig.array_get %arg7[[[V0]]] : <!hw.array<5xi32>>
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
  // CHECK-NEXT: llhd.sig.extract %arg6 from [[V24]] : <i1> -> <i1>
  moore.dyn_extract_ref %arg6 from %c0 : !moore.ref<!moore.i1>, !moore.i32 -> !moore.ref<!moore.i1>
  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 3 : (i32) -> i29
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i29
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i29
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i3
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant -1 : i3
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i3
  // CHECK-NEXT: llhd.sig.array_slice %arg7 at [[V24]] : <!hw.array<5xi32>> -> <!hw.array<2xi32>>
  moore.dyn_extract_ref %arg7 from %2 : !moore.ref<!moore.array<5 x i32>>, !moore.i32 -> !moore.ref<!moore.array<2 x i32>>
  // CHECK-NEXT: [[V21:%.+]] = comb.extract %c2_i32 from 3 : (i32) -> i29
  // CHECK-NEXT: [[CONST_0:%.+]] = hw.constant 0 : i29
  // CHECK-NEXT: [[V22:%.+]] = comb.icmp eq [[V21]], [[CONST_0]] : i29
  // CHECK-NEXT: [[V23:%.+]] = comb.extract %c2_i32 from 0 : (i32) -> i3
  // CHECK-NEXT: [[MAX:%.+]] = hw.constant -1 : i3
  // CHECK-NEXT: [[V24:%.+]] = comb.mux [[V22]], [[V23]], [[MAX]] : i3
  // CHECK-NEXT: llhd.sig.array_get %arg7[[[V24]]] : <!hw.array<5xi32>>
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

  // CHECK-NEXT: [[TMP:%.+]] = hw.constant 19 : i6
  // CHECK-NEXT: [[RES:%.+]] = comb.mux %arg0, %arg2, [[TMP]]
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
  // CHECK:        %var_l6 = llhd.sig
  // CHECK:        llhd.drv %var_l6, [[TMP]] after
  // CHECK-NEXT:   scf.yield [[TMP]] : i6
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[TMP:%.+]] = hw.constant 19 : i6
  // CHECK-NEXT:   scf.yield [[TMP]] : i6
  // CHECK-NEXT: }
  // CHECK-NEXT: comb.parity [[RES]] : i6
  %k1 = moore.conditional %arg1 : l1 -> l6 {
    %0 = moore.constant bXXXXXX : l6
    %var_l6 = moore.variable : <l6>
    moore.blocking_assign %var_l6, %0 : l6
    moore.yield %0 : l6
  } {
    %0 = moore.constant 19 : l6
    moore.yield %0 : l6
  }
  moore.reduce_xor %k1 : l6 -> l1

  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: ExtractRefArrayElement
func.func @ExtractRefArrayElement(%j: !moore.ref<array<1 x array<1 x l3>>>) -> (!moore.ref<array<1 x l3>>) {
  // CHECK: llhd.sig.array_get
  %0 = moore.extract_ref %j from 0 : <array<1 x array<1 x l3>>> -> <array<1 x l3>>
  return %0 : !moore.ref<array<1 x l3>>
}

// CHECK-LABEL: DynExtractArrayElement
func.func @DynExtractArrayElement(%j: !moore.array<2 x array<1 x l3>>, %idx: !moore.l1) -> (!moore.array<1 x l3>) {
  // CHECK: hw.array_get
  %0 = moore.dyn_extract %j from %idx : !moore.array<2 x array<1 x l3>>, !moore.l1 -> !moore.array<1 x l3>
  return %0 : !moore.array<1 x l3>
}

// CHECK-LABEL: DynExtractRefArrayElement
func.func @DynExtractRefArrayElement(%j: !moore.ref<array<2 x array<1 x l3>>>, %idx: !moore.l1) -> (!moore.ref<array<1 x l3>>) {
  // CHECK: llhd.sig.array_get
  %0 = moore.dyn_extract_ref %j from %idx : <array<2 x array<1 x l3>>>, !moore.l1 -> <array<1 x l3>>
  return %0 : !moore.ref<array<1 x l3>>
}

// CHECK-LABEL: func @AdvancedConversion
func.func @AdvancedConversion(%arg0: !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>) -> (!moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>, !moore.i320) {
  // CHECK: [[V0:%.+]] = hw.constant 3978585893941511189997889893581765703992223160870725712510875979948892565035285336817671 : i320
  %0 = moore.constant 3978585893941511189997889893581765703992223160870725712510875979948892565035285336817671 : i320
  // CHECK: [[V1:%.+]] = hw.bitcast [[V0]] : (i320) -> !hw.array<5xstruct<exp_bits: i32, man_bits: i32>> 
  %1 = moore.sbv_to_packed %0 : array<5 x struct<{exp_bits: i32, man_bits: i32}>>
  // CHECK: [[V2:%.+]] = hw.bitcast %arg0 : (!hw.array<5xstruct<exp_bits: i32, man_bits: i32>>) -> i320
  %2 = moore.packed_to_sbv %arg0 : array<5 x struct<{exp_bits: i32, man_bits: i32}>>
  // CHECK: return [[V1]], [[V2]]
  return %1, %2 : !moore.array<5 x struct<{exp_bits: i32, man_bits: i32}>>, !moore.i320
}

// CHECK-LABEL: func @Statements
func.func @Statements(%arg0: !moore.i42) {
  // CHECK: %x = llhd.sig
  %x = moore.variable : <i42>
  // CHECK: [[TMP:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %x, %arg0 after [[TMP]] : i42
  moore.blocking_assign %x, %arg0 : i42
  // CHECK: [[TMP:%.+]] = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: llhd.drv %x, %arg0 after [[TMP]] : i42
  moore.nonblocking_assign %x, %arg0 : i42
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @FormatStrings
func.func @FormatStrings(%arg0: !moore.i42, %arg1: !moore.f32, %arg2: !moore.f64) {
  // CHECK: [[TMP:%.+]] = sim.fmt.literal "hello"
  %0 = moore.fmt.literal "hello"
  // CHECK: sim.fmt.concat ([[TMP]], [[TMP]])
  %1 = moore.fmt.concat (%0, %0)
  // CHECK:  sim.fmt.dec %arg0 specifierWidth 42 : i42
  moore.fmt.int decimal %arg0, align right, pad space width 42 : i42
  // CHECK: sim.fmt.dec %arg0 isLeftAligned true paddingChar 48 : i42
  moore.fmt.int decimal %arg0, align left, pad zero : i42
  // CHECK: sim.fmt.dec %arg0 signed : i42
  moore.fmt.int decimal %arg0, align right, pad space signed : i42
  // CHECK: sim.fmt.bin %arg0 paddingChar 32 specifierWidth 42 : i42
  moore.fmt.int binary %arg0, align right, pad space width 42 : i42
  // CHECK: sim.fmt.bin %arg0 isLeftAligned true : i42
  moore.fmt.int binary %arg0, align left, pad zero : i42
  // CHECK: sim.fmt.oct %arg0 paddingChar 32 specifierWidth 42 : i42
  moore.fmt.int octal %arg0, align right, pad space width 42 : i42
  // CHECK: sim.fmt.oct %arg0 specifierWidth 42 : i42
  moore.fmt.int octal %arg0, align right, pad zero width 42 : i42
  // CHECK: sim.fmt.hex %arg0, isUpper false paddingChar 32 specifierWidth 42 : i42
  moore.fmt.int hex_lower %arg0, align right, pad space width 42 : i42
  // CHECK: sim.fmt.hex %arg0, isUpper false : i42
  moore.fmt.int hex_lower %arg0, align right, pad zero : i42
  // CHECK: sim.fmt.hex %arg0, isUpper true paddingChar 32 specifierWidth 42 : i42
  moore.fmt.int hex_upper %arg0, align right, pad space width 42 : i42

  // CHECK: sim.fmt.flt %arg1 isLeftAligned true : f32
  moore.fmt.real float %arg1, align left : f32
  // CHECK: sim.fmt.exp %arg2 isLeftAligned true : f64
  moore.fmt.real exponential %arg2, align left : f64
  // CHECK: sim.fmt.gen %arg1 isLeftAligned true : f32
  moore.fmt.real general %arg1, align left fracDigits 6 : f32
  // CHECK: sim.fmt.flt %arg2 isLeftAligned true fracDigits 10 : f64
  moore.fmt.real float %arg2, align left fracDigits 10 : f64
  // CHECK: sim.fmt.exp %arg1 fieldWidth 9 fracDigits 8 : f32
  moore.fmt.real exponential %arg1, align right fieldWidth 9 fracDigits 8 : f32
  // CHECK: sim.fmt.gen %arg2 : f64
  moore.fmt.real general %arg2, align right : f64
  // CHECK: sim.fmt.flt %arg1 fieldWidth 15 : f32
  moore.fmt.real float %arg1, align right fieldWidth 15 : f32
  // CHECK: sim.proc.print [[TMP]]
  moore.builtin.display %0
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

// CHECK-LABEL: hw.module @PreservePortOrderTop(
// CHECK-SAME:    out a : i42,
// CHECK-SAME:    in %b : i42
// CHECK-SAME:  ) {
moore.module @PreservePortOrderTop(out a: !moore.i42, in %b: !moore.i42) {
  // CHECK: [[TMP:%.+]] = hw.instance "inst" @PreservePortOrder(x: %b: i42, z: %b: i42) -> (y: i42)
  // CHECK: hw.output [[TMP]] : i42
  %0 = moore.instance "inst" @PreservePortOrder(x: %b: !moore.i42, z: %b: !moore.i42) -> (y: !moore.i42)
  moore.output %0 : !moore.i42
}

// CHECK-LABEL: hw.module private @PreservePortOrder(
// CHECK-SAME:    in %x : i42,
// CHECK-SAME:    out y : i42,
// CHECK-SAME:    in %z : i42
// CHECK-SAME:  ) {
moore.module private @PreservePortOrder(in %x: !moore.i42, out y: !moore.i42, in %z: !moore.i42) {
  moore.output %x : !moore.i42
}

// CHECK-LABEL: hw.module @Variable
moore.module @Variable() {
  // CHECK: [[TMP0:%.+]] = hw.constant 0 : i32
  // CHECK: %a = llhd.sig [[TMP0]] : i32
  %a = moore.variable : <i32>

  // CHECK: [[TMP1:%.+]] = hw.constant 0 : i8
  // CHECK: %b1 = llhd.sig [[TMP1]] : i8
  %b1 = moore.variable : <i8>

  // CHECK: [[PRB:%.+]] = llhd.prb %b1 : i8
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
  // CHECK: llhd.drv %a, [[TMP2]] after [[TIME]] : i32
  moore.assign %a, %3 : i32

  // CHECK: [[TMP:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llhd.sig [[TMP]] : !llvm.ptr
  moore.variable : <!moore.chandle>

  // CHECK: [[TMP:%.+]] = llhd.constant_time <0
  // CHECK: llhd.sig [[TMP]] : !llhd.time
  moore.variable : <!moore.time>

  // CHECK: [[TMP:%.+]] = llhd.constant_time
  // CHECK: llhd.sig [[TMP]] : !llhd.time
  %c42_fs = moore.constant_time 42 fs
  moore.variable %c42_fs : <!moore.time>

  // CHECK: [[TMP:%.+]] = arith.constant {{0.*}} : f32
  // CHECK: llhd.sig [[TMP]] : f32
  moore.variable : <!moore.f32>

  // CHECK: [[TMP:%.+]] = arith.constant {{0.*}} : f64
  // CHECK: llhd.sig [[TMP]] : f64
  moore.variable : <!moore.f64>

  // CHECK: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @Net
moore.module @Net() {
  // CHECK: [[TMP:%.+]] = hw.constant 0 : i32
  // CHECK: %a = llhd.sig [[TMP]] : i32
  %a = moore.net wire : <i32>

  // CHECK: [[PRB:%.+]] = llhd.prb %a : i32
  %0 = moore.read %a : <i32>

  // CHECK: [[TMP:%.+]] = hw.constant 0 : i32
  // CHECK: %b = llhd.sig [[TMP]] : i32
  // CHECK: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %b, [[PRB]] after [[TIME]] : i32
  %b = moore.net wire %0 : <i32>

  // CHECK: [[TMP:%.+]] = hw.constant 10 : i32
  %3 = moore.constant 10 : i32
  // CHECK: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: llhd.drv %a, [[TMP]] after [[TIME]] : i32
  moore.assign %a, %3 : i32
}

// CHECK-LABEL: hw.module @UnpackedArray
moore.module @UnpackedArray(in %arr : !moore.uarray<2 x i32>, in %sel : !moore.i1, out c : !moore.i32) {
  // CHECK: hw.array_get %arr[%sel] : !hw.array<2xi32>, i1
  %0 = moore.dyn_extract %arr from %sel : !moore.uarray<2 x i32>, !moore.i1 -> !moore.i32

  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: hw.array_get %arr[[[TRUE]]] : !hw.array<2xi32>, i1
  %1 = moore.extract %arr from 1 : !moore.uarray<2 x i32> -> !moore.i32

  // CHECK: [[C0_128:%.+]] = hw.constant 0 : i128
  // CHECK: [[INIT:%.+]] = hw.bitcast [[C0_128]] : (i128) -> !hw.array<4xi32>
  // CHECK: [[SIG_0:%.+]] = llhd.sig [[INIT]] : !hw.array<4xi32>
  %2 = moore.variable : <uarray<4 x i32>>

  // CHECK: [[C1:%.+]] = hw.constant 1 : i2
  // CHECK: llhd.sig.array_get [[SIG_0]][[[C1]]] : <!hw.array<4xi32>>
  %3 = moore.extract_ref %2 from 1 : !moore.ref<!moore.uarray<4 x i32>> -> !moore.ref<!moore.i32>
  moore.assign %3, %0 : i32

  // CHECK: [[C0_1024:%.+]] = hw.constant 0 : i1024
  // CHECK: [[INIT:%.+]] = hw.bitcast [[C0_1024]] : (i1024) -> !hw.array<4xarray<8xarray<8xi4>>>
  // CHECK: [[SIG_1:%.+]] = llhd.sig [[INIT]] : !hw.array<4xarray<8xarray<8xi4>>>
  %4 = moore.variable : <uarray<4 x uarray<8 x array<8 x i4>>>>

  moore.output %0 : !moore.i32
}

// CHECK-LABEL: hw.module @Struct
moore.module @Struct(in %a : !moore.i32, in %b : !moore.i32, in %arg0 : !moore.struct<{exp_bits: i32, man_bits: i32}>, in %arg1 : !moore.ref<!moore.struct<{exp_bits: i32, man_bits: i32}>>, out a : !moore.i32, out b : !moore.struct<{exp_bits: i32, man_bits: i32}>, out c : !moore.struct<{exp_bits: i32, man_bits: i32}>) {
  // CHECK: hw.struct_extract %arg0["exp_bits"] : !hw.struct<exp_bits: i32, man_bits: i32>
  %0 = moore.struct_extract %arg0, "exp_bits" : !moore.struct<{exp_bits: i32, man_bits: i32}> -> i32

  // CHECK: llhd.sig.struct_extract %arg1["exp_bits"] : <!hw.struct<exp_bits: i32, man_bits: i32>>
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

// CHECK-LABEL: func @ArrayCreate
// CHECK-SAME: () ->  !hw.array<2xi8>
func.func @ArrayCreate() -> !moore.array<2x!moore.i8> {
  // CHECK-NEXT: %c42_i8 = hw.constant 42 : i8 
  %c0 = moore.constant 42 : !moore.i8
  // CHECK-NEXT: [[ARR:%.*]] = hw.array_create %c42_i8, %c42_i8 : i8
  %arr = moore.array_create %c0, %c0 : !moore.i8, !moore.i8 -> !moore.array<2x!moore.i8>
  // CHECK-NEXT: return [[ARR:%.*]] : !hw.array<2xi8>
  return %arr : !moore.array<2x!moore.i8>
}

// CHECK-LABEL: func @UnpackedArrayCreate
// CHECK-SAME: () ->  !hw.array<2xi8>
func.func @UnpackedArrayCreate() -> !moore.uarray<2x!moore.i8> {
  // CHECK-NEXT: %c7_i8 = hw.constant 7 : i8 
  %a = moore.constant 7 : !moore.i8
  // CHECK-NEXT: [[ARR:%.*]] = hw.array_create %c7_i8, %c7_i8 : i8
  %arr = moore.array_create %a, %a : !moore.i8, !moore.i8 -> !moore.uarray<2x!moore.i8>
  // CHECK-NEXT: return [[ARR:%.*]] : !hw.array<2xi8>
  return %arr : !moore.uarray<2x!moore.i8>
}

// CHECK-LABEL:   hw.module @UnpackedStruct
moore.module @UnpackedStruct() {
  // CHECK: %[[C1_32:.*]] = hw.constant 1 : i32
  // CHECK: %[[C0_32:.*]] = hw.constant 0 : i32
  %0 = moore.constant 1 : i32
  %1 = moore.constant 0 : i32

  // CHECK: %[[C0_64:.*]] = hw.constant 0 : i64
  // CHECK: %[[INIT:.*]] = hw.bitcast %[[C0_64]] : (i64) -> !hw.struct<a: i32, b: i32>
  // CHECK: %[[USTRUCT:.*]] = llhd.sig %[[INIT]] : !hw.struct<a: i32, b: i32>
  %ms = moore.variable : <ustruct<{a: i32, b: i32}>>

// CHECK: llhd.process {
  moore.procedure initial {
    // CHECK: %[[STRUCT_0:.*]] = hw.struct_create (%[[C0_32]], %[[C1_32]]) : !hw.struct<a: i32, b: i32>
    %2 = moore.struct_create %1, %0 : !moore.i32, !moore.i32 -> ustruct<{a: i32, b: i32}>

    // CHECK: %[[TIME_0:.*]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK: llhd.drv %[[USTRUCT]], %[[STRUCT_0]] after %[[TIME_0]] : !hw.struct<a: i32, b: i32>
    moore.blocking_assign %ms, %2 : ustruct<{a: i32, b: i32}>

    // CHECK: %[[STRUCT_1:.*]] = hw.struct_create (%[[C1_32]], %[[C1_32]]) : !hw.struct<a: i32, b: i32>
    %3 = moore.struct_create %0, %0 : !moore.i32, !moore.i32 -> ustruct<{a: i32, b: i32}>

    // CHECK: %[[TIME_1:.*]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK: llhd.drv %[[USTRUCT]], %[[STRUCT_1]] after %[[TIME_1]] : !hw.struct<a: i32, b: i32>
    moore.blocking_assign %ms, %3 : ustruct<{a: i32, b: i32}>

    // CHECK: %[[TIME_2:.*]] = llhd.constant_time <0ns, 0d, 1e>
    // CHECK: llhd.drv %[[USTRUCT]], %[[STRUCT_1]] after %[[TIME_2]] : !hw.struct<a: i32, b: i32>
    moore.blocking_assign %ms, %3 : ustruct<{a: i32, b: i32}>

    // CHECK: llhd.halt
    moore.return
  }
  moore.output
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

// CHECK-LABEL: func.func @CmpReal
func.func @CmpReal(%arg0: !moore.f32, %arg1: !moore.f32) {
  // CHECK: arith.cmpf one, %arg0, %arg1 : f32
  moore.fne %arg0, %arg1 : f32 -> i1
  // CHECK: arith.cmpf olt, %arg0, %arg1 : f32
  moore.flt %arg0, %arg1 : f32 -> i1
  // CHECK: arith.cmpf ole, %arg0, %arg1 : f32
  moore.fle %arg0, %arg1 : f32 -> i1
  // CHECK: arith.cmpf ogt, %arg0, %arg1 : f32
  moore.fgt %arg0, %arg1 : f32 -> i1
  // CHECK: arith.cmpf oge, %arg0, %arg1 : f32
  moore.fge %arg0, %arg1 : f32 -> i1
  // CHECK: arith.cmpf oeq, %arg0, %arg1 : f32
  moore.feq %arg0, %arg1 : f32 -> i1

  return
}

// CHECK-LABEL: func.func @BinaryRealOps
func.func @BinaryRealOps(%arg0: !moore.f32, %arg1: !moore.f32) {
  // CHECK: arith.addf %arg0, %arg1 : f32
  moore.fadd %arg0, %arg1 : f32
  // CHECK: arith.subf %arg0, %arg1 : f32
  moore.fsub %arg0, %arg1 : f32
  // CHECK: arith.divf %arg0, %arg1 : f32
  moore.fdiv %arg0, %arg1 : f32
  // CHECK: arith.mulf %arg0, %arg1 : f32
  moore.fmul %arg0, %arg1 : f32
  // CHECK: math.powf %arg0, %arg1 : f32
  moore.fpow %arg0, %arg1 : f32

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
  // CHECK: %d = llhd.sig
  // CHECK: [[PRB_D4:%.+]] = llhd.prb %d
  // CHECK: [[PRB_D3:%.+]] = llhd.prb %d
  // CHECK: [[PRB_D2:%.+]] = llhd.prb %d
  // CHECK: [[PRB_D1:%.+]] = llhd.prb %d
  %a = moore.variable : <i1>
  %b = moore.variable : <i1>
  %c = moore.variable : <i1>
  %d = moore.variable : <i4>

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
    // CHECK:   cf.br ^[[RESUME:.+]]
    // CHECK: ^[[RESUME]]:
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
    // CHECK:   llhd.wait ([[PRB_A2]], [[PRB_B1]], [[PRB_C]], [[PRB_D1]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   cf.br
    moore.wait_event {
      %0 = moore.read %a : <i1>
      %1 = moore.read %b : <i1>
      %2 = moore.read %c : <i1>
      %3 = moore.read %d : <i4>
      moore.detect_event any %0 : i1
      moore.detect_event any %1 : i1
      moore.detect_event any %2 : i1
      moore.detect_event any %3 : i4
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

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE:%.+]] = llhd.prb %d
    // CHECK:   llhd.wait ([[PRB_D2]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER:%.+]] = llhd.prb %d
    // CHECK:   [[TMP1:%.+]] = comb.extract [[BEFORE]] from 0 : (i4) -> i1
    // CHECK:   [[TMP2:%.+]] = comb.extract [[AFTER]] from 0 : (i4) -> i1
    // CHECK:   [[TRUE:%.+]] = hw.constant true
    // CHECK:   [[TMP3:%.+]] = comb.xor bin [[TMP1]], [[TRUE]]
    // CHECK:   [[TMP4:%.+]] = comb.and bin [[TMP3]], [[TMP2]]
    // CHECK:   cf.cond_br [[TMP4]]
    moore.wait_event {
      %0 = moore.read %d : <i4>
      moore.detect_event posedge %0 : i4
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE:%.+]] = llhd.prb %d
    // CHECK:   llhd.wait ([[PRB_D3]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER:%.+]] = llhd.prb %d
    // CHECK:   [[TMP1:%.+]] = comb.extract [[BEFORE]] from 0 : (i4) -> i1
    // CHECK:   [[TMP2:%.+]] = comb.extract [[AFTER]] from 0 : (i4) -> i1
    // CHECK:   [[TRUE:%.+]] = hw.constant true
    // CHECK:   [[TMP3:%.+]] = comb.xor bin [[TMP2]], [[TRUE]]
    // CHECK:   [[TMP4:%.+]] = comb.and bin [[TMP1]], [[TMP3]]
    // CHECK:   cf.cond_br [[TMP4]]
    moore.wait_event {
      %0 = moore.read %d : <i4>
      moore.detect_event negedge %0 : i4
    }
    moore.return
  }

  // CHECK: llhd.process {
  moore.procedure initial {
    // CHECK:   [[BEFORE:%.+]] = llhd.prb %d
    // CHECK:   llhd.wait ([[PRB_D4]] : {{.+}}), ^[[CHECK:.+]]
    // CHECK: ^[[CHECK]]:
    // CHECK:   [[AFTER:%.+]] = llhd.prb %d
    // CHECK:   [[TMP1:%.+]] = comb.extract [[BEFORE]] from 0 : (i4) -> i1
    // CHECK:   [[TMP2:%.+]] = comb.extract [[AFTER]] from 0 : (i4) -> i1
    // CHECK:   [[TRUE:%.+]] = hw.constant true
    // CHECK:   [[TMP3:%.+]] = comb.xor bin [[TMP1]], [[TRUE]]
    // CHECK:   [[TMP4:%.+]] = comb.and bin [[TMP3]], [[TMP2]]
    // CHECK:   [[TMP5:%.+]] = comb.xor bin [[TMP2]], [[TRUE]]
    // CHECK:   [[TMP6:%.+]] = comb.and bin [[TMP1]], [[TMP5]]
    // CHECK:   [[TMP7:%.+]] = comb.or bin [[TMP4]], [[TMP6]]
    // CHECK:   cf.cond_br [[TMP7]]
    moore.wait_event {
      %0 = moore.read %d : <i4>
      moore.detect_event edge %0 : i4
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

  // CHECK: [[PRB_E:%.+]] = llhd.prb %e
  // CHECK: llhd.process {
  // CHECK:   cf.br ^[[BB1:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK:   llhd.prb %e
  // CHECK:   cf.br ^[[BB2:.+]]
  // CHECK: ^[[BB2]]:
  // CHECK:   llhd.wait ([[PRB_E]] : {{.*}}), ^[[BB1]]
  moore.procedure always_latch {
    %3 = moore.read %e : <i1>
    moore.return
  }

  // CHECK: %e = llhd.sig %false
  %e = moore.variable : <i1>

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

// CHECK-LABEL: hw.module @EmptyWaitEvent(
moore.module @EmptyWaitEvent(out out : !moore.l32) {
  // CHECK: [[OUT:%.+]] = llhd.sig %c0_i32
  // CHECK: llhd.process {
  // CHECK:   cf.br
  // CHECK: ^bb
  // CHECK:   llhd.halt
  // CHECK: ^bb{{.*}} // no predecessors
  // CHECK: }
  // CHECK: [[PRB:%.+]] = llhd.prb [[OUT]] : i32
  // CHECK: hw.output [[PRB]] : i32
  %0 = moore.constant 0 : l32
  %out = moore.variable : <l32>
  moore.procedure always {
    moore.wait_event {
    }
    moore.blocking_assign %out, %0 : l32
    moore.return
  }
  %1 = moore.read %out : <l32>
  moore.output %1 : !moore.l32
}


// CHECK-LABEL: hw.module @WaitDelay
moore.module @WaitDelay(in %d: !moore.time) {
  // CHECK: llhd.process {
  // CHECK:   [[TMP:%.+]] = llhd.constant_time <1000000fs, 0d, 0e>
  // CHECK:   func.call @dummyA()
  // CHECK:   llhd.wait delay [[TMP]], ^[[RESUME1:.+]]
  // CHECK: ^[[RESUME1]]:
  // CHECK:   func.call @dummyB()
  // CHECK:   llhd.wait delay %d, ^[[RESUME2:.+]]
  // CHECK: ^[[RESUME2]]:
  // CHECK:   func.call @dummyC()
  // CHECK:   llhd.halt
  // CHECK: }
  moore.procedure initial {
    %0 = moore.constant_time 1000000 fs
    func.call @dummyA() : () -> ()
    moore.wait_delay %0
    func.call @dummyB() : () -> ()
    moore.wait_delay %d
    func.call @dummyC() : () -> ()
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

// CHECK: [[TMP:%.+]] = hw.constant 42 : i32
%dbg0 = moore.constant 42 : l32
// CHECK: dbg.variable "a", [[TMP]] : i32
dbg.variable "a", %dbg0 : !moore.l32
// CHECK: [[SCOPE:%.+]] = dbg.scope
%dbg1 = dbg.scope "foo", "bar"
// CHECK: dbg.variable "b", [[TMP]] scope [[SCOPE]] : i32
dbg.variable "b", %dbg0 scope %dbg1 : !moore.l32
// CHECK: dbg.array [[[TMP]]] : i32
dbg.array [%dbg0] : !moore.l32
// CHECK: dbg.struct {"q": [[TMP]]} : i32
dbg.struct {"q": %dbg0} : !moore.l32

// CHECK-LABEL: hw.module @Assert
moore.module @Assert(in %cond : !moore.l1)  {
  moore.procedure always {
  // CHECK: verif.assert %cond label "cond" : i1
  moore.assert immediate %cond label "cond" : l1
  // CHECK: verif.assume %cond label "" : i1
  moore.assume observed %cond  : l1
  // CHECK: verif.cover %cond label "" : i1
  moore.cover final %cond : l1
  moore.return
  }
}

// CHECK-LABEL: func.func @ConstantString
func.func @ConstantString() {
  // CHECK: hw.constant 1415934836 : i32
  %str = moore.constant_string "Test" : i32
  // CHECK: hw.constant 1415934836 : i36
  %str1 = moore.constant_string "Test" : i36
  // CHECK: hw.constant 116 : i8
  %str2 = moore.constant_string "Test" : i8
  // CHECK: hw.constant 0 : i7
  %str_trunc = moore.constant_string "Test" : i7
  // CHECK: hw.constant 29556 : i17
  %str_trunc1 = moore.constant_string "Test" : i17
  // CHECK: hw.constant 0 : i0
  %str_empty = moore.constant_string "" : i0
  // CHECK: hw.constant 0 : i8
  %str_empty_zext = moore.constant_string "" : i8
  return
}

// CHECK-LABEL: func.func @RecurciveConditional
func.func @RecurciveConditional(%arg0 : !moore.l1, %arg1 : !moore.l1) {
  // CHECK: [[C_2:%.+]] = hw.constant -2 : i2
  // CHECK: [[C_1:%.+]] = hw.constant 1 : i2
  // CHECK: [[C_0:%.+]] = hw.constant 0 : i2
  %c_2 = moore.constant -2 : l2
  %c_1 = moore.constant 1 : l2
  %c_0 = moore.constant 0 : l2

  // CHECK: [[MUX0:%.+]] = comb.mux %arg1, [[C_0]], [[C_1]] : i2
  // CHECK: [[MUX1:%.+]] = comb.mux %arg0, [[MUX0]], [[C_2]] : i2
  %0 = moore.conditional %arg0 : l1 -> l2 {
    %1 = moore.conditional %arg1 : l1 -> l2 {
      moore.yield %c_0 : l2
    } {
      moore.yield %c_1 : l2
    }
    moore.yield %1 : l2
  } {
    moore.yield %c_2 : l2
  }

  return
}

// CHECK-LABEL: func.func @Conversions
func.func @Conversions(%arg0: !moore.i16, %arg1: !moore.l16, %arg2: !moore.l1) {
  // CHECK: [[TMP:%.+]] = comb.extract %arg0 from 0 : (i16) -> i8
  // CHECK: dbg.variable "trunc", [[TMP]]
  %0 = moore.trunc %arg0 : i16 -> i8
  dbg.variable "trunc", %0 : !moore.i8

  // CHECK: [[ZEXT:%.+]] = hw.constant 0 : i16
  // CHECK: [[TMP:%.+]] = comb.concat [[ZEXT]], %arg0 : i16, i16
  // CHECK: dbg.variable "zext", [[TMP]]
  %1 = moore.zext %arg0 : i16 -> i32
  dbg.variable "zext", %1 : !moore.i32

  // CHECK: [[SIGN:%.+]] = comb.extract %arg0 from 15 : (i16) -> i1
  // CHECK: [[SEXT:%.+]] = comb.replicate [[SIGN]] : (i1) -> i16
  // CHECK: [[TMP:%.+]] = comb.concat [[SEXT]], %arg0 : i16, i16
  // CHECK: dbg.variable "sext", [[TMP]]
  %2 = moore.sext %arg0 : i16 -> i32
  dbg.variable "sext", %2 : !moore.i32

  // CHECK: dbg.variable "i2l", %arg0 : i16
  %3 = moore.int_to_logic %arg0 : i16
  dbg.variable "i2l", %3 : !moore.l16

  // CHECK: dbg.variable "l2i", %arg1 : i16
  %4 = moore.logic_to_int %arg1 : l16
  dbg.variable "l2i", %4 : !moore.i16

  // CHECK: dbg.variable "builtin_bool", %arg2 : i1
  %5 = moore.to_builtin_bool %arg2 : l1
  dbg.variable "builtin_bool", %5 : i1

  return
}

// CHECK-LABEL: func.func @PowUOp
func.func @PowUOp(%arg0: !moore.l32, %arg1: !moore.l32) {
  // CHECK: %[[ZEROVAL:.*]] = hw.constant false
  // CHECK: %[[CONCATA:.*]] = comb.concat %[[ZEROVAL]], %arg0 : i1, i32
  // CHECK: %[[CONCATB:.*]] = comb.concat %[[ZEROVAL]], %arg1 : i1, i32
  // CHECK: %[[RES:.*]] = math.ipowi %[[CONCATA]], %[[CONCATB]] : i33
  // CHECK: comb.extract %[[RES]] from 0 : (i33) -> i32
  %0 = moore.powu %arg0, %arg1 : l32
  return
}

// CHECK-LABEL: func.func @PowSOp
func.func @PowSOp(%arg0: !moore.i32, %arg1: !moore.i32) {
  // CHECK: %[[RES:.*]] = math.ipowi %arg0, %arg1 : i32
  %0 = moore.pows %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @scfInsideProcess
moore.module @scfInsideProcess(in %in0: !moore.i32, in %in1: !moore.i32) {
  %var = moore.variable : <!moore.i32>
  // CHECK: llhd.process
  // CHECK-NOT: scf.for
  moore.procedure initial {
    %0 = moore.pows %in0, %in1 : !moore.i32
    moore.blocking_assign %var, %0 : !moore.i32
    moore.return
  }
}

// CHECK-LABEL: @blockArgAsObservedValue
moore.module @blockArgAsObservedValue(in %in0: !moore.i32, in %in1: !moore.i32) {
  %var = moore.variable : <!moore.i32>
  // CHECK: [[PRB:%.+]] = llhd.prb %var : i32
  // CHECK: llhd.process
  moore.procedure always_comb {
      %0 = moore.add %in0, %in1 : !moore.i32
      moore.blocking_assign %var, %0 : !moore.i32
      // CHECK:   llhd.wait (%in0, %in1, [[PRB]] : i32, i32, i32), ^bb1
      moore.return
  }
}

// CHECK-LABEL: @Time
// CHECK-SAME: (%arg0: !llhd.time) -> (!llhd.time, !llhd.time)
func.func @Time(%arg0: !moore.time) -> (!moore.time, !moore.time) {
  // CHECK-NEXT: [[TMP:%.+]] = llhd.constant_time <1234000fs, 0d, 0e>
  %0 = moore.constant_time 1234000 fs
  // CHECK-NEXT: return %arg0, [[TMP]] : !llhd.time, !llhd.time
  return %arg0, %0 : !moore.time, !moore.time
}

// CHECK-LABEL: @Unreachable
moore.module @Unreachable() {
  // CHECK: llhd.process
  moore.procedure initial {
    // CHECK-NEXT: llhd.halt
    moore.unreachable
  }
}

// CHECK-LABEL: @SimulationControl
func.func @SimulationControl() {
  // CHECK-NOT: moore.builtin.finish_message
  moore.builtin.finish_message false
  moore.builtin.finish_message true

  // CHECK-NEXT: sim.pause quiet
  moore.builtin.stop

  // CHECK-NEXT: sim.terminate success, quiet
  moore.builtin.finish 0
  // CHECK-NEXT: sim.terminate failure, quiet
  moore.builtin.finish 42

  return
}

// CHECK-LABEL: @SeverityToPrint
func.func @SeverityToPrint() {
  // CHECK: [[MSG:%.*]] = sim.fmt.literal "Fatal condition met!"
  // CHECK-NEXT: [[PFX:%.*]] = sim.fmt.literal "Fatal: "
  // CHECK-NEXT: [[CONCAT:%.*]] = sim.fmt.concat ([[PFX]], [[MSG]])
  // CHECK-NEXT: sim.proc.print [[CONCAT]]
  %0 = moore.fmt.literal "Fatal condition met!"
  moore.builtin.severity fatal %0

  // CHECK: [[MSG:%.*]] = sim.fmt.literal "Error condition met!"
  // CHECK-NEXT: [[PFX:%.*]] = sim.fmt.literal "Error: "
  // CHECK-NEXT: [[CONCAT:%.*]] = sim.fmt.concat ([[PFX]], [[MSG]])
  // CHECK-NEXT: sim.proc.print [[CONCAT]]
  %1 = moore.fmt.literal "Error condition met!"
  moore.builtin.severity error %1

  // CHECK: [[MSG:%.*]] = sim.fmt.literal "Warning condition met!"
  // CHECK-NEXT: [[PFX:%.*]] = sim.fmt.literal "Warning: "
  // CHECK-NEXT: [[CONCAT:%.*]] = sim.fmt.concat ([[PFX]], [[MSG]])
  // CHECK-NEXT: sim.proc.print [[CONCAT]]
  %2 = moore.fmt.literal "Warning condition met!"
  moore.builtin.severity warning %2

  return
}

// CHECK-LABEL: func.func @CHandle(%arg0: !llvm.ptr)
func.func @CHandle(%arg0: !moore.chandle) {
    return
}

// CHECK-LABEL: @MultiDimensionalSlice
moore.module @MultiDimensionalSlice(in %in : !moore.array<2 x array<2 x l2>>, out out : !moore.array<2 x l2>) {
  // CHECK-NEXT: [[IDX:%.*]] = hw.constant false
  // CHECK-NEXT: [[V:%.*]] = hw.array_get %in[[[IDX]]]
  // CHECK-NEXT: hw.output [[V]] : !hw.array<2xi2>
  %0 = moore.extract %in from 0 : array<2 x array<2 x l2>> -> array<2 x l2>
  moore.output %0 : !moore.array<2 x l2>
}

// CHECK-LABEL: hw.module @ContinuousAssignment
// CHECK-SAME: in %a : !llhd.ref<i42>
// CHECK-SAME: in %b : i42
// CHECK-SAME: in %c : !llhd.time
moore.module @ContinuousAssignment(in %a: !moore.ref<i42>, in %b: !moore.i42, in %c: !moore.time) {
  // CHECK-NEXT: [[DELTA:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: llhd.drv %a, %b after [[DELTA]]
  moore.assign %a, %b : i42
  // CHECK-NEXT: llhd.drv %a, %b after %c
  moore.delayed_assign %a, %b, %c : i42
}

// CHECK-LABEL: func.func @NonBlockingAssignment
// CHECK-SAME: %arg0: !llhd.ref<i42>
// CHECK-SAME: %arg1: i42
// CHECK-SAME: %arg2: !llhd.time
func.func @NonBlockingAssignment(%arg0: !moore.ref<i42>, %arg1: !moore.i42, %arg2: !moore.time) {
  // CHECK-NEXT: [[DELTA:%.+]] = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NEXT: llhd.drv %arg0, %arg1 after [[DELTA]]
  moore.nonblocking_assign %arg0, %arg1 : i42
  // CHECK-NEXT: llhd.drv %arg0, %arg1 after %arg2
  moore.delayed_nonblocking_assign %arg0, %arg1, %arg2 : i42
  return
}

// CHECK-LABEL: func.func @ConstantReals
func.func @ConstantReals() {
  // CHECK: arith.constant 1.234500e+00 : f32
  moore.constant_real 1.234500e+00 : f32
  // CHECK: arith.constant 1.234500e+00 : f64
  moore.constant_real 1.234500e+00 : f64
  return
}

// CHECK-LABEL: func.func @IntToRealLowering
func.func @IntToRealLowering(%arg0: !moore.i32, %arg1: !moore.i42) {
  // CHECK-NEXT: arith.sitofp {{%.*}} : i32 to f32
  // CHECK-NEXT: arith.uitofp {{%.*}} : i42 to f64
  %0 = moore.sint_to_real %arg0 : i32 -> f32
  %1 = moore.uint_to_real %arg1 : i42 -> f64
  return
}

// CHECK-LABEL: func.func @RealToIntLowering
func.func @RealToIntLowering(%arg0: !moore.f32, %arg1: !moore.f64) {
  // CHECK-NEXT: arith.fptosi %arg0 : f32 to i42
  // CHECK-NEXT: arith.fptosi %arg1 : f64 to i42
  %0 = moore.real_to_int %arg0 : f32 -> i42
  %1 = moore.real_to_int %arg1 : f64 -> i42
  return
}

// CHECK-LABEL: func.func @CurrentTime
func.func @CurrentTime() -> !moore.time {
  // CHECK-NEXT: [[TMP:%.+]] = llhd.current_time
  %0 = moore.builtin.time
  // CHECK-NEXT: return [[TMP]] : !llhd.time
  return %0 : !moore.time
}

// CHECK-LABEL: func.func @TimeConversion
func.func @TimeConversion(%arg0: !moore.l64, %arg1: !moore.time) -> (!moore.time, !moore.l64) {
  // CHECK-NEXT: [[TMP0:%.+]] = llhd.int_to_time %arg0
  %0 = moore.logic_to_time %arg0
  // CHECK-NEXT: [[TMP1:%.+]] = llhd.time_to_int %arg1
  %1 = moore.time_to_logic %arg1
  // CHECK-NEXT: return [[TMP0]], [[TMP1]]
  return %0, %1 : !moore.time, !moore.l64
}

// CHECK-LABEL: func.func @IntToStringConversion
func.func @IntToStringConversion(%arg0: !moore.i45) {
  // CHECK-NEXT: sim.string.int_to_string %arg0 : i45
  moore.int_to_string %arg0 : i45
  return
}

// CHECK-LABEL: func.func @StringOperations
// CHECK-SAME: %arg0: i32
// CHECK-SAME: %arg1: !sim.dstring
// CHECK-SAME: %arg2: !sim.dstring
func.func @StringOperations(%arg0: !moore.i32, %arg1: !moore.string, %arg2: !moore.string) {
  // CHECK: sim.string.int_to_string %arg0 : i32
  moore.int_to_string %arg0 : i32
  // CHECK: sim.string.concat ()
  moore.string.concat ()
  // CHECK: sim.string.concat (%arg1)
  moore.string.concat (%arg1)
  // CHECK: sim.string.concat (%arg1, %arg2)
  moore.string.concat (%arg1, %arg2)
  // CHECK: sim.string.length %arg1
  moore.string.len %arg1
  return
}
