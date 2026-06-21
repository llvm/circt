// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @UArrayCmpClass {
}

// CHECK-LABEL: func.func @UnpackedArrayCmp
// CHECK-SAME: (%arg0: !hw.array<2xi3>, %arg1: !hw.array<2xi3>) -> (i1, i1)
func.func @UnpackedArrayCmp(%lhs: !moore.uarray<2 x i3>, %rhs: !moore.uarray<2 x i3>) -> (!moore.i1, !moore.i1) {
  // CHECK: [[LHS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xi3>) -> i6
  // CHECK: [[RHS:%.+]] = hw.bitcast %arg1 : (!hw.array<2xi3>) -> i6
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[LHS]], [[RHS]] : i6
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x i3> -> !moore.i1
  // CHECK: [[LHS2:%.+]] = hw.bitcast %arg0 : (!hw.array<2xi3>) -> i6
  // CHECK: [[RHS2:%.+]] = hw.bitcast %arg1 : (!hw.array<2xi3>) -> i6
  // CHECK: [[NE:%.+]] = comb.icmp ne [[LHS2]], [[RHS2]] : i6
  %ne = moore.uarray_cmp ne %lhs, %rhs : !moore.uarray<2 x i3> -> !moore.i1
  // CHECK: return [[EQ]], [[NE]] : i1, i1
  return %eq, %ne : !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @UnpackedArrayOfChandleCmp
// CHECK-SAME: (%arg0: !hw.array<2x!llvm.ptr>, %arg1: !hw.array<2x!llvm.ptr>) -> (i1, i1)
func.func @UnpackedArrayOfChandleCmp(%lhs: !moore.uarray<2 x chandle>, %rhs: !moore.uarray<2 x chandle>) -> (!moore.i1, !moore.i1) {
  // CHECK: [[LHS0:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[RHS0:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[EQ0:%.+]] = llvm.icmp "eq" [[LHS0]], [[RHS0]] : !llvm.ptr
  // CHECK: [[LHS1:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[RHS1:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[EQ1:%.+]] = llvm.icmp "eq" [[LHS1]], [[RHS1]] : !llvm.ptr
  // CHECK: [[EQ:%.+]] = comb.and bin [[EQ0]], [[EQ1]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x chandle> -> !moore.i1
  // CHECK: [[NE_EQ0:%.+]] = llvm.icmp "eq" {{%.+}}, {{%.+}} : !llvm.ptr
  // CHECK: [[NE_EQ1:%.+]] = llvm.icmp "eq" {{%.+}}, {{%.+}} : !llvm.ptr
  // CHECK: [[NE_EQUAL:%.+]] = comb.and bin [[NE_EQ0]], [[NE_EQ1]] : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: [[NE:%.+]] = comb.xor bin [[NE_EQUAL]], [[TRUE]] : i1
  %ne = moore.uarray_cmp ne %lhs, %rhs : !moore.uarray<2 x chandle> -> !moore.i1
  // CHECK: return [[EQ]], [[NE]] : i1, i1
  return %eq, %ne : !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @UnpackedArrayOfClassCmp
// CHECK-SAME: (%arg0: !hw.array<2x!llvm.ptr>, %arg1: !hw.array<2x!llvm.ptr>) -> (i1, i1)
func.func @UnpackedArrayOfClassCmp(%lhs: !moore.uarray<2 x class<@UArrayCmpClass>>, %rhs: !moore.uarray<2 x class<@UArrayCmpClass>>) -> (!moore.i1, !moore.i1) {
  // CHECK: [[LHS0:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[RHS0:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[EQ0:%.+]] = llvm.icmp "eq" [[LHS0]], [[RHS0]] : !llvm.ptr
  // CHECK: [[LHS1:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[RHS1:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2x!llvm.ptr>, i1
  // CHECK: [[EQ1:%.+]] = llvm.icmp "eq" [[LHS1]], [[RHS1]] : !llvm.ptr
  // CHECK: [[EQ:%.+]] = comb.and bin [[EQ0]], [[EQ1]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x class<@UArrayCmpClass>> -> !moore.i1
  // CHECK: [[NE_EQ0:%.+]] = llvm.icmp "eq" {{%.+}}, {{%.+}} : !llvm.ptr
  // CHECK: [[NE_EQ1:%.+]] = llvm.icmp "eq" {{%.+}}, {{%.+}} : !llvm.ptr
  // CHECK: [[NE_EQUAL:%.+]] = comb.and bin [[NE_EQ0]], [[NE_EQ1]] : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: [[NE:%.+]] = comb.xor bin [[NE_EQUAL]], [[TRUE]] : i1
  %ne = moore.uarray_cmp ne %lhs, %rhs : !moore.uarray<2 x class<@UArrayCmpClass>> -> !moore.i1
  // CHECK: return [[EQ]], [[NE]] : i1, i1
  return %eq, %ne : !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @UnpackedArrayOfRealCmp
// CHECK-SAME: (%arg0: !hw.array<2xf64>, %arg1: !hw.array<2xf64>) -> (i1, i1)
func.func @UnpackedArrayOfRealCmp(%lhs: !moore.uarray<2 x f64>, %rhs: !moore.uarray<2 x f64>) -> (!moore.i1, !moore.i1) {
  // CHECK: [[LHS0:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2xf64>, i1
  // CHECK: [[RHS0:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2xf64>, i1
  // CHECK: [[EQ0:%.+]] = arith.cmpf oeq, [[LHS0]], [[RHS0]] : f64
  // CHECK: [[LHS1:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2xf64>, i1
  // CHECK: [[RHS1:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2xf64>, i1
  // CHECK: [[EQ1:%.+]] = arith.cmpf oeq, [[LHS1]], [[RHS1]] : f64
  // CHECK: [[EQ:%.+]] = comb.and bin [[EQ0]], [[EQ1]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x f64> -> !moore.i1
  // CHECK: [[NE_EQ0:%.+]] = arith.cmpf oeq, {{%.+}}, {{%.+}} : f64
  // CHECK: [[NE_EQ1:%.+]] = arith.cmpf oeq, {{%.+}}, {{%.+}} : f64
  // CHECK: [[NE_EQUAL:%.+]] = comb.and bin [[NE_EQ0]], [[NE_EQ1]] : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: [[NE:%.+]] = comb.xor bin [[NE_EQUAL]], [[TRUE]] : i1
  %ne = moore.uarray_cmp ne %lhs, %rhs : !moore.uarray<2 x f64> -> !moore.i1
  // CHECK: return [[EQ]], [[NE]] : i1, i1
  return %eq, %ne : !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @UnpackedArrayOfTimeCmp
// CHECK-SAME: (%arg0: !hw.array<2x!llhd.time>, %arg1: !hw.array<2x!llhd.time>) -> (i1, i1)
func.func @UnpackedArrayOfTimeCmp(%lhs: !moore.uarray<2 x time>, %rhs: !moore.uarray<2 x time>) -> (!moore.i1, !moore.i1) {
  // CHECK: [[LHS0:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2x!llhd.time>, i1
  // CHECK: [[RHS0:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2x!llhd.time>, i1
  // CHECK: [[LHS0_INT:%.+]] = llhd.time_to_int [[LHS0]]
  // CHECK: [[RHS0_INT:%.+]] = llhd.time_to_int [[RHS0]]
  // CHECK: [[EQ0:%.+]] = comb.icmp eq [[LHS0_INT]], [[RHS0_INT]] : i64
  // CHECK: [[LHS1:%.+]] = hw.array_get %arg0[{{%.+}}] : !hw.array<2x!llhd.time>, i1
  // CHECK: [[RHS1:%.+]] = hw.array_get %arg1[{{%.+}}] : !hw.array<2x!llhd.time>, i1
  // CHECK: [[LHS1_INT:%.+]] = llhd.time_to_int [[LHS1]]
  // CHECK: [[RHS1_INT:%.+]] = llhd.time_to_int [[RHS1]]
  // CHECK: [[EQ1:%.+]] = comb.icmp eq [[LHS1_INT]], [[RHS1_INT]] : i64
  // CHECK: [[EQ:%.+]] = comb.and bin [[EQ0]], [[EQ1]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x time> -> !moore.i1
  // CHECK: [[NE_EQ0:%.+]] = comb.icmp eq {{%.+}}, {{%.+}} : i64
  // CHECK: [[NE_EQ1:%.+]] = comb.icmp eq {{%.+}}, {{%.+}} : i64
  // CHECK: [[NE_EQUAL:%.+]] = comb.and bin [[NE_EQ0]], [[NE_EQ1]] : i1
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: [[NE:%.+]] = comb.xor bin [[NE_EQUAL]], [[TRUE]] : i1
  %ne = moore.uarray_cmp ne %lhs, %rhs : !moore.uarray<2 x time> -> !moore.i1
  // CHECK: return [[EQ]], [[NE]] : i1, i1
  return %eq, %ne : !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @UnpackedArrayOfUnpackedStructWithChandleCmp
// CHECK-SAME: (%arg0: !hw.array<2xstruct<h: !llvm.ptr, bits: i3>>, %arg1: !hw.array<2xstruct<h: !llvm.ptr, bits: i3>>) -> i1
func.func @UnpackedArrayOfUnpackedStructWithChandleCmp(%lhs: !moore.uarray<2 x ustruct<{h: chandle, bits: i3}>>, %rhs: !moore.uarray<2 x ustruct<{h: chandle, bits: i3}>>) -> !moore.i1 {
  // CHECK: hw.struct_extract {{%.+}}["h"] : !hw.struct<h: !llvm.ptr, bits: i3>
  // CHECK: [[HANDLE_EQ:%.+]] = llvm.icmp "eq" {{%.+}}, {{%.+}} : !llvm.ptr
  // CHECK: hw.struct_extract {{%.+}}["bits"] : !hw.struct<h: !llvm.ptr, bits: i3>
  // CHECK: [[BITS_EQ:%.+]] = comb.icmp eq {{%.+}}, {{%.+}} : i3
  // CHECK: comb.and bin [[HANDLE_EQ]], [[BITS_EQ]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x ustruct<{h: chandle, bits: i3}>> -> !moore.i1
  return %eq : !moore.i1
}

// CHECK-LABEL: func.func @UnpackedArrayOfStructCmp
// CHECK-SAME: (%arg0: !hw.array<2xstruct<a: i3, b: i2>>, %arg1: !hw.array<2xstruct<a: i3, b: i2>>) -> i1
func.func @UnpackedArrayOfStructCmp(%lhs: !moore.uarray<2 x struct<{a: i3, b: i2}>>, %rhs: !moore.uarray<2 x struct<{a: i3, b: i2}>>) -> !moore.i1 {
  // CHECK: [[LHS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xstruct<a: i3, b: i2>>) -> i10
  // CHECK: [[RHS:%.+]] = hw.bitcast %arg1 : (!hw.array<2xstruct<a: i3, b: i2>>) -> i10
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[LHS]], [[RHS]] : i10
  // CHECK: return [[EQ]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x struct<{a: i3, b: i2}>> -> !moore.i1
  return %eq : !moore.i1
}

// CHECK-LABEL: func.func @UnpackedArrayOfUnpackedStructCmp
// CHECK-SAME: (%arg0: !hw.array<2xstruct<a: i3, b: i2>>, %arg1: !hw.array<2xstruct<a: i3, b: i2>>) -> i1
func.func @UnpackedArrayOfUnpackedStructCmp(%lhs: !moore.uarray<2 x ustruct<{a: i3, b: i2}>>, %rhs: !moore.uarray<2 x ustruct<{a: i3, b: i2}>>) -> !moore.i1 {
  // CHECK: [[LHS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xstruct<a: i3, b: i2>>) -> i10
  // CHECK: [[RHS:%.+]] = hw.bitcast %arg1 : (!hw.array<2xstruct<a: i3, b: i2>>) -> i10
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[LHS]], [[RHS]] : i10
  // CHECK: return [[EQ]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x ustruct<{a: i3, b: i2}>> -> !moore.i1
  return %eq : !moore.i1
}

// CHECK-LABEL: func.func @NestedUnpackedArrayCmp
// CHECK-SAME: (%arg0: !hw.array<2xarray<2xi3>>, %arg1: !hw.array<2xarray<2xi3>>) -> i1
func.func @NestedUnpackedArrayCmp(%lhs: !moore.uarray<2 x uarray<2 x i3>>, %rhs: !moore.uarray<2 x uarray<2 x i3>>) -> !moore.i1 {
  // CHECK: [[LHS:%.+]] = hw.bitcast %arg0 : (!hw.array<2xarray<2xi3>>) -> i12
  // CHECK: [[RHS:%.+]] = hw.bitcast %arg1 : (!hw.array<2xarray<2xi3>>) -> i12
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[LHS]], [[RHS]] : i12
  // CHECK: return [[EQ]] : i1
  %eq = moore.uarray_cmp eq %lhs, %rhs : !moore.uarray<2 x uarray<2 x i3>> -> !moore.i1
  return %eq : !moore.i1
}
