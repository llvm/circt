//RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: convert_bitwise_i1
// CHECK-SAME: %[[LHS:.*]]: i1,
// CHECK-SAME: %[[RHS:.*]]: i1
func @convert_bitwise_i1(%lhs : i1, %rhs : i1) {
  // CHECK-NEXT: %{{.*}} = llvm.and %[[LHS]], %[[RHS]] : i1
  %1 = comb.and %lhs, %rhs : i1
  // CHECK-NEXT: %{{.*}} = llvm.or %[[LHS]], %[[RHS]] : i1
  %2 = comb.or %lhs, %rhs : i1
  // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[RHS]] : i1
  %3 = comb.xor %lhs, %rhs : i1

  return
}

// CHECK-LABEL: convert_bitwise_i32
// CHECK-SAME: %[[LHS:.*]]: i32,
// CHECK-SAME: %[[RHS:.*]]: i32
func @convert_bitwise_i32(%lhs : i32, %rhs : i32) {
  // CHECK-NEXT: %{{.*}} = llvm.and %[[LHS]], %[[RHS]] : i32
  comb.and %lhs, %rhs : i32
  // CHECK-NEXT: %{{.*}} = llvm.or %[[LHS]], %[[RHS]] : i32
  comb.or %lhs, %rhs : i32
  // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[RHS]] : i32
  comb.xor %lhs, %rhs : i32

  return
}

// CHECK-LABEL: convert_bitwise_i32_variadic
func @convert_bitwise_i32_variadic(%arg0 : i32, %arg1 : i32, %arg2 : i32) {
  %a = comb.and %arg0 : i32
  %b = comb.or %arg1 : i32
  %c = comb.xor %arg2 : i32

  // CHECK-NEXT: %[[AND:.*]] = llvm.and %arg0, %arg1 : i32
  // CHECK-NEXT: llvm.and %[[AND]], %arg2 : i32
  comb.and %a, %b, %c : i32
  // CHECK-NEXT: %[[OR:.*]] = llvm.or %arg0, %arg1 : i32
  // CHECK-NEXT: llvm.or %[[OR]], %arg2 : i32
  comb.or %a, %b, %c : i32
  // CHECK-NEXT: %[[XOR:.*]] = llvm.xor %arg0, %arg1 : i32
  // CHECK-NEXT: llvm.xor %[[XOR]], %arg2 : i32
  comb.xor %a, %b, %c : i32

  return
}

// CHECK-LABEL: convert_shl_i5_i2_i2
// CHECK-SAME: %[[BASE:.*]]: i5,
// CHECK-SAME: %[[HIDDEN:.*]]: i2,
// CHECK-SAME: %[[AMOUNT:.*]]: i2
func @convert_shl_i5_i2_i2(%base : i5, %hidden : i2, %amount : i2) {
  // CHECK-NEXT: %[[ZEXTB:.*]] = llvm.zext %[[BASE]] : i5 to i7
  // CHECK-NEXT: %[[ZEXTH:.*]] = llvm.zext %[[HIDDEN]] : i2 to i7
  // CHECK-NEXT: %[[ZEXTA:.*]] = llvm.zext %[[AMOUNT]] : i2 to i7
  // CHECK-NEXT: %[[HDNW:.*]] = llvm.mlir.constant(2 : i7) : i7
  // CHECK-NEXT: %[[SHB:.*]] = llvm.shl %[[ZEXTB]], %[[HDNW]] : i7
  // CHECK-NEXT: %[[COMB:.*]] = llvm.or %[[SHB]], %[[ZEXTH]] : i7
  // CHECK-NEXT: %[[SA:.*]] = llvm.sub %[[HDNW]], %[[ZEXTA]] : i7
  // CHECK-NEXT: %[[SH:.*]] = llvm.lshr %[[COMB]], %[[SA]] : i7
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SH]] : i7 to i5
  %0 = llhd.shl %base, %hidden, %amount : (i5, i2, i2) -> i5

  return
}

// CHECK-LABEL: convert_shr_i5_i2_i2
// CHECK-SAME: %[[BASE:.*]]: i5,
// CHECK-SAME: %[[HIDDEN:.*]]: i2,
// CHECK-SAME: %[[AMOUNT:.*]]: i2
func @convert_shr_i5_i2_i2(%base : i5, %hidden : i2, %amount : i2) {
  // CHECK-NEXT: %[[ZEXTB:.*]] = llvm.zext %[[BASE]] : i5 to i7
  // CHECK-NEXT: %[[ZEXTH:.*]] = llvm.zext %[[HIDDEN]] : i2 to i7
  // CHECK-NEXT: %[[ZEXTA:.*]] = llvm.zext %[[AMOUNT]] : i2 to i7
  // CHECK-NEXT: %[[BASEW:.*]] = llvm.mlir.constant(5 : i7) : i7
  // CHECK-NEXT: %[[SHH:.*]] = llvm.shl %[[ZEXTH]], %[[BASEW]] : i7
  // CHECK-NEXT: %[[COMB:.*]] = llvm.or %[[SHH]], %[[ZEXTB]] : i7
  // CHECK-NEXT: %[[SH:.*]] = llvm.lshr %[[COMB]], %[[ZEXTA]] : i7
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SH]] : i7 to i5
  %0 = llhd.shr %base, %hidden, %amount : (i5, i2, i2) -> i5

  return
}

// CHECK-LABEL:   llvm.func @convert_shr_sig(
// CHECK-SAME:                               %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                               %[[VAL_1:.*]]: !llvm.ptr<struct<()>>,
// CHECK-SAME:                               %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_12]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_13]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_18:.*]] = llvm.zext %[[VAL_5]] : i32 to i64
// CHECK:           %[[VAL_19:.*]] = llvm.add %[[VAL_11]], %[[VAL_18]] : i64
// CHECK:           %[[VAL_20:.*]] = llvm.ptrtoint %[[VAL_9]] : !llvm.ptr<i8> to i64
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK:           %[[VAL_22:.*]] = llvm.udiv %[[VAL_19]], %[[VAL_21]] : i64
// CHECK:           %[[VAL_23:.*]] = llvm.add %[[VAL_20]], %[[VAL_22]] : i64
// CHECK:           %[[VAL_24:.*]] = llvm.inttoptr %[[VAL_23]] : i64 to !llvm.ptr<i8>
// CHECK:           %[[VAL_25:.*]] = llvm.urem %[[VAL_19]], %[[VAL_21]] : i64
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_26]][0 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_25]], %[[VAL_27]][1 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_28]][2 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_30:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_29]][3 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_32:.*]] = llvm.alloca %[[VAL_31]] x !llvm.struct<(ptr<i8>, i64, i64, i64)> {alignment = 4 : i64} : (i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           llvm.store %[[VAL_30]], %[[VAL_32]] : !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @convert_shr_sig (%sI32 : !llhd.sig<i32>) -> () {
  %0 = llhd.const 8 : i32
  %1 = llhd.shr %sI32, %sI32, %0 : (!llhd.sig<i32>, !llhd.sig<i32>, i32) -> !llhd.sig<i32>
}
