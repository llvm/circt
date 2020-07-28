//RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: @convert_const
llvm.func @convert_const() {
  // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(true) : !llvm.i1
  %0 = llhd.const 1 : i1

  // CHECK-NEXT %{{.*}} = llvm.mlir.constant(0 : i32) : !llvm.i32
  %1 = llhd.const 0 : i32

  // this gets erased
  %2 = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time

  // CHECK-NEXT %{{.*}} = llvm.mlir.constant(123 : i64) : !llvm.i64
  %3 = llhd.const 123 : i64

  llvm.return
}

// CHECK-LABEL: @convert_extract_slice_int
// CHECK-SAME: %[[CI32:.*]]: !llvm.i32
// CHECK-SAME: %[[CI100:.*]]: !llvm.i100
func @convert_extract_slice_int(%cI32 : i32, %cI100 : i100) {
  // CHECK-NEXT: %[[CIND0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK-NEXT: %[[ADJUST0:.*]] = llvm.trunc %[[CIND0]] : !llvm.i64 to !llvm.i32
  // CHECK-NEXT: %[[SHR0:.*]] = llvm.lshr %[[CI32]], %[[ADJUST0]] : !llvm.i32
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR0]] : !llvm.i32 to !llvm.i10
  %0 = llhd.extract_slice %cI32, 0 : i32 -> i10
  // CHECK-NEXT: %[[CIND2:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK-NEXT: %[[ADJUST1:.*]] = llvm.zext %[[CIND2]] : !llvm.i64 to !llvm.i100
  // CHECK-NEXT: %[[SHR1:.*]] = llvm.lshr %[[CI100]], %[[ADJUST1]] : !llvm.i100
  // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR1]] : !llvm.i100 to !llvm.i10
  %2 = llhd.extract_slice %cI100, 0 : i100 -> i10

  return
}

// CHECK-LABEL:   llvm.func @convert_extract_slice_sig(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: !llvm.ptr<struct<()>>,
// CHECK-SAME:                                         %[[VAL_2:.*]]: !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_12]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_13]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_18:.*]] = llvm.add %[[VAL_11]], %[[VAL_5]] : !llvm.i64
// CHECK:           %[[VAL_19:.*]] = llvm.ptrtoint %[[VAL_9]] : !llvm.ptr<i8> to !llvm.i64
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK:           %[[VAL_21:.*]] = llvm.udiv %[[VAL_18]], %[[VAL_20]] : !llvm.i64
// CHECK:           %[[VAL_22:.*]] = llvm.add %[[VAL_19]], %[[VAL_21]] : !llvm.i64
// CHECK:           %[[VAL_23:.*]] = llvm.inttoptr %[[VAL_22]] : !llvm.i64 to !llvm.ptr<i8>
// CHECK:           %[[VAL_24:.*]] = llvm.urem %[[VAL_18]], %[[VAL_20]] : !llvm.i64
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_25]][0 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_26]][1 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_27]][2 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_28]][3 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_31:.*]] = llvm.alloca %[[VAL_30]] x !llvm.struct<(ptr<i8>, i64, i64, i64)> {alignment = 4 : i64} : (!llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           llvm.store %[[VAL_29]], %[[VAL_31]] : !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @convert_extract_slice_sig (%sI32 : !llhd.sig<i32>) -> () {
  %0 = llhd.extract_slice %sI32, 0 : !llhd.sig<i32> -> !llhd.sig<i10>
}

// CHECK-LABEL:   llvm.func @convert_insert_slice_int(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !llvm.i1,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !llvm.i10) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:           %[[VAL_3:.*]] = llvm.trunc %[[VAL_2]] : !llvm.i64 to !llvm.i1
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(true) : !llvm.i1
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(false) : !llvm.i1
// CHECK:           %[[VAL_6:.*]] = llvm.zext %[[VAL_0]] : !llvm.i1 to !llvm.i1
// CHECK:           %[[VAL_7:.*]] = llvm.shl %[[VAL_6]], %[[VAL_3]] : !llvm.i1
// CHECK:           %[[VAL_8:.*]] = llvm.or %[[VAL_7]], %[[VAL_5]] : !llvm.i1
// CHECK:           %[[VAL_9:.*]] = llvm.or %[[VAL_0]], %[[VAL_4]] : !llvm.i1
// CHECK:           %[[VAL_10:.*]] = llvm.and %[[VAL_9]], %[[VAL_8]] : !llvm.i1
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(5 : index) : !llvm.i64
// CHECK:           %[[VAL_12:.*]] = llvm.trunc %[[VAL_11]] : !llvm.i64 to !llvm.i10
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(32 : i10) : !llvm.i10
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(-33 : i10) : !llvm.i10
// CHECK:           %[[VAL_15:.*]] = llvm.zext %[[VAL_0]] : !llvm.i1 to !llvm.i10
// CHECK:           %[[VAL_16:.*]] = llvm.shl %[[VAL_15]], %[[VAL_12]] : !llvm.i10
// CHECK:           %[[VAL_17:.*]] = llvm.or %[[VAL_16]], %[[VAL_14]] : !llvm.i10
// CHECK:           %[[VAL_18:.*]] = llvm.or %[[VAL_1]], %[[VAL_13]] : !llvm.i10
// CHECK:           %[[VAL_19:.*]] = llvm.and %[[VAL_18]], %[[VAL_17]] : !llvm.i10
// CHECK:           llvm.return
// CHECK:         }
func @convert_insert_slice_int (%i1 : i1, %i10 : i10) -> () {
  %0 = llhd.insert_slice %i1, %i1, 0 : i1, i1
  %1 = llhd.insert_slice %i10, %i1, 5 : i10, i1

  return
}