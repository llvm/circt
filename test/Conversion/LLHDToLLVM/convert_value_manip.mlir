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
// CHECK-SAME:                                         %[[VAL_1:.*]]: !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_2]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_5]], %[[VAL_5]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK:           %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_5]], %[[VAL_11]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_5]], %[[VAL_12]]] : (!llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_17:.*]] = llvm.add %[[VAL_10]], %[[VAL_4]] : !llvm.i64
// CHECK:           %[[VAL_18:.*]] = llvm.ptrtoint %[[VAL_8]] : !llvm.ptr<i8> to !llvm.i64
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK:           %[[VAL_20:.*]] = llvm.udiv %[[VAL_17]], %[[VAL_19]] : !llvm.i64
// CHECK:           %[[VAL_21:.*]] = llvm.add %[[VAL_18]], %[[VAL_20]] : !llvm.i64
// CHECK:           %[[VAL_22:.*]] = llvm.inttoptr %[[VAL_21]] : !llvm.i64 to !llvm.ptr<i8>
// CHECK:           %[[VAL_23:.*]] = llvm.urem %[[VAL_17]], %[[VAL_19]] : !llvm.i64
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_22]], %[[VAL_24]][0 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_25]][1 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_26]][2 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_28:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_27]][3 : i32] : !llvm.struct<(ptr<i8>, i64, i64, i64)>
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_30:.*]] = llvm.alloca %[[VAL_29]] x !llvm.struct<(ptr<i8>, i64, i64, i64)> {alignment = 4 : i64} : (!llvm.i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           llvm.store %[[VAL_28]], %[[VAL_30]] : !llvm.ptr<struct<(ptr<i8>, i64, i64, i64)>>
// CHECK:           llvm.return
// CHECK:         }
llhd.entity @convert_extract_slice_sig (%sI32 : !llhd.sig<i32>) -> () {
  %0 = llhd.extract_slice %sI32, 0 : !llhd.sig<i32> -> !llhd.sig<i10>
}
