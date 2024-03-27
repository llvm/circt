// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @convertSigExtract(
// CHECK-SAME:    %arg0: i5, %arg1: !llvm.ptr)
func.func @convertSigExtract(%arg0: i5, %arg1: !llhd.sig<i32>) {
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[VALUE_PTR:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> !llvm.ptr
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[OFFSET:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[INST_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[GLOBAL_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64

  // Adjust offset
  // CHECK: [[TMP:%.+]] = llvm.zext %arg0 : i5 to i64
  // CHECK: [[NEW_OFFSET:%.+]] = llvm.add [[OFFSET]], [[TMP]]

  // Adjust value pointer to closest byte
  // CHECK: [[TMP1:%.+]] = llvm.ptrtoint [[VALUE_PTR]]
  // CHECK: [[C8_I64:%.+]] = llvm.mlir.constant(8 :
  // CHECK: [[BYTE_OFFSET:%.+]] = llvm.udiv [[NEW_OFFSET]], [[C8_I64]]
  // CHECK: [[TMP2:%.+]] = llvm.add [[TMP1]], [[BYTE_OFFSET]]
  // CHECK: [[VALUE_PTR:%.+]] = llvm.inttoptr [[TMP2]]

  // Adjust offset to closest byte
  // CHECK: [[OFFSET:%.+]] = llvm.urem [[NEW_OFFSET]], [[C8_I64]]

  // Create new signal struct on the stack.
  // CHECK: [[TMP1:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[TMP2:%.+]] = llvm.insertvalue [[VALUE_PTR]], [[TMP1]][0]
  // CHECK: [[TMP3:%.+]] = llvm.insertvalue [[OFFSET]], [[TMP2]][1]
  // CHECK: [[TMP4:%.+]] = llvm.insertvalue [[INST_IDX]], [[TMP3]][2]
  // CHECK: [[TMP5:%.+]] = llvm.insertvalue [[GLOBAL_IDX]], [[TMP4]][3]
  // CHECK: [[BUF:%.+]] = llvm.alloca {{%.+}} x !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: llvm.store [[TMP5]], [[BUF]]

  llhd.sig.extract %arg1 from %arg0 : (!llhd.sig<i32>) -> !llhd.sig<i10>
  return
}

// CHECK-LABEL: llvm.func @convertSigArrayGet(
// CHECK-SAME:    %arg0: i2, %arg1: !llvm.ptr)
func.func @convertSigArrayGet(%arg0 : i2, %arg1 : !llhd.sig<!hw.array<4xi4>>) {
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[VALUE_PTR:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> !llvm.ptr
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[OFFSET:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[INST_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[GLOBAL_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64

  // Adjust value pointer
  // CHECK: [[TMP:%.+]] = llvm.zext %arg0 : i2 to i3
  // CHECK: [[NEW_VALUE_PTR:%.+]] = llvm.getelementptr [[VALUE_PTR]][0, [[TMP]]] : (!llvm.ptr, i3) -> !llvm.ptr, !llvm.array<4 x i4>

  // Create new signal struct on the stack.
  // CHECK: [[TMP1:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[TMP2:%.+]] = llvm.insertvalue [[NEW_VALUE_PTR]], [[TMP1]][0]
  // CHECK: [[TMP3:%.+]] = llvm.insertvalue [[OFFSET]], [[TMP2]][1]
  // CHECK: [[TMP4:%.+]] = llvm.insertvalue [[INST_IDX]], [[TMP3]][2]
  // CHECK: [[TMP5:%.+]] = llvm.insertvalue [[GLOBAL_IDX]], [[TMP4]][3]
  // CHECK: [[BUF:%.+]] = llvm.alloca {{%.+}} x !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: llvm.store [[TMP5]], [[BUF]]

  llhd.sig.array_get %arg1[%arg0] : !llhd.sig<!hw.array<4xi4>>
  return
}

// CHECK-LABEL: llvm.func @convertSigArraySlice(
// CHECK-SAME:    %arg0: i2, %arg1: !llvm.ptr)
func.func @convertSigArraySlice(%arg0: i2, %arg1: !llhd.sig<!hw.array<4xi4>>) {
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[VALUE_PTR:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> !llvm.ptr
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[OFFSET:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[INST_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[GLOBAL_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64

  // Adjust value pointer
  // CHECK: [[TMP:%.+]] = llvm.zext %arg0 : i2 to i3
  // CHECK: [[NEW_VALUE_PTR:%.+]] = llvm.getelementptr [[VALUE_PTR]][0, [[TMP]]] : (!llvm.ptr, i3) -> !llvm.ptr, !llvm.array<4 x i4>

  // Create new signal struct on the stack.
  // CHECK: [[TMP1:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[TMP2:%.+]] = llvm.insertvalue [[NEW_VALUE_PTR]], [[TMP1]][0]
  // CHECK: [[TMP3:%.+]] = llvm.insertvalue [[OFFSET]], [[TMP2]][1]
  // CHECK: [[TMP4:%.+]] = llvm.insertvalue [[INST_IDX]], [[TMP3]][2]
  // CHECK: [[TMP5:%.+]] = llvm.insertvalue [[GLOBAL_IDX]], [[TMP4]][3]
  // CHECK: [[BUF:%.+]] = llvm.alloca {{%.+}} x !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: llvm.store [[TMP5]], [[BUF]]

  llhd.sig.array_slice %arg1 at %arg0 : (!llhd.sig<!hw.array<4xi4>>) -> !llhd.sig<!hw.array<2xi4>>
  return
}

// CHECK-LABEL: llvm.func @convertSigStructExtract(
// CHECK-SAME:    %arg0: !llvm.ptr)
func.func @convertSigStructExtract(%arg0: !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[VALUE_PTR:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> !llvm.ptr
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[OFFSET:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[INST_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[GLOBAL_IDX:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64

  // Adjust value pointer
  // CHECK: [[NEW_VALUE_PTR:%.+]] = llvm.getelementptr [[VALUE_PTR]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i3, i2, i1)>

  // Create new signal struct on the stack.
  // CHECK: [[TMP1:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[TMP2:%.+]] = llvm.insertvalue [[NEW_VALUE_PTR]], [[TMP1]][0]
  // CHECK: [[TMP3:%.+]] = llvm.insertvalue [[OFFSET]], [[TMP2]][1]
  // CHECK: [[TMP4:%.+]] = llvm.insertvalue [[INST_IDX]], [[TMP3]][2]
  // CHECK: [[TMP5:%.+]] = llvm.insertvalue [[GLOBAL_IDX]], [[TMP4]][3]
  // CHECK: [[BUF:%.+]] = llvm.alloca {{%.+}} x !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: llvm.store [[TMP5]], [[BUF]]

  llhd.sig.struct_extract %arg0["bar"] : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>
  return
}
