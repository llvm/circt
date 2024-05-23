// RUN: circt-opt %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @convert_empty(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CHECK:         llvm.return
// CHECK:       }
hw.module @convert_empty() {}

// CHECK-LABEL: llvm.func @convert_one_input(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CHECK:         [[IN0:%.+]] = llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:         llvm.return
// CHECK:       }
hw.module @convert_one_input(inout %in0 : i1) {}

// CHECK-LABEL: llvm.func @convert_one_output(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CHECK:         [[OUT0:%.*]] = llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:         llvm.return
// CHECK:       }
hw.module @convert_one_output(inout %out0 : i1) {}

// CHECK-LABEL:   llvm.func @convert_input_and_output(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
// CHECK:           [[IN0:%.*]] = llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           [[OUT0:%.*]] = llvm.getelementptr %arg2[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
// CHECK:           llvm.return
// CHECK:         }
hw.module @convert_input_and_output(inout %in0 : i1, inout %out0 : i1) {}
