// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @driveSignal(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64)

// CHECK-LABEL: llvm.func @Foo(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
llhd.entity @Foo () -> () {
  // Unused in entity definition. Only used at instantiation site.
  // CHECK: [[C0:%.+]] = llvm.mlir.constant(false) : i1
  %0 = hw.constant 0 : i1

  // CHECK: [[SIG_PTR:%.+]] = llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  %toggle = llhd.sig "toggle" %0 : i1

  // CHECK: [[TMP:%.+]] = llvm.getelementptr [[SIG_PTR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[SIG_VALUE_PTR:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> !llvm.ptr
  // CHECK: [[TMP:%.+]] = llvm.getelementptr [[SIG_PTR]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[SIG_OFFSET:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[SIG_VALUE:%.+]] = llvm.load [[SIG_VALUE_PTR]] : !llvm.ptr -> i16
  // CHECK: [[TMP:%.+]] = llvm.trunc [[SIG_OFFSET]] : i64 to i16
  // CHECK: [[SIG_VALUE_SHIFTED:%.+]] = llvm.lshr [[SIG_VALUE]], [[TMP]]  : i16
  // CHECK: [[SIG_VALUE:%.+]] = llvm.trunc [[SIG_VALUE_SHIFTED]] : i16 to i1
  %1 = llhd.prb %toggle : !llhd.sig<i1>

  // CHECK: [[DRV_VALUE:%.+]] = llvm.xor
  %allset = hw.constant 1 : i1
  %2 = comb.xor %1, %allset : i1

  // CHECK: [[DT:%.+]] = llvm.mlir.constant(dense<[1000, 0, 0]>
  %dt = llhd.constant_time #llhd.time<1ns, 0d, 0e>

  // CHECK: [[C1_I64:%.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[BUF:%.+]] = llvm.alloca [[C1_I32]] x i1
  // CHECK: llvm.store [[DRV_VALUE]], [[BUF]] : i1, !llvm.ptr
  // CHECK: [[DT_S:%.+]] = llvm.extractvalue [[DT]][0] : !llvm.array<3 x i64>
  // CHECK: [[DT_D:%.+]] = llvm.extractvalue [[DT]][1] : !llvm.array<3 x i64>
  // CHECK: [[DT_E:%.+]] = llvm.extractvalue [[DT]][2] : !llvm.array<3 x i64>
  // CHECK: llvm.call @driveSignal(%arg0, [[SIG_PTR]], [[BUF]], [[C1_I64]], [[DT_S]], [[DT_D]], [[DT_E]])
  llhd.drv %toggle, %2 after %dt : !llhd.sig<i1>
}

// CHECK-LABEL: @convertConstantTime
llvm.func @convertConstantTime() {
  // CHECK-NEXT: {{%.+}} = llvm.mlir.constant(dense<[0, 1, 2]> : tensor<3xi64>) : !llvm.array<3 x i64>
  %2 = llhd.constant_time #llhd.time<0ns, 1d, 2e>
  // CHECK-NEXT: llvm.return
  llvm.return
}
