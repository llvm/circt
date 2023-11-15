// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @driveSignal(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64)

// CHECK-LABEL: llvm.func @convert_sig(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
llhd.entity @convert_sig() -> () {
  // Unused in entity definition. Only used at instantiation site.
  %0 = hw.constant 0 : i1
  %1 = hw.array_create %0, %0, %0, %0 : i1

  // CHECK: llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: llvm.getelementptr %arg2[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  llhd.sig "sig0" %0 : i1
  llhd.sig "sig1" %1 : !hw.array<4xi1>
}

// CHECK-LABEL: llvm.func @convert_prb(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
llhd.entity @convert_prb(%a: !llhd.sig<i1>, %b: !llhd.sig<!hw.array<3xi5>>) -> () {
  // CHECK: [[SIGPTR_A:%.+]] = llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[SIGPTR_B:%.+]] = llvm.getelementptr %arg2[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>

  // CHECK: [[TMP:%.+]] = llvm.getelementptr [[SIGPTR_A]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[VALUEPTR_A:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> !llvm.ptr
  // CHECK: [[TMP:%.+]] = llvm.getelementptr [[SIGPTR_A]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[OFFSET_A:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> i64
  // CHECK: [[VALUE_A:%.+]] = llvm.load [[VALUEPTR_A]] : !llvm.ptr -> i16
  // CHECK: [[TMP1:%.+]] = llvm.trunc [[OFFSET_A]] : i64 to i16
  // CHECK: [[TMP2:%.+]] = llvm.lshr [[VALUE_A]], [[TMP1]]  : i16
  // CHECK: [[VALUE_A:%.+]] = llvm.trunc [[TMP2]] : i16 to i1
  %0 = llhd.prb %a : !llhd.sig<i1>

  // CHECK: [[TMP:%.+]] = llvm.getelementptr [[SIGPTR_B]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[VALUEPTR_B:%.+]] = llvm.load [[TMP]] : !llvm.ptr -> !llvm.ptr
  // CHECK: [[VALUE_B:%.+]] = llvm.load [[VALUEPTR_B]] : !llvm.ptr -> !llvm.array<3 x i5>
  %1 = llhd.prb %b : !llhd.sig<!hw.array<3xi5>>
}

// CHECK-LABEL: llvm.func @convert_drv(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
llhd.entity @convert_drv(%a: !llhd.sig<i1>, %b: !llhd.sig<!hw.array<3xi5>>) -> () {
  // CHECK: [[SIGPTR_A:%.+]] = llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[SIGPTR_B:%.+]] = llvm.getelementptr %arg2[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>

  // CHECK: [[C0_I1:%.+]] = llvm.mlir.constant(false) : i1
  // CHECK: [[C0_I5:%.+]] = llvm.mlir.constant(0 : i5) : i5
  %c0_i1 = hw.constant 0 : i1
  %c0_i5 = hw.constant 0 : i5

  // CHECK: [[ARRPTR:%.+]] = llvm.mlir.addressof {{@.+}} : !llvm.ptr
  // CHECK: [[ARR:%.+]] = llvm.load [[ARRPTR]] : !llvm.ptr -> !llvm.array<3 x i5>
  %0 = hw.array_create %c0_i5, %c0_i5, %c0_i5 : i5

  // CHECK: [[DT:%.+]] = llvm.mlir.constant(dense<[1000, 0, 0]>
  %1 = llhd.constant_time #llhd.time<1ns, 0d, 0e>

  // CHECK: [[C1_I64:%.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[BUF:%.+]] = llvm.alloca [[C1_I32]] x i1
  // CHECK: llvm.store [[C0_I1]], [[BUF]] : i1, !llvm.ptr
  // CHECK: [[DTS:%.+]] = llvm.extractvalue [[DT]][0] : !llvm.array<3 x i64>
  // CHECK: [[DTD:%.+]] = llvm.extractvalue [[DT]][1] : !llvm.array<3 x i64>
  // CHECK: [[DTE:%.+]] = llvm.extractvalue [[DT]][2] : !llvm.array<3 x i64>
  // CHECK: llvm.call @driveSignal(%arg0, [[SIGPTR_A]], [[BUF]], [[C1_I64]], [[DTS]], [[DTD]], [[DTE]])
  llhd.drv %a, %c0_i1 after %1 : !llhd.sig<i1>

  // CHECK: [[C8_I64:%.+]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK: [[TMP1:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[TMP2:%.+]] = llvm.getelementptr [[TMP1]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i5>
  // CHECK: [[ARRBYTES:%.+]] = llvm.ptrtoint [[TMP2]] : !llvm.ptr to i64
  // CHECK: [[ARRBITS:%.+]] = llvm.mul [[ARRBYTES]], [[C8_I64]]

  // CHECK: [[C1_I32:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[BUF:%.+]] = llvm.alloca [[C1_I32]] x !llvm.array<3 x i5>
  // CHECK: llvm.store [[ARR]], [[BUF]] : !llvm.array<3 x i5>, !llvm.ptr
  // CHECK: [[DTS:%.+]] = llvm.extractvalue [[DT]][0] : !llvm.array<3 x i64>
  // CHECK: [[DTD:%.+]] = llvm.extractvalue [[DT]][1] : !llvm.array<3 x i64>
  // CHECK: [[DTE:%.+]] = llvm.extractvalue [[DT]][2] : !llvm.array<3 x i64>
  // CHECK: llvm.call @driveSignal(%arg0, [[SIGPTR_B]], [[BUF]], [[ARRBITS]], [[DTS]], [[DTD]], [[DTE]])
  llhd.drv %b, %0 after %1 : !llhd.sig<!hw.array<3xi5>>
}

// CHECK-LABEL: llvm.func @convert_drv_enable(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
llhd.entity @convert_drv_enable(%a: !llhd.sig<i1>) -> () {
  // Last piece of read logic.
  // CHECK: [[VALUE_A:%.+]] = llvm.trunc {{%.+}} : i16 to i1
  %0 = llhd.prb %a : !llhd.sig<i1>
  %1 = llhd.constant_time #llhd.time<1ns, 0d, 0e>

  // CHECK: [[C1_I1:%.+]] = llvm.mlir.constant(1 {{.*}}) : i1
  // CHECK: [[ENABLE:%.+]] = llvm.icmp "eq" {{%.+}}, [[C1_I1]] : i1
  // CHECK: llvm.cond_br [[ENABLE]], ^bb1, ^bb2
  // CHECK: ^bb1:
  // CHECK: llvm.call @driveSignal
  // CHECK: llvm.br ^bb2
  // CHECK: ^bb2:
  llhd.drv %a, %0 after %1 if %0 : !llhd.sig<i1>
}

// TODO: Fix `llhd.reg` code generation and add test.
