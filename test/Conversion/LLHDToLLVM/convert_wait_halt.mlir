// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @llhdSuspend(!llvm.ptr, !llvm.ptr, i64, i64, i64)

// CHECK-LABEL: llvm.func @convert_wait(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
llhd.proc @convert_wait(%a: !llhd.sig<i1>, %b: !llhd.sig<i1>) -> () {
  // CHECK: [[SIGPTR_A:%.+]] = llvm.getelementptr %arg2[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>
  // CHECK: [[SIGPTR_B:%.+]] = llvm.getelementptr %arg2[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i64, i64)>

  // CHECK: [[RESUME_PTR:%.+]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, i32
  // CHECK: [[RESUME:%.+]] = llvm.load [[RESUME_PTR]] : !llvm.ptr -> i32
  // CHECK: llvm.br [[BB:\^.+]]
  // CHECK: [[BB]]:

  // Resume 1 (after wait)
  // CHECK: [[C1_I32:%.+]] = llvm.mlir.constant(1 :
  // CHECK: [[EQ:%.+]] = llvm.icmp "eq" [[RESUME]], [[C1_I32]]
  // CHECK: llvm.cond_br [[EQ]], [[BB_END:\^.+]], [[BB:\^.+]]
  // CHECK: [[BB]]:

  // Resume 0 (entry point)
  // CHECK: [[C0_I32:%.+]] = llvm.mlir.constant(0 :
  // CHECK: [[EQ:%.+]] = llvm.icmp "eq" [[RESUME]], [[C0_I32]]
  // CHECK: llvm.cond_br [[EQ]], [[BB_ENTRY:\^.+]], {{\^.+}}

  // CHECK: [[BB_ENTRY]]:
  %0 = llhd.constant_time #llhd.time<1ns, 0d, 0e>

  // Update resume index to 1 (after wait)
  // CHECK: [[C1_I32:%.+]] = llvm.mlir.constant(1 :
  // CHECK: [[RESUME_PTR:%.+]] = llvm.getelementptr %arg1[1]
  // CHECK: llvm.store [[C1_I32]], [[RESUME_PTR]]

  // Clear sensitivity flags for all signals.
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[2]
  // CHECK: [[SENSE_PTR:%.+]] = llvm.load [[TMP]]
  // CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false)
  // CHECK: [[SENSE_A:%.+]] = llvm.getelementptr [[SENSE_PTR]][0]
  // CHECK: llvm.store [[FALSE]], [[SENSE_A]] : i1, !llvm.ptr
  // CHECK: [[SENSE_B:%.+]] = llvm.getelementptr [[SENSE_PTR]][1]
  // CHECK: llvm.store [[FALSE]], [[SENSE_B]] : i1, !llvm.ptr

  // Set sensitivity flag for signal "b" (index 1).
  // CHECK: [[SIGIDX_PTR:%.+]] = llvm.getelementptr [[SIGPTR_B]][2]
  // CHECK: [[SIGIDX:%.+]] = llvm.load [[SIGIDX_PTR]] : !llvm.ptr -> i64
  // CHECK: [[TRUE:%.+]] = llvm.mlir.constant(true)
  // CHECK: [[SENSE:%.+]] = llvm.getelementptr [[SENSE_PTR]][[[SIGIDX]]]
  // CHECK: llvm.store [[TRUE]], [[SENSE]] : i1, !llvm.ptr

  // CHECK: llvm.call @llhdSuspend(%arg0, %arg1, {{%.+}}, {{%.+}}, {{%.+}})
  llhd.wait for %0, (%b : !llhd.sig<i1>), ^end

  // CHECK: [[BB_END]]:
  // CHECK: llvm.br [[BB_END]]
^end:
  cf.br ^end
}

// CHECK-LABEL: llvm.func @convert_halt(
// CHECK-SAME:    %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
llhd.proc @convert_halt() -> (%a: !llhd.sig<i1>, %b: !llhd.sig<i1>) {
  // Clear sensitivity flags for all signals.
  // CHECK: [[TMP:%.+]] = llvm.getelementptr %arg1[2]
  // CHECK: [[SENSE_PTR:%.+]] = llvm.load [[TMP]]
  // CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false)
  // CHECK: [[SENSE_A:%.+]] = llvm.getelementptr [[SENSE_PTR]][0]
  // CHECK: llvm.store [[FALSE]], [[SENSE_A]] : i1, !llvm.ptr
  // CHECK: [[SENSE_B:%.+]] = llvm.getelementptr [[SENSE_PTR]][1]
  // CHECK: llvm.store [[FALSE]], [[SENSE_B]] : i1, !llvm.ptr
  llhd.halt
}
