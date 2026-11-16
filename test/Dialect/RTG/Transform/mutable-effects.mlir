// RUN: circt-opt %s --cse --canonicalize | FileCheck %s

// These tests exercise static effects only, without elaborating the invocations.
rtg.effect @opaque : (memref<1xindex>, !rtg.mut<index>) -> ()
rtg.effect @produce : () -> index

// Handlers may access DefaultResource memory as well as mutable cells.
// CHECK-LABEL: func.func @perform_barrier(
// CHECK-SAME: [[MEM:%[^ ,:]+]]: memref<1xindex>, [[CELL:%[^ ,:]+]]: !rtg.mut<index>)
func.func @perform_barrier(%mem: memref<1xindex>, %cell: !rtg.mut<index>) -> (index, index, index, index) {
  // CHECK: [[ZERO:%.+]] = index.constant 0
  %zero = index.constant 0
  // CHECK-NEXT: [[MEM_BEFORE:%.+]] = memref.load [[MEM]][[[ZERO]]] : memref<1xindex>
  %mem_before = memref.load %mem[%zero] : memref<1xindex>
  // CHECK-NEXT: [[CELL_BEFORE:%.+]] = rtg.mut_read [[CELL]] : !rtg.mut<index>
  %cell_before = rtg.mut_read %cell : !rtg.mut<index>
  // CHECK-NEXT: rtg.perform @opaque([[MEM]], [[CELL]]) : (memref<1xindex>, !rtg.mut<index>) -> none
  rtg.perform @opaque(%mem, %cell) : (memref<1xindex>, !rtg.mut<index>) -> none
  // CHECK-NEXT: [[MEM_AFTER:%.+]] = memref.load [[MEM]][[[ZERO]]] : memref<1xindex>
  %mem_after = memref.load %mem[%zero] : memref<1xindex>
  // CHECK-NEXT: [[CELL_AFTER:%.+]] = rtg.mut_read [[CELL]] : !rtg.mut<index>
  %cell_after = rtg.mut_read %cell : !rtg.mut<index>
  // CHECK-NEXT: return [[MEM_BEFORE]], [[CELL_BEFORE]], [[MEM_AFTER]], [[CELL_AFTER]] : index, index, index, index
  return %mem_before, %cell_before, %mem_after, %cell_after : index, index, index, index
}

// A resumed computation can access the same arbitrary resources.
// CHECK-LABEL: func.func @resume_barrier(
// CHECK-SAME: [[MEM:%[^ ,:]+]]: memref<1xindex>, [[CELL:%[^ ,:]+]]: !rtg.mut<index>, [[K:%[^ ,:]+]]: !rtg.continuation<none>)
func.func @resume_barrier(%mem: memref<1xindex>, %cell: !rtg.mut<index>, %k: !rtg.continuation<none>) -> (index, index, index, index) {
  // CHECK: [[ZERO:%.+]] = index.constant 0
  %zero = index.constant 0
  // CHECK-NEXT: [[MEM_BEFORE:%.+]] = memref.load [[MEM]][[[ZERO]]] : memref<1xindex>
  %mem_before = memref.load %mem[%zero] : memref<1xindex>
  // CHECK-NEXT: [[CELL_BEFORE:%.+]] = rtg.mut_read [[CELL]] : !rtg.mut<index>
  %cell_before = rtg.mut_read %cell : !rtg.mut<index>
  // CHECK-NEXT: rtg.resume [[K]] : <none>
  rtg.resume %k : !rtg.continuation<none>
  // CHECK-NEXT: [[MEM_AFTER:%.+]] = memref.load [[MEM]][[[ZERO]]] : memref<1xindex>
  %mem_after = memref.load %mem[%zero] : memref<1xindex>
  // CHECK-NEXT: [[CELL_AFTER:%.+]] = rtg.mut_read [[CELL]] : !rtg.mut<index>
  %cell_after = rtg.mut_read %cell : !rtg.mut<index>
  // CHECK-NEXT: return [[MEM_BEFORE]], [[CELL_BEFORE]], [[MEM_AFTER]], [[CELL_AFTER]] : index, index, index, index
  return %mem_before, %cell_before, %mem_after, %cell_after : index, index, index, index
}

// Invocations must not be removed or merged even when their results are unused.
// CHECK-LABEL: func.func @retain_invocations(
// CHECK-SAME: [[K:%[^ ,:]+]]: !rtg.continuation<index>, [[VALUE:%[^ ,:]+]]: index)
func.func @retain_invocations(%k: !rtg.continuation<index>, %value: index) {
  // CHECK-NEXT: {{%.+}} = rtg.perform @produce() : () -> index
  %unused0 = rtg.perform @produce() : () -> index
  // CHECK-NEXT: {{%.+}} = rtg.perform @produce() : () -> index
  %unused1 = rtg.perform @produce() : () -> index
  // CHECK-NEXT: rtg.resume [[K]], [[VALUE]] : <index>, index
  rtg.resume %k, %value : !rtg.continuation<index>, index
  // CHECK-NEXT: rtg.resume [[K]], [[VALUE]] : <index>, index
  rtg.resume %k, %value : !rtg.continuation<index>, index
  // CHECK-NEXT: return
  return
}

// Identical initial values still allocate distinct cells. Writes survive DCE
// even though neither cell is read.
// CHECK-LABEL: func.func @distinct_cells(
// CHECK-SAME: [[INIT:%[^ ,:]+]]: index)
func.func @distinct_cells(%init: index) -> (!rtg.mut<index>, !rtg.mut<index>) {
  // CHECK-NEXT: [[FIRST:%.+]] = rtg.mut_create [[INIT]] : index
  %first = rtg.mut_create %init : index
  // CHECK-NEXT: [[SECOND:%.+]] = rtg.mut_create [[INIT]] : index
  %second = rtg.mut_create %init : index
  // CHECK-NEXT: rtg.mut_write [[FIRST]], [[INIT]] : !rtg.mut<index>
  rtg.mut_write %first, %init : !rtg.mut<index>
  // CHECK-NEXT: rtg.mut_write [[SECOND]], [[INIT]] : !rtg.mut<index>
  rtg.mut_write %second, %init : !rtg.mut<index>
  // CHECK-NEXT: return [[FIRST]], [[SECOND]] : !rtg.mut<index>, !rtg.mut<index>
  return %first, %second : !rtg.mut<index>, !rtg.mut<index>
}
