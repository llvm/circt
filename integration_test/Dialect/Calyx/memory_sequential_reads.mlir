// This test checks that the generated Calyx code has and drives the "read_en" signal to the external memory.
// RUN: circt-opt %s \
// RUN:     --lower-scf-to-calyx | FileCheck %s

// CHECK: %ext_mem0_read_en: i1 {mem = {id = 0 : i32, tag = "read_en"}}
// CHECK: calyx.group
// CHECK: calyx.assign %ext_mem0_read_en = %true : i1
func.func @main(%arg0 : i32, %arg1: memref<0xi32> {calyx.sequential_reads = true}) -> i32 {
  %1 = arith.index_cast %arg0 : i32 to index
  %2 = memref.load %arg1[%1] : memref<0xi32>
  return %2 : i32
}
