// RUN: circt-opt %s --lower-scf-to-calyx -canonicalize -split-input-file | FileCheck %s

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
// CHECK-DAG:      %c64_i32 = hw.constant 64 : i32
// CHECK-DAG:      %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:      %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i6
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
// CHECK-DAG:      %mem_1.addr0, %mem_1.write_data, %mem_1.write_en, %mem_1.clk, %mem_1.read_data, %mem_1.done = calyx.memory @mem_1 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i32, i1
// CHECK-DAG:      %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.clk, %mem_0.read_data, %mem_0.done = calyx.memory @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i32, i1
// CHECK-DAG:      %while_0_arg0_reg.in, %while_0_arg0_reg.write_en, %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %while_0_arg0_reg.out, %while_0_arg0_reg.done = calyx.register @while_0_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.group @assign_while_0_init_0  {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.comb_group @bb0_0  {
// CHECK-NEXT:         calyx.assign %std_lt_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_lt_0.right = %c64_i32 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb0_2  {
// CHECK-NEXT:         calyx.assign %std_slice_1.in = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %mem_1.addr0 = %std_slice_1.out : i6
// CHECK-NEXT:         calyx.assign %mem_1.write_data = %mem_0.read_data : i32
// CHECK-NEXT:         calyx.assign %mem_1.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:         calyx.group_done %mem_1.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_latch  {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @assign_while_0_init_0
// CHECK-NEXT:         calyx.while %std_lt_0.out with @bb0_0  {
// CHECK-NEXT:           calyx.seq  {
// CHECK-NEXT:             calyx.enable @bb0_2
// CHECK-NEXT:             calyx.enable @assign_while_0_latch
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>
    scf.while(%arg0 = %c0) : (index) -> (index) {
      %cond = arith.cmpi ult, %arg0, %c64 : index
      scf.condition(%cond) %arg0 : index
    } do {
    ^bb0(%arg1: index):
      %v = memref.load %0[%arg1] : memref<64xi32>
      memref.store %v, %1[%arg1] : memref<64xi32>
      %inc = arith.addi %arg1, %c1 : index
      scf.yield %inc : index
    }
    return
  }
}

// -----

// Test combinational value used across sequential group boundary. This requires
// that any referenced combinational assignments are re-applied in each
// sequential group.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:      %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:      %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.clk, %mem_0.read_data, %mem_0.done = calyx.memory @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @bb0_1  {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:         calyx.assign %mem_0.write_data = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %mem_0.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.group_done %mem_0.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %std_add_1.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_1.left = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.assign %std_add_1.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @bb0_1
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%arg0 : i32) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %1 = arith.addi %arg0, %c1 : i32
    memref.store %1, %0[%c0] : memref<64xi32>
    %3 = arith.addi %1, %c1 : i32
    return %3 : i32
  }
}

// -----

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:      %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:      %std_add_2.left, %std_add_2.right, %std_add_2.out = calyx.std_add @std_add_2 : i32, i32, i32
// CHECK-DAG:      %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.clk, %mem_0.read_data, %mem_0.done = calyx.memory @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @bb0_2  {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:         calyx.assign %mem_0.write_data = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %mem_0.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.group_done %mem_0.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %std_add_2.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_2.left = %std_add_1.out : i32
// CHECK-NEXT:         calyx.assign %std_add_1.left = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.assign %std_add_1.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.assign %std_add_2.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @bb0_2
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%arg0 : i32) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %1 = arith.addi %arg0, %c1 : i32
    %2 = arith.addi %1, %c1 : i32
    memref.store %1, %0[%c0] : memref<64xi32>
    %3 = arith.addi %2, %c1 : i32
    return %3 : i32
  }
}

// -----
// Test multiple reads from the same memory (structural hazard).

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i6, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i6
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %load_1_reg.in, %load_1_reg.write_en, %load_1_reg.clk, %load_1_reg.reset, %load_1_reg.out, %load_1_reg.done = calyx.register @load_1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %load_0_reg.in, %load_0_reg.write_en, %load_0_reg.clk, %load_0_reg.reset, %load_0_reg.out, %load_0_reg.done = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %std_pad_0.in, %std_pad_0.out = calyx.std_pad @std_pad_0 : i6, i32
// CHECK-DAG:      %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.clk, %mem_0.read_data, %mem_0.done = calyx.memory @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @bb0_1  {
// CHECK-NEXT:         calyx.assign %std_slice_1.in = %std_pad_0.out : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_1.out : i6
// CHECK-NEXT:         calyx.assign %load_0_reg.in = %mem_0.read_data : i32
// CHECK-NEXT:         calyx.assign %load_0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_pad_0.in = %in0 : i6
// CHECK-NEXT:         calyx.group_done %load_0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb0_2  {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %c1_i32 : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:         calyx.assign %load_1_reg.in = %mem_0.read_data : i32
// CHECK-NEXT:         calyx.assign %load_1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %load_1_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_0.left = %load_0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %load_1_reg.out : i32
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @bb0_1
// CHECK-NEXT:         calyx.enable @bb0_2
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%arg0 : i6) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %c1 = arith.constant 1 : index
    %arg0_idx =  arith.index_cast %arg0 : i6 to index
    %1 = memref.load %0[%arg0_idx] : memref<64xi32>
    %2 = memref.load %0[%c1] : memref<64xi32>
    %3 = arith.addi %1, %2 : i32
    return %3 : i32
  }
}

// -----

// Test index types as inputs.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-NEXT:   calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:      %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.clk, %mem_0.read_data, %mem_0.done = calyx.memory @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %mem_0.read_data : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%i : index) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.load %0[%i] : memref<64xi32>
    return %1 : i32
  }
}

// -----

// Test index types as outputs.

// CHECH:     module attributes {calyx.entrypoint = "main"} {
// CHECH-NEXT:   calyx.component @main(%in0: i8, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECH-DAG:      %true = hw.constant true
// CHECH-DAG:      %std_pad_0.in, %std_pad_0.out = calyx.std_pad @std_pad_0 : i8, i32
// CHECH-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECH-NEXT:     calyx.wires  {
// CHECH-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECH-NEXT:       calyx.group @ret_assign_0  {
// CHECH-NEXT:         calyx.assign %ret_arg0_reg.in = %std_pad_0.out : i32
// CHECH-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECH-NEXT:         calyx.assign %std_pad_0.in = %in0 : i8
// CHECH-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECH-NEXT:       }
// CHECH-NEXT:     }
// CHECH-NEXT:     calyx.control  {
// CHECH-NEXT:       calyx.seq  {
// CHECH-NEXT:         calyx.enable @ret_assign_0
// CHECH-NEXT:       }
// CHECH-NEXT:     }
// CHECH-NEXT:   } {toplevel}
// CHECH-NEXT: }
module {
  func.func @main(%i : i8) -> index {
    %0 = arith.index_cast %i : i8 to index
    return %0 : index
  }
}

// -----

// External memory store.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK:      calyx.component @main(%in0: i32, %ext_mem0_read_data: i32 {mem = {id = 0 : i32, tag = "read_data"}}, %ext_mem0_done: i1 {mem = {id = 0 : i32, tag = "done"}}, %in2: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%ext_mem0_write_data: i32 {mem = {id = 0 : i32, tag = "write_data"}}, %ext_mem0_addr0: i3 {mem = {addr_idx = 0 : i32, id = 0 : i32, tag = "addr"}}, %ext_mem0_write_en: i1 {mem = {id = 0 : i32, tag = "write_en"}}, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i3
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.group @bb0_0  {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %in2 : i32
// CHECK-NEXT:         calyx.assign %ext_mem0_addr0 = %std_slice_0.out : i3
// CHECK-NEXT:         calyx.assign %ext_mem0_write_data = %in0 : i32
// CHECK-NEXT:         calyx.assign %ext_mem0_write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %ext_mem0_done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @bb0_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%arg0 : i32, %mem0 : memref<8xi32>, %i : index) {
    memref.store %arg0, %mem0[%i] : memref<8xi32>
    return
  }
}

// -----

// External memory load.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK:      calyx.component @main(%in0: i32, %ext_mem0_read_data: i32 {mem = {id = 0 : i32, tag = "read_data"}}, %ext_mem0_done: i1 {mem = {id = 0 : i32, tag = "done"}}, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%ext_mem0_write_data: i32 {mem = {id = 0 : i32, tag = "write_data"}}, %ext_mem0_addr0: i3 {mem = {addr_idx = 0 : i32, id = 0 : i32, tag = "addr"}}, %ext_mem0_write_en: i1 {mem = {id = 0 : i32, tag = "write_en"}}, %out0: i32, %done: i1 {done}) {
// CHECK:          %true = hw.constant true
// CHECK:          %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i3
// CHECK:          %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %ext_mem0_read_data : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %ext_mem0_addr0 = %std_slice_0.out : i3
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%i : index, %mem0 : memref<8xi32>) -> i32 {
    %0 = memref.load %mem0[%i] : memref<8xi32>
    return %0 : i32
  }
}

// -----

// External memory hazard.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK:      calyx.component @main(%in0: i32, %in1: i32, %ext_mem0_read_data: i32 {mem = {id = 0 : i32, tag = "read_data"}}, %ext_mem0_done: i1 {mem = {id = 0 : i32, tag = "done"}}, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%ext_mem0_write_data: i32 {mem = {id = 0 : i32, tag = "write_data"}}, %ext_mem0_addr0: i3 {mem = {addr_idx = 0 : i32, id = 0 : i32, tag = "addr"}}, %ext_mem0_write_en: i1 {mem = {id = 0 : i32, tag = "write_en"}}, %out0: i32, %out1: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i3
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i3
// CHECK-DAG:      %load_1_reg.in, %load_1_reg.write_en, %load_1_reg.clk, %load_1_reg.reset, %load_1_reg.out, %load_1_reg.done = calyx.register @load_1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %load_0_reg.in, %load_0_reg.write_en, %load_0_reg.clk, %load_0_reg.reset, %load_0_reg.out, %load_0_reg.done = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg1_reg.in, %ret_arg1_reg.write_en, %ret_arg1_reg.clk, %ret_arg1_reg.reset, %ret_arg1_reg.out, %ret_arg1_reg.done = calyx.register @ret_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out1 = %ret_arg1_reg.out : i32
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @bb0_0  {
// CHECK-NEXT:         calyx.assign %std_slice_1.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %ext_mem0_addr0 = %std_slice_1.out : i3
// CHECK-NEXT:         calyx.assign %load_0_reg.in = %ext_mem0_read_data : i32
// CHECK-NEXT:         calyx.assign %load_0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %load_0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb0_1  {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %in1 : i32
// CHECK-NEXT:         calyx.assign %ext_mem0_addr0 = %std_slice_0.out : i3
// CHECK-NEXT:         calyx.assign %load_1_reg.in = %ext_mem0_read_data : i32
// CHECK-NEXT:         calyx.assign %load_1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %load_1_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %load_0_reg.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %ret_arg1_reg.in = %load_1_reg.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         %0 = comb.and %ret_arg1_reg.done, %ret_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.enable @bb0_0
// CHECK-NEXT:         calyx.enable @bb0_1
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%i0 : index, %i1 : index, %mem0 : memref<8xi32>) -> (i32, i32) {
    %0 = memref.load %mem0[%i0] : memref<8xi32>
    %1 = memref.load %mem0[%i1] : memref<8xi32>
    return %0, %1 : i32, i32
  }
}

// -----

// Load followed by store to the same memory should be placed in separate groups.

// CHECK:         calyx.group @bb0_0  {
// CHECK-NEXT:         calyx.assign %std_slice_1.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_1.out : i1
// CHECK-NEXT:         calyx.assign %load_0_reg.in = %mem_0.read_data : i32
// CHECK-NEXT:         calyx.assign %load_0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %load_0_reg.done : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      calyx.group @bb0_1  {
// CHECK-NEXT:        calyx.assign %std_slice_0.in = %in0 : i32
// CHECK-NEXT:        calyx.assign %mem_0.addr0 = %std_slice_0.out : i1
// CHECK-NEXT:        calyx.assign %mem_0.write_data = %c1_i32 : i32
// CHECK-NEXT:        calyx.assign %mem_0.write_en = %true : i1
// CHECK-NEXT:        calyx.group_done %mem_0.done : i1
// CHECK-NEXT:      }
module {
  func.func @main(%i : index) -> i32 {
    %c1_32 = arith.constant 1 : i32
    %0 = memref.alloc() : memref<1xi32>
    %1 = memref.load %0[%i] : memref<1xi32>
    memref.store %c1_32, %0[%i] : memref<1xi32>
    return %1 : i32
  }
}

// -----

// Load from memory with more elements than index width (32 bits).

// CHECK: calyx.std_slice {{.*}} i32, i6
module {
  func.func @main(%mem : memref<33xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %0 = memref.load %mem[%c0] : memref<33xi32>
    return %0 : i32
  }
}

// -----

// Check nonzero-width memref address ports for memrefs with some dimension = 1
// See: https://github.com/llvm/circt/issues/2660 and https://github.com/llvm/circt/pull/2661

// CHECK-DAG:       %std_slice_3.in, %std_slice_3.out = calyx.std_slice @std_slice_3 : i32, i1
// CHECK-DAG:       %std_slice_2.in, %std_slice_2.out = calyx.std_slice @std_slice_2 : i32, i1
// CHECK-DAG:           calyx.assign %mem_0.addr0 = %std_slice_3.out : i1
// CHECK-DAG:           calyx.assign %mem_0.addr1 = %std_slice_2.out : i1
module {
  func.func @main() {
    %c1_32 = arith.constant 1 : i32
    %i = arith.constant 0 : index
    %0 = memref.alloc() : memref<1x1x1x1xi32>
    memref.store %c1_32, %0[%i, %i, %i, %i] : memref<1x1x1x1xi32>
    return
  }
}

// -----

// Convert memrefs w/o shape (e.g., memref<i32>) to 1 dimensional Calyx memories 
// of size 1

// CHECK-DAG: %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.clk, %mem_0.read_data, %mem_0.done = calyx.memory @mem_0 <[1] x 32> [1] {external = true} : i1, i32, i1, i1, i32, i1
//CHECK-NEXT: calyx.wires {
//CHECK-NEXT:   calyx.group @bb0_0 {
//CHECK-NEXT:     calyx.assign %mem_0.addr0 = %false : i1
//CHECK-NEXT:     calyx.assign %mem_0.write_data = %c1_i32 : i32
//CHECK-NEXT:     calyx.assign %mem_0.write_en = %true : i1
//CHECK-NEXT:     calyx.group_done %mem_0.done : i1
//CHECK-NEXT:   }
//CHECK-NEXT: }
module {
  func.func @main() {
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<i32>
    memref.store %c1_i32, %alloca[] : memref<i32>
    return
  }
}

