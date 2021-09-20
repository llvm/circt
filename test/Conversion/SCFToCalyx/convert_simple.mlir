// RUN: circt-opt %s --lower-scf-to-calyx -split-input-file | FileCheck %s

// Test chaining combinational logic

// CHECK:      module  {
// CHECK-NEXT:   calyx.program  {
// CHECK-NEXT:     calyx.component @main(%in0: i32, %in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-NEXT:       %true = hw.constant true
// CHECK-NEXT:       %std_sub_0.left, %std_sub_0.right, %std_sub_0.out = calyx.std_sub "std_sub_0" : i32, i32, i32
// CHECK-NEXT:       %std_lsh_0.left, %std_lsh_0.right, %std_lsh_0.out = calyx.std_lsh "std_lsh_0" : i32, i32, i32
// CHECK-NEXT:       %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add "std_add_0" : i32, i32, i32
// CHECK-NEXT:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register "ret_arg0_reg" : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:       calyx.wires  {
// CHECK-NEXT:         calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:         calyx.group @ret_assign_0  {
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.in = %std_sub_0.out : i32
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:           calyx.assign %std_sub_0.left = %std_lsh_0.out : i32
// CHECK-NEXT:           calyx.assign %std_lsh_0.left = %std_add_0.out : i32
// CHECK-NEXT:           calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:           calyx.assign %std_add_0.right = %in1 : i32
// CHECK-NEXT:           calyx.assign %std_lsh_0.right = %in0 : i32
// CHECK-NEXT:           calyx.assign %std_sub_0.right = %std_add_0.out : i32
// CHECK-NEXT:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.control  {
// CHECK-NEXT:         calyx.seq  {
// CHECK-NEXT:           calyx.enable @ret_assign_0 {compiledGroups = []}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func @main(%a0 : i32, %a1 : i32) -> i32 {
    %0 = addi %a0, %a1 : i32
    %1 = shift_left %0, %a0 : i32
    %2 = subi %1, %0 : i32
    return %2 : i32
  }
}

// -----

// Test index casts

// CHECK:      module  {
// CHECK-NEXT:   calyx.program  {
// CHECK-NEXT:     calyx.component @main(%in0: i24, %in1: i24, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i8, %done: i1 {done}) {
// CHECK-NEXT:       %true = hw.constant true
// CHECK-NEXT:       %std_slice_0.in, %std_slice_0.out = calyx.std_slice "std_slice_0" : i32, i8
// CHECK-NEXT:       %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add "std_add_0" : i32, i32, i32
// CHECK-NEXT:       %std_pad_1.in, %std_pad_1.out = calyx.std_pad "std_pad_1" : i24, i32
// CHECK-NEXT:       %std_pad_0.in, %std_pad_0.out = calyx.std_pad "std_pad_0" : i24, i32
// CHECK-NEXT:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register "ret_arg0_reg" : i8, i1, i1, i1, i8, i1
// CHECK-NEXT:       calyx.wires  {
// CHECK-NEXT:         calyx.assign %out0 = %ret_arg0_reg.out : i8
// CHECK-NEXT:         calyx.group @ret_assign_0  {
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.in = %std_slice_0.out : i8
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:           calyx.assign %std_slice_0.in = %std_add_0.out : i32
// CHECK-NEXT:           calyx.assign %std_add_0.left = %std_pad_0.out : i32
// CHECK-NEXT:           calyx.assign %std_pad_0.in = %in1 : i24
// CHECK-NEXT:           calyx.assign %std_add_0.right = %std_pad_1.out : i32
// CHECK-NEXT:           calyx.assign %std_pad_1.in = %in0 : i24
// CHECK-NEXT:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.control  {
// CHECK-NEXT:         calyx.seq  {
// CHECK-NEXT:           calyx.enable @ret_assign_0 {compiledGroups = []}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func @main(%arg0 : i24, %arg1 : i24) -> i8 {
    %0 = index_cast %arg1 : i24 to index
    %1 = index_cast %arg0 : i24 to index
    %2 = addi %0, %1 : index
    %3 = index_cast %2 : index to i8
    return %3 : i8
  }
}

// -----

// Test multiple returns

// CHECK:      module  {
// CHECK-NEXT:   calyx.program  {
// CHECK-NEXT:     calyx.component @main(%in0: i32, %in1: i32, %in2: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-NEXT:       %true = hw.constant true
// CHECK-NEXT:       %std_slt_0.left, %std_slt_0.right, %std_slt_0.out = calyx.std_slt "std_slt_0" : i32, i32, i1
// CHECK-NEXT:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register "ret_arg0_reg" : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:       calyx.wires  {
// CHECK-NEXT:         calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:         calyx.comb_group @bb0_0  {
// CHECK-NEXT:           calyx.assign %std_slt_0.left = %in0 : i32
// CHECK-NEXT:           calyx.assign %std_slt_0.right = %in1 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.group @ret_assign_0  {
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.in = %in1 : i32
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.group @ret_assign_1  {
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.in = %in2 : i32
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.control  {
// CHECK-NEXT:         calyx.seq  {
// CHECK-NEXT:           calyx.if %std_slt_0.out with @bb0_0  {
// CHECK-NEXT:             calyx.seq  {
// CHECK-NEXT:               calyx.enable @ret_assign_0 {compiledGroups = []}
// CHECK-NEXT:             }
// CHECK-NEXT:           } else  {
// CHECK-NEXT:             calyx.seq  {
// CHECK-NEXT:               calyx.enable @ret_assign_1 {compiledGroups = []}
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func @main(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    %0 = cmpi slt, %arg0, %arg1 : i32
    cond_br %0, ^bb1, ^bb2
  ^bb1:
    return %arg1 : i32
  ^bb2:
    return %arg2 : i32
  }
}
