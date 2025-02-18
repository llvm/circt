// RUN: circt-opt %s --lower-scf-to-calyx -canonicalize -split-input-file | FileCheck %s

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i1, %in1: i32, %in2: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_sub_0.left, %std_sub_0.right, %std_sub_0.out = calyx.std_sub @std_sub_0 : i32, i32, i32
// CHECK-DAG:      %std_lsh_0.left, %std_lsh_0.right, %std_lsh_0.out = calyx.std_lsh @std_lsh_0 : i32, i32, i32
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @ret_assign_0 {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %std_mux_0.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_mux_0.cond = %in0 : i1
// CHECK-NEXT:         calyx.assign %std_mux_0.tru = %std_sub_0.out : i32
// CHECK-NEXT:         calyx.assign %std_sub_0.left = %std_lsh_0.out : i32
// CHECK-NEXT:         calyx.assign %std_lsh_0.left = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.left = %in1 : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %in2 : i32
// CHECK-NEXT:         calyx.assign %std_lsh_0.right = %in1 : i32
// CHECK-NEXT:         calyx.assign %std_sub_0.right = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %std_mux_0.fal = %std_add_0.out : i32
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
  func.func @main(%sel : i1, %a0 : i32, %a1 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.shli %0, %a0 : i32
    %2 = arith.subi %1, %0 : i32
    %3 = arith.select %sel, %2, %0 : i32
    return %3 : i32
  }
}

// -----

// Test multiple return values.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i32, %in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %out1: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %ret_arg1_reg.in, %ret_arg1_reg.write_en, %ret_arg1_reg.clk, %ret_arg1_reg.reset, %ret_arg1_reg.out, %ret_arg1_reg.done = calyx.register @ret_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out1 = %ret_arg1_reg.out : i32
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %ret_arg1_reg.in = %in1 : i32
// CHECK-NEXT:         calyx.assign %ret_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         %0 = comb.and %ret_arg1_reg.done, %ret_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
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
  func.func @main(%a0 : i32, %a1 : i32) -> (i32, i32) {
    return %a0, %a1 : i32, i32
  }
}

// -----

module {
  func.func @main(%a0 : i32, %a1 : i32) -> i32 {
// CHECK:       calyx.group @bb0_0  {
// CHECK-DAG:    calyx.assign %std_mult_pipe_0.left = %in0 : i32
// CHECK-DAG:    calyx.assign %std_mult_pipe_0.right = %in1 : i32
// CHECK-DAG:    calyx.assign %muli_0_reg.in = %std_mult_pipe_0.out : i32
// CHECK-DAG:    calyx.assign %muli_0_reg.write_en = %std_mult_pipe_0.done : i1
// CHECK-DAG:    %0 = comb.xor %std_mult_pipe_0.done, %true : i1
// CHECK-DAG:    calyx.assign %std_mult_pipe_0.go = %0 ? %true : i1
// CHECK-DAG:    calyx.group_done %muli_0_reg.done : i1
// CHECK-NEXT:  }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %muli_0_reg.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
    %0 = arith.muli %a0, %a1 : i32
    return %0 : i32
  }
}

// -----

module {
  func.func @main(%a0 : i32, %a1 : i32) -> i32 {
// CHECK:       calyx.group @bb0_0  {
// CHECK-DAG:    calyx.assign %std_divu_pipe_0.left = %in0 : i32
// CHECK-DAG:    calyx.assign %std_divu_pipe_0.right = %in1 : i32
// CHECK-DAG:    calyx.assign %divui_0_reg.in = %std_divu_pipe_0.out_quotient : i32
// CHECK-DAG:    calyx.assign %divui_0_reg.write_en = %std_divu_pipe_0.done : i1
// CHECK-DAG:    %0 = comb.xor %std_divu_pipe_0.done, %true : i1
// CHECK-DAG:    calyx.assign %std_divu_pipe_0.go = %0 ? %true : i1
// CHECK-DAG:    calyx.group_done %divui_0_reg.done : i1
// CHECK-NEXT:  }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %divui_0_reg.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
    %0 = arith.divui %a0, %a1 : i32
    return %0 : i32
  }
}

// -----

module {
  func.func @main(%a0 : i32, %a1 : i32) -> i32 {
// CHECK:       calyx.group @bb0_0  {
// CHECK-DAG:    calyx.assign %std_remu_pipe_0.left = %in0 : i32
// CHECK-DAG:    calyx.assign %std_remu_pipe_0.right = %in1 : i32
// CHECK-DAG:    calyx.assign %remui_0_reg.in = %std_remu_pipe_0.out_remainder : i32
// CHECK-DAG:    calyx.assign %remui_0_reg.write_en = %std_remu_pipe_0.done : i1
// CHECK-DAG:    %0 = comb.xor %std_remu_pipe_0.done, %true : i1
// CHECK-DAG:    calyx.assign %std_remu_pipe_0.go = %0 ? %true : i1
// CHECK-DAG:    calyx.group_done %remui_0_reg.done : i1
// CHECK-NEXT:  }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %remui_0_reg.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
    %0 = arith.remui %a0, %a1 : i32
    return %0 : i32
  }
}

// -----

module {
  func.func @main(%a0 : i32, %a1 : i32) -> i32 {
// CHECK:       calyx.group @bb0_0  {
// CHECK-DAG:    calyx.assign %std_divs_pipe_0.left = %in0 : i32
// CHECK-DAG:    calyx.assign %std_divs_pipe_0.right = %in1 : i32
// CHECK-DAG:    calyx.assign %divsi_0_reg.in = %std_divs_pipe_0.out_quotient : i32
// CHECK-DAG:    calyx.assign %divsi_0_reg.write_en = %std_divs_pipe_0.done : i1
// CHECK-DAG:    %0 = comb.xor %std_divs_pipe_0.done, %true : i1
// CHECK-DAG:    calyx.assign %std_divs_pipe_0.go = %0 ? %true : i1
// CHECK-DAG:    calyx.group_done %divsi_0_reg.done : i1
// CHECK-NEXT:  }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %divsi_0_reg.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
    %0 = arith.divsi %a0, %a1 : i32
    return %0 : i32
  }
}

// -----

module {
  func.func @main(%a0 : i32, %a1 : i32) -> i32 {
// CHECK:       calyx.group @bb0_0  {
// CHECK-DAG:    calyx.assign %std_rems_pipe_0.left = %in0 : i32
// CHECK-DAG:    calyx.assign %std_rems_pipe_0.right = %in1 : i32
// CHECK-DAG:    calyx.assign %remsi_0_reg.in = %std_rems_pipe_0.out_remainder : i32
// CHECK-DAG:    calyx.assign %remsi_0_reg.write_en = %std_rems_pipe_0.done : i1
// CHECK-DAG:    %0 = comb.xor %std_rems_pipe_0.done, %true : i1
// CHECK-DAG:    calyx.assign %std_rems_pipe_0.go = %0 ? %true : i1
// CHECK-DAG:    calyx.group_done %remsi_0_reg.done : i1
// CHECK-NEXT:  }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %remsi_0_reg.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
    %0 = arith.remsi %a0, %a1 : i32
    return %0 : i32
  }
}

// -----

// CHECK:       calyx.group @ret_assign_0 {
// CHECK-DAG:      calyx.assign %ret_arg0_reg.in = %in0 : i32
// CHECK-DAG:      calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:      calyx.assign %ret_arg1_reg.in = %in0 : i32
// CHECK-DAG:      calyx.assign %ret_arg1_reg.write_en = %true : i1
// CHECK-DAG:      calyx.assign %ret_arg2_reg.in = %in0 : i32
// CHECK-DAG:      calyx.assign %ret_arg2_reg.write_en = %true : i1
// CHECK-DAG:      calyx.assign %ret_arg3_reg.in = %in0 : i32
// CHECK-DAG:      calyx.assign %ret_arg3_reg.write_en = %true : i1
// CHECK-DAG:      calyx.assign %ret_arg4_reg.in = %in0 : i32
// CHECK-DAG:      calyx.assign %ret_arg4_reg.write_en = %true : i1
// CHECK-DAG:      %0 = comb.and %ret_arg4_reg.done, %ret_arg3_reg.done, %ret_arg2_reg.done, %ret_arg1_reg.done, %ret_arg0_reg.done : i1
// CHECK-DAG:      calyx.group_done %0 ? %true : i1
// CHECK-NEXT: }
module {
  func.func @main(%a0 : i32) -> (i32, i32, i32, i32, i32) {
    return %a0, %a0, %a0, %a0, %a0 : i32, i32, i32, i32, i32
  }
}

// -----

// Test sign extensions

// CHECK:     calyx.group @ret_assign_0 {
// CHECK-DAG:   calyx.assign %ret_arg0_reg.in = %std_pad_0.out : i8
// CHECK-DAG:   calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:   calyx.assign %ret_arg1_reg.in = %std_signext_0.out : i8
// CHECK-DAG:   calyx.assign %ret_arg1_reg.write_en = %true : i1
// CHECK-DAG:   calyx.assign %std_pad_0.in = %in0 : i4
// CHECK-DAG:   calyx.assign %std_signext_0.in = %in0 : i4
// CHECK-DAG:   %0 = comb.and %ret_arg1_reg.done, %ret_arg0_reg.done : i1
// CHECK-DAG:   calyx.group_done %0 ? %true : i1
// CHECK-DAG: }

module {
  func.func @main(%arg0 : i4) -> (i8, i8) {
    %0 = arith.extui %arg0 : i4 to i8
    %1 = arith.extsi %arg0 : i4 to i8
    return %0, %1 : i8, i8
  }
}

// -----

// Test integer and floating point constant

// CHECK:     calyx.group @ret_assign_0 {
// CHECK-DAG:   calyx.assign %ret_arg0_reg.in = %in0 : i32
// CHECK-DAG:   calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:   calyx.assign %ret_arg1_reg.in = %c42_i32 : i32
// CHECK-DAG:   calyx.assign %ret_arg1_reg.write_en = %true : i1
// CHECK-DAG:   calyx.assign %ret_arg2_reg.in = %cst : i32
// CHECK-DAG:   calyx.assign %ret_arg2_reg.write_en = %true : i1
// CHECK-DAG:   %0 = comb.and %ret_arg2_reg.done, %ret_arg1_reg.done, %ret_arg0_reg.done : i1
// CHECK-DAG:   calyx.group_done %0 ? %true : i1
// CHECK-DAG: }

module {
  func.func @main(%arg0 : f32) -> (f32, i32, f32) {
    %0 = arith.constant 42 : i32
    %1 = arith.constant 4.2e+1 : f32

    return %arg0, %0, %1 : f32, i32, f32
  }
}

// -----

// Test floating point add

// CHECK:             calyx.group @bb0_0 {
// CHECK-DAG:               calyx.assign %std_addFN_0.left = %in0 : i32
// CHECK-DAG:               calyx.assign %std_addFN_0.right = %cst : i32
// CHECK-DAG:               calyx.assign %addf_0_reg.in = %std_addFN_0.out : i32
// CHECK-DAG:               calyx.assign %addf_0_reg.write_en = %std_addFN_0.done : i1
// CHECK-DAG:               %0 = comb.xor %std_addFN_0.done, %true : i1
// CHECK-DAG:               calyx.assign %std_addFN_0.go = %0 ? %true : i1
// CHECK-DAG:               calyx.assign %std_addFN_0.subOp = %false : i1
// CHECK-DAG:               calyx.group_done %addf_0_reg.done : i1
// CHECK-DAG:             }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %addf_0_reg.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }

module {
  func.func @main(%arg0 : f32) -> f32 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.addf %arg0, %0 : f32

    return %1 : f32
  }
}

// -----

// Test floating point mul

// CHECK:        %cst = calyx.constant @cst_0 <4.200000e+00 : f32> : i32
// CHECK-DAG:    %true = hw.constant true
// CHECK-DAG:    %mulf_0_reg.in, %mulf_0_reg.write_en, %mulf_0_reg.clk, %mulf_0_reg.reset, %mulf_0_reg.out, %mulf_0_reg.done = calyx.register @mulf_0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:    %std_mulFN_0.clk, %std_mulFN_0.reset, %std_mulFN_0.go, %std_mulFN_0.control, %std_mulFN_0.left, %std_mulFN_0.right, %std_mulFN_0.roundingMode, %std_mulFN_0.out, %std_mulFN_0.exceptionalFlags, %std_mulFN_0.done = calyx.ieee754.mul @std_mulFN_0 : i1, i1, i1, i1, i32, i32, i3, i32, i5, i1
// CHECK-DAG:    %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:      calyx.group @bb0_0 {
// CHECK-DAG:        calyx.assign %std_mulFN_0.left = %in0 : i32
// CHECK-DAG:        calyx.assign %std_mulFN_0.right = %cst : i32
// CHECK-DAG:        calyx.assign %mulf_0_reg.in = %std_mulFN_0.out : i32
// CHECK-DAG:        calyx.assign %mulf_0_reg.write_en = %std_mulFN_0.done : i1
// CHECK-DAG:        %0 = comb.xor %std_mulFN_0.done, %true : i1
// CHECK-DAG:        calyx.assign %std_mulFN_0.go = %0 ? %true : i1
// CHECK-DAG:        calyx.group_done %mulf_0_reg.done : i1
// CHECK-DAG:      }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %mulf_0_reg.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
module {
  func.func @main(%arg0 : f32) -> f32 {
    %0 = arith.constant 4.2 : f32
    %1 = arith.mulf %arg0, %0 : f32

    return %1 : f32
  }
}

// -----

// Test parallel op lowering

// CHECK:    calyx.wires {
// CHECK-DAG:      calyx.group @bb0_0 {
// CHECK-DAG:        calyx.assign %std_slice_5.in = %c1_i32 : i32
// CHECK-DAG:        calyx.assign %arg_mem_0.addr0 = %std_slice_5.out : i1
// CHECK-DAG:        calyx.assign %arg_mem_0.content_en = %true : i1
// CHECK-DAG:        calyx.assign %arg_mem_0.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_0_reg.in = %arg_mem_0.read_data : i32
// CHECK-DAG:        calyx.assign %load_0_reg.write_en = %arg_mem_0.done : i1
// CHECK-DAG:        calyx.group_done %load_0_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb0_1 {
// CHECK-DAG:        calyx.assign %std_slice_4.in = %c1_i32 : i32
// CHECK-DAG:        calyx.assign %arg_mem_2.addr0 = %std_slice_4.out : i1
// CHECK-DAG:        calyx.assign %arg_mem_2.content_en = %true : i1
// CHECK-DAG:        calyx.assign %arg_mem_2.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_1_reg.in = %arg_mem_2.read_data : i32
// CHECK-DAG:        calyx.assign %load_1_reg.write_en = %arg_mem_2.done : i1
// CHECK-DAG:        calyx.group_done %load_1_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb0_3 {
// CHECK-DAG:        calyx.assign %std_slice_3.in = %c1_i32 : i32
// CHECK-DAG:        calyx.assign %arg_mem_1.addr0 = %std_slice_3.out : i1
// CHECK-DAG:        calyx.assign %arg_mem_1.write_data = %std_add_0.out : i32
// CHECK-DAG:        calyx.assign %arg_mem_1.write_en = %true : i1
// CHECK-DAG:        calyx.assign %arg_mem_1.content_en = %true : i1
// CHECK-DAG:        calyx.assign %std_add_0.left = %load_0_reg.out : i32
// CHECK-DAG:        calyx.assign %std_add_0.right = %load_1_reg.out : i32
// CHECK-DAG:        calyx.group_done %arg_mem_1.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb0_4 {
// CHECK-DAG:        calyx.assign %std_slice_2.in = %c0_i32 : i32
// CHECK-DAG:        calyx.assign %arg_mem_0.addr0 = %std_slice_2.out : i1
// CHECK-DAG:        calyx.assign %arg_mem_0.content_en = %true : i1
// CHECK-DAG:        calyx.assign %arg_mem_0.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_2_reg.in = %arg_mem_0.read_data : i32
// CHECK-DAG:        calyx.assign %load_2_reg.write_en = %arg_mem_0.done : i1
// CHECK-DAG:        calyx.group_done %load_2_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb0_5 {
// CHECK-DAG:        calyx.assign %std_slice_1.in = %c0_i32 : i32
// CHECK-DAG:        calyx.assign %arg_mem_2.addr0 = %std_slice_1.out : i1
// CHECK-DAG:        calyx.assign %arg_mem_2.content_en = %true : i1
// CHECK-DAG:        calyx.assign %arg_mem_2.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_3_reg.in = %arg_mem_2.read_data : i32
// CHECK-DAG:        calyx.assign %load_3_reg.write_en = %arg_mem_2.done : i1
// CHECK-DAG:        calyx.group_done %load_3_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb0_7 {
// CHECK-DAG:        calyx.assign %std_slice_0.in = %c0_i32 : i32
// CHECK-DAG:        calyx.assign %arg_mem_1.addr0 = %std_slice_0.out : i1
// CHECK-DAG:        calyx.assign %arg_mem_1.write_data = %std_add_1.out : i32
// CHECK-DAG:        calyx.assign %arg_mem_1.write_en = %true : i1
// CHECK-DAG:        calyx.assign %arg_mem_1.content_en = %true : i1
// CHECK-DAG:        calyx.assign %std_add_1.left = %load_2_reg.out : i32
// CHECK-DAG:        calyx.assign %std_add_1.right = %load_3_reg.out : i32
// CHECK-DAG:        calyx.group_done %arg_mem_1.done : i1
// CHECK-DAG:      }
// CHECK-DAG:    }
// CHECK-DAG:    calyx.control {
// CHECK-DAG:      calyx.seq {
// CHECK-DAG:        calyx.par {
// CHECK-DAG:          calyx.seq {
// CHECK-DAG:            calyx.enable @bb0_0
// CHECK-DAG:            calyx.enable @bb0_1
// CHECK-DAG:            calyx.enable @bb0_3
// CHECK-DAG:          }
// CHECK-DAG:          calyx.seq {
// CHECK-DAG:            calyx.enable @bb0_4
// CHECK-DAG:            calyx.enable @bb0_5
// CHECK-DAG:            calyx.enable @bb0_7
// CHECK-DAG:          }
// CHECK-DAG:        }
// CHECK-DAG:      }
// CHECK-DAG:    }

module {
  func.func @main(%arg0: memref<2xi32>, %arg1: memref<2xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<2xi32>
    scf.parallel (%arg2) = (%c0) to (%c1) step (%c1) {
      scf.execute_region {
        %0 = memref.load %arg0[%c1] : memref<2xi32>
        %1 = memref.load %alloc[%c1] : memref<2xi32>
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %arg1[%c1] : memref<2xi32>
        scf.yield
      }
      scf.execute_region {
        %0 = memref.load %arg0[%c0] : memref<2xi32>
        %1 = memref.load %alloc[%c0] : memref<2xi32>
        %2 = arith.addi %0, %1 : i32
        memref.store %2, %arg1[%c0] : memref<2xi32>
        scf.yield
      }
    } {calyx.unroll = true}
    return
  }
}

// -----

// Test lowering SelectOp and CmpFOp with floating point operands

// CHECK:    %std_mux_1.cond, %std_mux_1.tru, %std_mux_1.fal, %std_mux_1.out = calyx.std_mux @std_mux_1 : i1, i64, i64, i64
// CHECK-DAG:    %unordered_port_1_reg.in, %unordered_port_1_reg.write_en, %unordered_port_1_reg.clk, %unordered_port_1_reg.reset, %unordered_port_1_reg.out, %unordered_port_1_reg.done = calyx.register @unordered_port_1_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:    %cmpf_1_reg.in, %cmpf_1_reg.write_en, %cmpf_1_reg.clk, %cmpf_1_reg.reset, %cmpf_1_reg.out, %cmpf_1_reg.done = calyx.register @cmpf_1_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:    %std_compareFN_1.clk, %std_compareFN_1.reset, %std_compareFN_1.go, %std_compareFN_1.left, %std_compareFN_1.right, %std_compareFN_1.signaling, %std_compareFN_1.lt, %std_compareFN_1.eq, %std_compareFN_1.gt, %std_compareFN_1.unordered, %std_compareFN_1.exceptionalFlags, %std_compareFN_1.done = calyx.ieee754.compare @std_compareFN_1 : i1, i1, i1, i64, i64, i1, i1, i1, i1, i1, i5, i1
// CHECK-DAG:    %std_mux_0.cond, %std_mux_0.tru, %std_mux_0.fal, %std_mux_0.out = calyx.std_mux @std_mux_0 : i1, i64, i64, i64
// CHECK-DAG:    %std_and_0.left, %std_and_0.right, %std_and_0.out = calyx.std_and @std_and_0 : i1, i1, i1
// CHECK-DAG:    %std_or_0.left, %std_or_0.right, %std_or_0.out = calyx.std_or @std_or_0 : i1, i1, i1
// CHECK-DAG:    %unordered_port_0_reg.in, %unordered_port_0_reg.write_en, %unordered_port_0_reg.clk, %unordered_port_0_reg.reset, %unordered_port_0_reg.out, %unordered_port_0_reg.done = calyx.register @unordered_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:    %compare_port_0_reg.in, %compare_port_0_reg.write_en, %compare_port_0_reg.clk, %compare_port_0_reg.reset, %compare_port_0_reg.out, %compare_port_0_reg.done = calyx.register @compare_port_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:    %cmpf_0_reg.in, %cmpf_0_reg.write_en, %cmpf_0_reg.clk, %cmpf_0_reg.reset, %cmpf_0_reg.out, %cmpf_0_reg.done = calyx.register @cmpf_0_reg : i1, i1, i1, i1, i1, i1
// CHECK-DAG:    %std_compareFN_0.clk, %std_compareFN_0.reset, %std_compareFN_0.go, %std_compareFN_0.left, %std_compareFN_0.right, %std_compareFN_0.signaling, %std_compareFN_0.lt, %std_compareFN_0.eq, %std_compareFN_0.gt, %std_compareFN_0.unordered, %std_compareFN_0.exceptionalFlags, %std_compareFN_0.done = calyx.ieee754.compare @std_compareFN_0 : i1, i1, i1, i64, i64, i1, i1, i1, i1, i1, i5, i1
// CHECK:    calyx.wires {
// CHECK:      calyx.group @bb0_0 {
// CHECK-DAG:        calyx.assign %std_compareFN_0.left = %in0 : i64
// CHECK-DAG:        calyx.assign %std_compareFN_0.right = %in1 : i64
// CHECK-DAG:        calyx.assign %std_compareFN_0.signaling = %true : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %compare_port_0_reg.in = %std_compareFN_0.gt : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.write_en = %std_compareFN_0.done : i1
// CHECK-DAG:        calyx.assign %unordered_port_0_reg.in = %std_compareFN_0.unordered : i1
// CHECK-DAG:        calyx.assign %std_or_0.left = %compare_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_or_0.right = %unordered_port_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_and_0.left = %compare_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %std_and_0.right = %unordered_port_0_reg.done : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.in = %std_or_0.out : i1
// CHECK-DAG:        calyx.assign %cmpf_0_reg.write_en = %std_and_0.out : i1
// CHECK-DAG:        %0 = comb.xor %std_compareFN_0.done, %true : i1
// CHECK-DAG:        calyx.assign %std_compareFN_0.go = %0 ? %true : i1
// CHECK-DAG:        calyx.group_done %cmpf_0_reg.done : i1
// CHECK-DAG:      }
// CHECK:      calyx.group @bb0_2 {
// CHECK-DAG:        calyx.assign %std_compareFN_1.left = %in1 : i64
// CHECK-DAG:        calyx.assign %std_compareFN_1.right = %in1 : i64
// CHECK-DAG:        calyx.assign %std_compareFN_1.signaling = %false : i1
// CHECK-DAG:        calyx.assign %unordered_port_1_reg.write_en = %std_compareFN_1.done : i1
// CHECK-DAG:        calyx.assign %unordered_port_1_reg.in = %std_compareFN_1.unordered : i1
// CHECK-DAG:        calyx.assign %cmpf_1_reg.in = %unordered_port_1_reg.out : i1
// CHECK-DAG:        calyx.assign %cmpf_1_reg.write_en = %unordered_port_1_reg.done : i1
// CHECK-DAG:        %0 = comb.xor %std_compareFN_1.done, %true : i1
// CHECK-DAG:        calyx.assign %std_compareFN_1.go = %0 ? %true : i1
// CHECK-DAG:        calyx.group_done %cmpf_1_reg.done : i1
// CHECK-DAG:      }
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %std_mux_1.out : i64
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.assign %std_mux_1.cond = %cmpf_1_reg.out : i1
// CHECK-DAG:        calyx.assign %std_mux_1.tru = %in1 : i64
// CHECK-DAG:        calyx.assign %std_mux_1.fal = %std_mux_0.out : i64
// CHECK-DAG:        calyx.assign %std_mux_0.cond = %cmpf_0_reg.out : i1
// CHECK-DAG:        calyx.assign %std_mux_0.tru = %in0 : i64
// CHECK-DAG:        calyx.assign %std_mux_0.fal = %in1 : i64
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:    }

module {
  func.func @main(%arg0: f64, %arg1: f64) -> f64 {
    %0 = arith.cmpf ugt, %arg0, %arg1 : f64
    %1 = arith.select %0, %arg0, %arg1 : f64
    %2 = arith.cmpf uno, %arg1, %arg1 : f64
    %3 = arith.select %2, %arg1, %1 : f64
    return %3 : f64
  }
}

// Test SelectOp with signed integer type to signless integer type

// -----

// CHECK:    %std_mux_0.cond, %std_mux_0.tru, %std_mux_0.fal, %std_mux_0.out = calyx.std_mux @std_mux_0 : i1, i32, i32, i32
// CHECK-DAG:    %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:    calyx.wires {
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-DAG:        calyx.assign %ret_arg0_reg.in = %std_mux_0.out : i32
// CHECK-DAG:        calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-DAG:        calyx.assign %std_mux_0.cond = %in2 : i1
// CHECK-DAG:        calyx.assign %std_mux_0.tru = %in0 : i32
// CHECK-DAG:        calyx.assign %std_mux_0.fal = %in1 : i32
// CHECK-DAG:        calyx.group_done %ret_arg0_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:    }

module {
  func.func @main(%true : si32, %false: si32, %cond: i1) -> si32 {
    %res = "arith.select" (%cond, %true, %false) : (i1, si32, si32) -> si32
    return %res : si32
  }
}

