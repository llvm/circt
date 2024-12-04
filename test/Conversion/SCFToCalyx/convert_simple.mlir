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
// CHECK-DAG:        calyx.assign %std_slice_7.in = %c0_i32 : i32
// CHECK-DAG:        calyx.assign %mem_1.addr0 = %std_slice_7.out : i3
// CHECK-DAG:        calyx.assign %mem_1.content_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_1.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_0_reg.in = %mem_1.read_data : i32
// CHECK-DAG:        calyx.assign %load_0_reg.write_en = %mem_1.done : i1
// CHECK-DAG:        calyx.group_done %load_0_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb0_1 {
// CHECK-DAG:        calyx.assign %std_slice_6.in = %c0_i32 : i32
// CHECK-DAG:        calyx.assign %mem_0.addr0 = %std_slice_6.out : i3
// CHECK-DAG:        calyx.assign %mem_0.write_data = %load_0_reg.out : i32
// CHECK-DAG:        calyx.assign %mem_0.write_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_0.content_en = %true : i1
// CHECK-DAG:        calyx.group_done %mem_0.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb1_0 {
// CHECK-DAG:        calyx.assign %std_slice_5.in = %c4_i32 : i32
// CHECK-DAG:        calyx.assign %mem_1.addr0 = %std_slice_5.out : i3
// CHECK-DAG:        calyx.assign %mem_1.content_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_1.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_1_reg.in = %mem_1.read_data : i32
// CHECK-DAG:        calyx.assign %load_1_reg.write_en = %mem_1.done : i1
// CHECK-DAG:        calyx.group_done %load_1_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb1_1 {
// CHECK-DAG:        calyx.assign %std_slice_4.in = %c1_i32 : i32
// CHECK-DAG:        calyx.assign %mem_0.addr0 = %std_slice_4.out : i3
// CHECK-DAG:        calyx.assign %mem_0.write_data = %load_1_reg.out : i32
// CHECK-DAG:        calyx.assign %mem_0.write_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_0.content_en = %true : i1
// CHECK-DAG:        calyx.group_done %mem_0.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb2_0 {
// CHECK-DAG:        calyx.assign %std_slice_3.in = %c2_i32 : i32
// CHECK-DAG:        calyx.assign %mem_1.addr0 = %std_slice_3.out : i3
// CHECK-DAG:        calyx.assign %mem_1.content_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_1.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_2_reg.in = %mem_1.read_data : i32
// CHECK-DAG:        calyx.assign %load_2_reg.write_en = %mem_1.done : i1
// CHECK-DAG:        calyx.group_done %load_2_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb2_1 {
// CHECK-DAG:        calyx.assign %std_slice_2.in = %c4_i32 : i32
// CHECK-DAG:        calyx.assign %mem_0.addr0 = %std_slice_2.out : i3
// CHECK-DAG:        calyx.assign %mem_0.write_data = %load_2_reg.out : i32
// CHECK-DAG:        calyx.assign %mem_0.write_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_0.content_en = %true : i1
// CHECK-DAG:        calyx.group_done %mem_0.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb3_0 {
// CHECK-DAG:        calyx.assign %std_slice_1.in = %c6_i32 : i32
// CHECK-DAG:        calyx.assign %mem_1.addr0 = %std_slice_1.out : i3
// CHECK-DAG:        calyx.assign %mem_1.content_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_1.write_en = %false : i1
// CHECK-DAG:        calyx.assign %load_3_reg.in = %mem_1.read_data : i32
// CHECK-DAG:        calyx.assign %load_3_reg.write_en = %mem_1.done : i1
// CHECK-DAG:        calyx.group_done %load_3_reg.done : i1
// CHECK-DAG:      }
// CHECK-DAG:      calyx.group @bb3_1 {
// CHECK-DAG:        calyx.assign %std_slice_0.in = %c5_i32 : i32
// CHECK-DAG:        calyx.assign %mem_0.addr0 = %std_slice_0.out : i3
// CHECK-DAG:        calyx.assign %mem_0.write_data = %load_3_reg.out : i32
// CHECK-DAG:        calyx.assign %mem_0.write_en = %true : i1
// CHECK-DAG:        calyx.assign %mem_0.content_en = %true : i1
// CHECK-DAG:        calyx.group_done %mem_0.done : i1
// CHECK-DAG:      }
// CHECK-DAG:    }
// CHECK-DAG:    calyx.control {
// CHECK-DAG:      calyx.seq {
// CHECK-DAG:        calyx.par {
// CHECK-DAG:          calyx.seq {
// CHECK-DAG:            calyx.enable @bb0_0
// CHECK-DAG:            calyx.enable @bb0_1
// CHECK-DAG:          }
// CHECK-DAG:          calyx.seq {
// CHECK-DAG:            calyx.enable @bb1_0
// CHECK-DAG:            calyx.enable @bb1_1
// CHECK-DAG:          }
// CHECK-DAG:          calyx.seq {
// CHECK-DAG:            calyx.enable @bb2_0
// CHECK-DAG:            calyx.enable @bb2_1
// CHECK-DAG:          }
// CHECK-DAG:          calyx.seq {
// CHECK-DAG:            calyx.enable @bb3_0
// CHECK-DAG:            calyx.enable @bb3_1
// CHECK-DAG:          }
// CHECK-DAG:        }
// CHECK-DAG:      }
// CHECK-DAG:    }

module {
  func.func @main() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<6xi32>
    %alloc_1 = memref.alloc() : memref<6xi32>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c3, %c2) step (%c2, %c1) {
      %4 = arith.shli %arg3, %c2 : index
      %5 = arith.addi %4, %arg2 : index
      %6 = memref.load %alloc_1[%5] : memref<6xi32>
      %7 = arith.shli %arg2, %c1 : index
      %8 = arith.addi %7, %arg3 : index
      memref.store %6, %alloc[%8] : memref<6xi32>
      scf.reduce 
    }
    return
  }
}

