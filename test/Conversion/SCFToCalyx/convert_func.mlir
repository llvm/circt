// RUN: circt-opt %s --lower-scf-to-calyx="top-level-function=main" -canonicalize -split-input-file | FileCheck %s

// CHECK:      calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:    %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:    %c2_i32 = hw.constant 2 : i32
// CHECK-DAG:    %true = hw.constant true
// CHECK-DAG:    %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:    %fun_instance.in0, %fun_instance.in1, %fun_instance.clk, %fun_instance.reset, %fun_instance.go, %fun_instance.out0, %fun_instance.done = calyx.instance @fun_instance of @fun : i32, i32, i1, i1, i1, i32, i1
// CHECK-NEXT:   calyx.wires {
// CHECK-NEXT:     calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:     calyx.group @init_fun_instance {
// CHECK-NEXT:       calyx.assign %fun_instance.reset = %true : i1
// CHECK-NEXT:       calyx.assign %fun_instance.go = %true : i1
// CHECK-NEXT:       calyx.group_done %fun_instance.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @ret_assign_0 {
// CHECK-NEXT:       calyx.assign %ret_arg0_reg.in = %fun_instance.out0 : i32
// CHECK-NEXT:       calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   calyx.control {
// CHECK-NEXT:     calyx.seq {
// CHECK-NEXT:       calyx.enable @init_fun_instance
// CHECK-NEXT:       calyx.invoke @fun_instance(%fun_instance.in0 = %c1_i32, %fun_instance.in1 = %c2_i32) -> (i32, i32)
// CHECK-NEXT:       calyx.enable @ret_assign_0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: } {toplevel}

module {
  func.func @fun(%a0 : i32, %a1 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.shli %0, %a0 : i32
    %2 = arith.subi %1, %0 : i32
    return %2 : i32
  }

  func.func @main() -> i32 {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 2 : i32
    %ret = func.call @fun(%0, %1) : (i32, i32) -> i32 
    func.return %ret : i32
  } 
}

// -----

// CHECK:      calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
// CHECK-DAG:    %true = hw.constant true
// CHECK-DAG:    %c64_i32 = hw.constant 64 : i32
// CHECK-DAG:    %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:    %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:    %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i6
// CHECK-DAG:    %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:    %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
// CHECK-DAG:    %mem_1.addr0, %mem_1.write_data, %mem_1.write_en, %mem_1.write_done, %mem_1.clk, %mem_1.read_data, %mem_1.read_en, %mem_1.read_done = calyx.seq_mem @mem_1 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK-DAG:    %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.write_done, %mem_0.clk, %mem_0.read_data, %mem_0.read_en, %mem_0.read_done = calyx.seq_mem @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK-DAG:    %while_0_arg0_reg.in, %while_0_arg0_reg.write_en, %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %while_0_arg0_reg.out, %while_0_arg0_reg.done = calyx.register @while_0_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:    %fun_instance.in0, %fun_instance.in1, %fun_instance.clk, %fun_instance.reset, %fun_instance.go, %fun_instance.out0, %fun_instance.done = calyx.instance @fun_instance of @fun : i32, i32, i1, i1, i1, i32, i1
// CHECK-NEXT:   calyx.wires {
// CHECK-NEXT:     calyx.group @init_fun_instance {
// CHECK-NEXT:       calyx.assign %fun_instance.reset = %true : i1
// CHECK-NEXT:       calyx.assign %fun_instance.go = %true : i1
// CHECK-NEXT:       calyx.group_done %fun_instance.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @assign_while_0_init_0 {
// CHECK-NEXT:       calyx.assign %while_0_arg0_reg.in = %c0_i32 : i32
// CHECK-NEXT:       calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.comb_group @bb0_0 {
// CHECK-NEXT:       calyx.assign %std_lt_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:       calyx.assign %std_lt_0.right = %c64_i32 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @bb0_1 {
// CHECK-NEXT:       calyx.assign %std_slice_1.in = %while_0_arg0_reg.out : i32
// CHECK-NEXT:       calyx.assign %mem_0.addr0 = %std_slice_1.out : i6
// CHECK-NEXT:       calyx.assign %mem_0.read_en = %true : i1
// CHECK-NEXT:       calyx.group_done %mem_0.read_done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @bb0_2 {
// CHECK-NEXT:       calyx.assign %std_slice_0.in = %while_0_arg0_reg.out : i32
// CHECK-NEXT:       calyx.assign %mem_1.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:       calyx.assign %mem_1.write_data = %fun_instance.out0 : i32
// CHECK-NEXT:       calyx.assign %mem_1.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %mem_1.write_done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @assign_while_0_latch {
// CHECK-NEXT:       calyx.assign %while_0_arg0_reg.in = %std_add_0.out : i32
// CHECK-NEXT:       calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:       calyx.assign %std_add_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:       calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:       calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   calyx.control {
// CHECK-NEXT:     calyx.seq {
// CHECK-NEXT:       calyx.enable @assign_while_0_init_0
// CHECK-NEXT:       calyx.while %std_lt_0.out with @bb0_0 {
// CHECK-NEXT:         calyx.seq {
// CHECK-NEXT:           calyx.enable @bb0_1
// CHECK-NEXT:           calyx.enable @init_fun_instance
// CHECK-NEXT:           calyx.invoke @fun_instance(%fun_instance.in0 = %mem_0.read_data, %fun_instance.in1 = %mem_0.read_data) -> (i32, i32)
// CHECK-NEXT:           calyx.enable @bb0_2
// CHECK-NEXT:           calyx.enable @assign_while_0_latch
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: } {toplevel}

module {
  func.func @fun(%a0 : i32, %a1 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.shli %0, %a0 : i32
    %2 = arith.subi %1, %0 : i32
    return %2 : i32
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>  
    %c0i32 = arith.constant 1 : i32
    %c1i32 = arith.constant 2 : i32
    scf.while(%arg0 = %c0) : (index) -> (index) {
      %cond = arith.cmpi ult, %arg0, %c64 : index
      scf.condition(%cond) %arg0 : index
    } do {
    ^bb0(%arg1: index):
      %v = memref.load %0[%arg1] : memref<64xi32>
      %c = func.call @fun(%v, %v) : (i32, i32) -> i32
      memref.store %c, %1[%arg1] : memref<64xi32>
      %inc = arith.addi %arg1, %c1 : index
      scf.yield %inc : index
    }
    return
  }
}

// -----

// CHECK:      calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
// CHECK-DAG:    %true = hw.constant true
// CHECK-DAG:    %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:    %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:    %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i6
// CHECK-DAG:    %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:    %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:    %load_0_reg.in, %load_0_reg.write_en, %load_0_reg.clk, %load_0_reg.reset, %load_0_reg.out, %load_0_reg.done = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:    %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.write_done, %mem_0.clk, %mem_0.read_data, %mem_0.read_en, %mem_0.read_done = calyx.seq_mem @mem_0 <[40] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK-DAG:    %for_0_induction_var_reg.in, %for_0_induction_var_reg.write_en, %for_0_induction_var_reg.clk, %for_0_induction_var_reg.reset, %for_0_induction_var_reg.out, %for_0_induction_var_reg.done = calyx.register @for_0_induction_var_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:    %fun_instance.in0, %fun_instance.in1, %fun_instance.clk, %fun_instance.reset, %fun_instance.go, %fun_instance.out0, %fun_instance.done = calyx.instance @fun_instance of @fun : i32, i32, i1, i1, i1, i32, i1
// CHECK-NEXT:   calyx.wires {
// CHECK-NEXT:     calyx.group @init_fun_instance {
// CHECK-NEXT:       calyx.assign %fun_instance.reset = %true : i1
// CHECK-NEXT:       calyx.assign %fun_instance.go = %true : i1
// CHECK-NEXT:       calyx.group_done %fun_instance.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @init_for_0_induction_var {
// CHECK-NEXT:       calyx.assign %for_0_induction_var_reg.in = %c0_i32 : i32
// CHECK-NEXT:       calyx.assign %for_0_induction_var_reg.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %for_0_induction_var_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @bb0_0 {
// CHECK-NEXT:       calyx.assign %std_slice_1.in = %for_0_induction_var_reg.out : i32
// CHECK-NEXT:       calyx.assign %mem_0.addr0 = %std_slice_1.out : i6
// CHECK-NEXT:       calyx.assign %mem_0.read_en = %true : i1
// CHECK-NEXT:       calyx.assign %load_0_reg.in = %mem_0.read_data : i32
// CHECK-NEXT:       calyx.assign %load_0_reg.write_en = %mem_0.read_done : i1
// CHECK-NEXT:       calyx.group_done %load_0_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @bb0_1 {
// CHECK-NEXT:       calyx.assign %std_slice_0.in = %for_0_induction_var_reg.out : i32
// CHECK-NEXT:       calyx.assign %mem_0.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:       calyx.assign %mem_0.write_data = %fun_instance.out0 : i32
// CHECK-NEXT:       calyx.assign %mem_0.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %mem_0.write_done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @incr_for_0_induction_var {
// CHECK-NEXT:       calyx.assign %std_add_0.left = %for_0_induction_var_reg.out : i32
// CHECK-NEXT:       calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:       calyx.assign %for_0_induction_var_reg.in = %std_add_0.out : i32
// CHECK-NEXT:       calyx.assign %for_0_induction_var_reg.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %for_0_induction_var_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   calyx.control {
// CHECK-NEXT:     calyx.seq {
// CHECK-NEXT:       calyx.enable @init_for_0_induction_var
// CHECK-NEXT:       calyx.repeat 40 {
// CHECK-NEXT:         calyx.seq {
// CHECK-NEXT:           calyx.enable @bb0_0
// CHECK-NEXT:           calyx.enable @init_fun_instance
// CHECK-NEXT:           calyx.invoke @fun_instance(%fun_instance.in0 = %load_0_reg.out, %fun_instance.in1 = %load_0_reg.out) -> (i32, i32)
// CHECK-NEXT:           calyx.enable @bb0_1
// CHECK-NEXT:           calyx.enable @incr_for_0_induction_var
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: } {toplevel}

module {
  func.func @fun(%a0 : i32, %a1 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.shli %0, %a0 : i32
    %2 = arith.subi %1, %0 : i32
    return %2 : i32
  }

  func.func @main() {
    %alloca = memref.alloca() : memref<40xi32>
    %c0 = arith.constant 0 : index
    %c40 = arith.constant 40 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c40 step %c1 {
      %0 = memref.load %alloca[%arg0] : memref<40xi32>
      %1 = func.call @fun(%0, %0) : (i32, i32) -> i32 
      memref.store %1, %alloca[%arg0] : memref<40xi32>
    }
    return
  }
}

// -----

// CHECK:      calyx.component @main(%in0: i32, %in1: i32, %in2: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:    %true = hw.constant true
// CHECK-DAG:    %std_ge_0.left, %std_ge_0.right, %std_ge_0.out = calyx.std_ge @std_ge_0 : i32, i32, i1
// CHECK-DAG:    %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK-DAG:    %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:    %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:    %func_instance.in0, %func_instance.in1, %func_instance.clk, %func_instance.reset, %func_instance.go, %func_instance.out0, %func_instance.done = calyx.instance @func_instance of @func : i32, i32, i1, i1, i1, i32, i1
// CHECK-NEXT:   calyx.wires {
// CHECK-NEXT:     calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:     calyx.group @init_func_instance {
// CHECK-NEXT:       calyx.assign %func_instance.reset = %true : i1
// CHECK-NEXT:       calyx.assign %func_instance.go = %true : i1
// CHECK-NEXT:       calyx.group_done %func_instance.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.comb_group @bb0_2 {
// CHECK-NEXT:       calyx.assign %std_ge_0.left = %std_add_1.out : i32
// CHECK-NEXT:       calyx.assign %std_ge_0.right = %in2 : i32
// CHECK-NEXT:       calyx.assign %std_add_1.left = %std_add_0.out : i32
// CHECK-NEXT:       calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:       calyx.assign %std_add_0.right = %in1 : i32
// CHECK-NEXT:       calyx.assign %std_add_1.right = %in1 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @ret_assign_0 {
// CHECK-NEXT:       calyx.assign %ret_arg0_reg.in = %func_instance.out0 : i32
// CHECK-NEXT:       calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.group @ret_assign_1 {
// CHECK-NEXT:       calyx.assign %ret_arg0_reg.in = %func_instance.out0 : i32
// CHECK-NEXT:       calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:       calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   calyx.control {
// CHECK-NEXT:     calyx.seq {
// CHECK-NEXT:       calyx.if %std_ge_0.out with @bb0_2 {
// CHECK-NEXT:         calyx.seq {
// CHECK-NEXT:           calyx.enable @init_func_instance
// CHECK-NEXT:           calyx.invoke @func_instance(%func_instance.in0 = %std_add_0.out, %func_instance.in1 = %std_add_1.out) -> (i32, i32)
// CHECK-NEXT:           calyx.enable @ret_assign_0
// CHECK-NEXT:         }
// CHECK-NEXT:       } else {
// CHECK-NEXT:         calyx.seq {
// CHECK-NEXT:           calyx.enable @init_func_instance
// CHECK-NEXT:           calyx.invoke @func_instance(%func_instance.in0 = %std_add_0.out, %func_instance.in1 = %std_add_1.out) -> (i32, i32)
// CHECK-NEXT:           calyx.enable @ret_assign_1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: } {toplevel}

module {
  func.func @func(%a0 : i32, %a1 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.shli %0, %a0 : i32
    %2 = arith.subi %1, %0 : i32
    return %2 : i32
  }

  func.func @main(%a0 : i32, %a1 : i32, %a2 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.addi %0, %a1 : i32
    %b = arith.cmpi uge, %1, %a2 : i32
    cf.cond_br %b, ^bb1, ^bb2
  ^bb1:
    %ret0 = func.call @func(%0, %1) : (i32, i32) -> i32
    return %ret0 : i32
  ^bb2:
    %ret1 = func.call @func(%0, %1) : (i32, i32) -> i32
    return %ret1 : i32
  }
}
