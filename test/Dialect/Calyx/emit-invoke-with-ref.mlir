// RUN: circt-translate --export-calyx --split-input-file --verify-diagnostics %s | FileCheck %s --strict-whitespace

module attributes {calyx.entrypoint = "main"} {
  calyx.component @func(%in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
    %true = hw.constant true
    %false = hw.constant false
    %c0_i32 = hw.constant 0 : i32
    %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i1
    %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
    %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
    // CHECK: ref arg_mem_0 = seq_mem_d1(32, 1, 1);
    %arg_mem_0.addr0, %arg_mem_0.clk, %arg_mem_0.reset, %arg_mem_0.content_en, %arg_mem_0.write_en, %arg_mem_0.write_data, %arg_mem_0.read_data, %arg_mem_0.done = calyx.seq_mem @arg_mem_0 <[1] x 32> [1] : i1, i1, i1, i1, i1, i32, i32, i1
    calyx.wires {
      calyx.assign %out0 = %ret_arg0_reg.out : i32
      calyx.group @bb0_0 {
        calyx.assign %std_slice_0.in = %c0_i32 : i32
        calyx.assign %arg_mem_0.addr0 = %std_slice_0.out : i1
        calyx.assign %arg_mem_0.content_en = %true : i1
        calyx.assign %arg_mem_0.write_en = %false : i1
        calyx.group_done %arg_mem_0.done : i1
      }
      calyx.group @ret_assign_0 {
        calyx.assign %ret_arg0_reg.in = %std_add_0.out : i32
        calyx.assign %ret_arg0_reg.write_en = %true : i1
        calyx.assign %std_add_0.left = %arg_mem_0.read_data : i32
        calyx.assign %std_add_0.right = %in1 : i32
        calyx.group_done %ret_arg0_reg.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.enable @bb0_0
        calyx.enable @ret_assign_0
      }
    }
  }
  calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
    %c0_i32 = hw.constant 0 : i32
    %true = hw.constant true
    // CHECK: @external(1) mem_0 = seq_mem_d1(32, 1, 1);
    %mem_0.addr0, %mem_0.clk, %mem_0.reset, %mem_0.content_en, %mem_0.write_en, %mem_0.write_data, %mem_0.read_data, %mem_0.done = calyx.seq_mem @mem_0 <[1] x 32> [1] {external = true} : i1, i1, i1, i1, i1, i32, i32, i1
    %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
    %func_instance.in1, %func_instance.clk, %func_instance.reset, %func_instance.go, %func_instance.out0, %func_instance.done = calyx.instance @func_instance of @func : i32, i1, i1, i1, i32, i1
    calyx.wires {
      calyx.assign %out0 = %ret_arg0_reg.out : i32
      calyx.group @init_func_instance {
        calyx.assign %func_instance.reset = %true : i1
        calyx.assign %func_instance.go = %true : i1
        calyx.group_done %func_instance.done : i1
      }
      calyx.group @ret_assign_0 {
        calyx.assign %ret_arg0_reg.in = %func_instance.out0 : i32
        calyx.assign %ret_arg0_reg.write_en = %true : i1
        calyx.group_done %ret_arg0_reg.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.seq {
          calyx.enable @init_func_instance
          // CHECK: invoke func_instance[arg_mem_0 = mem_0](in1 = 32'd0)();
          calyx.invoke @func_instance[arg_mem_0 = mem_0](%func_instance.in1 = %c0_i32) -> (i32)
        }
        calyx.enable @ret_assign_0
      }
    }
  } {toplevel}
}
