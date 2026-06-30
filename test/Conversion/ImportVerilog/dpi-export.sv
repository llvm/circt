// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-llhd %s | FileCheck %s --check-prefix=LLHD
// REQUIRES: slang
// UNSUPPORTED: valgrind

package p;
  function int exported_func(input int x);
    return x + 1;
  endfunction
  export "DPI-C" c_exported_func = function exported_func;

  task automatic exported_task(input int x);
    #1ns;
  endtask
  export "DPI-C" task exported_task;
endpackage

module m;
  function int module_func(input int x);
    return x - 1;
  endfunction
  export "DPI-C" c_module_func = function module_func;
endmodule

// MOORE: func.func @"p::exported_func"
// MOORE-SAME: circt.dpi.export = "c_exported_func"
// MOORE: moore.coroutine @"p::exported_task"
// MOORE-SAME: circt.dpi.export = "exported_task"
// MOORE: func.func @module_func
// MOORE-SAME: circt.dpi.export = "c_module_func"

// LLHD: func.func @"p::exported_func"
// LLHD-SAME: circt.dpi.export = "c_exported_func"
// LLHD: llhd.coroutine @"p::exported_task"
// LLHD-SAME: circt.dpi.export = "exported_task"
// LLHD: func.func @module_func
// LLHD-SAME: circt.dpi.export = "c_module_func"
