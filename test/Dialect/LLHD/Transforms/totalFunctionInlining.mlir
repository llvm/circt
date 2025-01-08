// RUN: circt-opt %s -llhd-function-elimination | FileCheck %s

// This test checks the presence of inlining into entities and processes
// and their general structure after inlining. It also checks that the functions
// are deleted by the elimination pass.
// Note: Only functions which can be reduced to one basic block can be inlined
// into entities.

// CHECK-NOT: func
func.func @complex(%flag : i1) -> i32 {
  cf.cond_br %flag, ^bb1, ^bb2
^bb1:
  %0 = hw.constant 5 : i32
  return %0 : i32
^bb2:
  %1 = hw.constant 7 : i32
  return %1 : i32
}

// CHECK-LABEL: @check_proc_inline
hw.module @check_proc_inline(inout %arg: i1, inout %out: i32) {
  llhd.process {
    %0 = llhd.prb %arg : !hw.inout<i1>
    %1 = func.call @complex(%0) : (i1) -> i32
    %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>
    llhd.drv %out, %1 after %time : !hw.inout<i32>
    llhd.yield
  }
  hw.output
}
