// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @A(%clk: !rtl.clk, %d: i1) -> (i1) {
    // Pass '%d' through a register, delaying the value by one cycle of '%clk'.
    %d_registered = rtl.reg %clk, %d : i1
    // Output values
    rtl.output %d_registered: i1
  }

}
