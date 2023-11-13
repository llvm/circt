// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt --lower-seq-shiftreg %s | FileCheck %s --check-prefix=LO

// CHECK: %r0 = seq.shiftreg[3] %i, %clk, %ce  : i32
// CHECK: %myShiftReg = seq.shiftreg[3] sym @myShiftReg %i, %clk, %ce reset %rst, %c0_i32 powerOn %c0_i32 : i32

// LO:    %r0_sh1 = seq.compreg.ce sym @r0_sh1  %i, %clk, %ce : i32  
// LO:    %r0_sh2 = seq.compreg.ce sym @r0_sh2  %r0_sh1, %clk, %ce : i32  
// LO:    %r0_sh3 = seq.compreg.ce sym @r0_sh3  %r0_sh2, %clk, %ce : i32  
// LO:    %myShiftReg_sh1 = seq.compreg.ce sym @myShiftReg_sh1  %i, %clk, %ce reset %rst, %c0_i32 powerOn %c0_i32 : i32  
// LO:    %myShiftReg_sh2 = seq.compreg.ce sym @myShiftReg_sh2  %myShiftReg_sh1, %clk, %ce reset %rst, %c0_i32 powerOn %c0_i32 : i32  
// LO:    %myShiftReg_sh3 = seq.compreg.ce sym @myShiftReg_sh3  %myShiftReg_sh2, %clk, %ce reset %rst, %c0_i32 powerOn %c0_i32 : i32  
// LO:    hw.output %r0_sh3, %myShiftReg_sh3 : i32, i32

hw.module @top(in %clk: !seq.clock, in %rst: i1, in %ce: i1, in %i: i32, out out1 : i32, out out2 : i32) {
  %rv = hw.constant 0 : i32
  %r0 = seq.shiftreg [3] %i, %clk, %ce : i32
  %myShiftReg = seq.shiftreg [3] sym @myShiftReg %i, %clk, %ce reset %rst, %rv powerOn %rv  : i32
  hw.output %r0, %myShiftReg : i32, i32
}
