// RUN: circt-opt --llhd-sig2reg %s | FileCheck %s

func.func @getTime() -> !llhd.time {
  %time = llhd.constant_time <1ns, 0d, 0e>
  return %time : !llhd.time
}

hw.module @basic(in %init : i32, in %cond : i1, in %in0 : i32, in %in1 : i32, out prb1 : i32, out prb2 : i32, out prb3 : i32, out prb4 : i32, out prb5 : i32, out prb6 : i32, out prb7 : i32) {
  %opaque_time = func.call @getTime() : () -> !llhd.time
  %epsilon = llhd.constant_time <0ns, 0d, 1e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  // Promoted without delay op
  %sig1 = llhd.sig %init : i32
  // Promoted with delay op
  %sig2 = llhd.sig %init : i32
  // Not promoted because time is opaque, can support if the delay op takes a
  // time value instead of attribute
  %sig3 = llhd.sig %init : i32
  // Promoted to %init because no drive present
  %sig4 = llhd.sig %init : i32
  // Not promoted because of drive condition
  %sig5 = llhd.sig %init : i32
  // Not promoted because a user is in a nested region
  %sig6 = llhd.sig %init : i32
  // Not promoted because of multiple drivers
  %sig7 = llhd.sig %init : i32

  llhd.drv %sig1, %in0 after %epsilon : !hw.inout<i32>
  // CHECK: [[DELAY:%.+]] = llhd.delay %in0 by <0ns, 1d, 0e> : i32
  llhd.drv %sig2, %in0 after %delta : !hw.inout<i32>
  llhd.drv %sig3, %in0 after %opaque_time : !hw.inout<i32>

  llhd.drv %sig5, %in0 after %epsilon if %cond : !hw.inout<i32>

  scf.if %cond {
    llhd.drv %sig6, %in0 after %epsilon : !hw.inout<i32>
  }
  
  llhd.drv %sig7, %in0 after %epsilon : !hw.inout<i32>
  llhd.drv %sig7, %in1 after %delta : !hw.inout<i32>

  %prb1 = llhd.prb %sig1 : !hw.inout<i32>
  %prb2 = llhd.prb %sig2 : !hw.inout<i32>
  // CHECK: [[PRB3:%.+]] = llhd.prb %sig3
  %prb3 = llhd.prb %sig3 : !hw.inout<i32>
  %prb4 = llhd.prb %sig4 : !hw.inout<i32>
  // CHECK: [[PRB5:%.+]] = llhd.prb %sig5
  %prb5 = llhd.prb %sig5 : !hw.inout<i32>
  // CHECK: [[PRB6:%.+]] = llhd.prb %sig6
  %prb6 = llhd.prb %sig6 : !hw.inout<i32>
  // CHECK: [[PRB7:%.+]] = llhd.prb %sig7
  %prb7 = llhd.prb %sig7 : !hw.inout<i32>

  // CHECK: hw.output %in0, [[DELAY]], [[PRB3]], %init, [[PRB5]], [[PRB6]], [[PRB7]] :
  hw.output %prb1, %prb2, %prb3, %prb4, %prb5, %prb6, %prb7 : i32, i32, i32, i32, i32, i32, i32
}
