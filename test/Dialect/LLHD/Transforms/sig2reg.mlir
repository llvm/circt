// RUN: circt-opt --llhd-sig2reg -cse %s | FileCheck %s

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

  llhd.drv %sig1, %in0 after %epsilon : i32
  // CHECK: [[DELAY:%.+]] = llhd.delay %in0 by <0ns, 1d, 0e> : i32
  llhd.drv %sig2, %in0 after %delta : i32
  llhd.drv %sig3, %in0 after %opaque_time : i32

  llhd.drv %sig5, %in0 after %epsilon if %cond : i32

  scf.if %cond {
    llhd.drv %sig6, %in0 after %epsilon : i32
  }
  
  llhd.drv %sig7, %in0 after %epsilon : i32
  llhd.drv %sig7, %in1 after %delta : i32

  %prb1 = llhd.prb %sig1 : i32
  %prb2 = llhd.prb %sig2 : i32
  // CHECK: [[PRB3:%.+]] = llhd.prb %sig3
  %prb3 = llhd.prb %sig3 : i32
  %prb4 = llhd.prb %sig4 : i32
  // CHECK: [[PRB5:%.+]] = llhd.prb %sig5
  %prb5 = llhd.prb %sig5 : i32
  // CHECK: [[PRB6:%.+]] = llhd.prb %sig6
  %prb6 = llhd.prb %sig6 : i32
  // CHECK: [[PRB7:%.+]] = llhd.prb %sig7
  %prb7 = llhd.prb %sig7 : i32

  // CHECK: hw.output %in0, [[DELAY]], [[PRB3]], %init, [[PRB5]], [[PRB6]], [[PRB7]] :
  hw.output %prb1, %prb2, %prb3, %prb4, %prb5, %prb6, %prb7 : i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: hw.module @aliasStatic
hw.module @aliasStatic(in %init : i4, in %in0 : i1, in %in1 : i1, out out: i4) {
  // CHECK-NEXT: [[C_2_I4:%.+]] = hw.constant -2 : i4
  // CHECK-NEXT: [[V0:%.+]] = comb.and %init, [[C_2_I4]] : i4
  // CHECK-NEXT: [[C0_I3:%.+]] = hw.constant 0 : i3
  // CHECK-NEXT: [[V1:%.+]] = comb.concat [[C0_I3]], %in1 : i3, i1
  // CHECK-NEXT: [[V2:%.+]] = comb.or [[V1]], [[V0]] : i4
  // CHECK-NEXT: [[C1_I4:%.+]] = hw.constant 1 : i4
  // CHECK-NEXT: [[C_3_I4:%.+]] = hw.constant -3 : i4
  // CHECK-NEXT: [[V3:%.+]] = comb.and [[V2]], [[C_3_I4]] : i4
  // CHECK-NEXT: [[V4:%.+]] = comb.concat [[C0_I3]], %in0 : i3, i1
  // CHECK-NEXT: [[V5:%.+]] = comb.shl [[V4]], [[C1_I4]] : i4
  // CHECK-NEXT: [[V6:%.+]] = comb.or [[V5]], [[V3]] : i4
  // CHECK-NEXT: [[C2_I4:%.+]] = hw.constant 2 : i4
  // CHECK-NEXT: [[C_5_I4:%.+]] = hw.constant -5 : i4
  // CHECK-NEXT: [[V7:%.+]] = comb.and [[V6]], [[C_5_I4]] : i4
  // CHECK-NEXT: [[V8:%.+]] = comb.shl [[V1]], [[C2_I4]] : i4
  // CHECK-NEXT: [[V9:%.+]] = comb.or [[V8]], [[V7]] : i4
  // CHECK-NEXT: hw.output [[V9]] : i4

  %0 = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %c1_c2 = hw.constant 1 : i2
  %c0_c2 = hw.constant 0 : i2
  %false = hw.constant false
  %out = llhd.sig %init : i4
  %3 = llhd.sig.extract %out from %c1_c2 : <i4> -> <i2>
  %6 = llhd.sig.extract %3 from %false : <i2> -> <i1>
  %7 = llhd.sig.extract %3 from %true : <i2> -> <i1>
  llhd.drv %6, %in0 after %0 : i1
  llhd.drv %7, %in1 after %0 : i1
  %4 = llhd.sig.extract %out from %c0_c2 : <i4> -> <i1>
  llhd.drv %4, %in1 after %0 : i1
  %5 = llhd.prb %out : i4
  hw.output %5 : i4
}

// CHECK-LABEL: hw.module @aliasDynamicSuccess
hw.module @aliasDynamicSuccess(in %init : i8, in %in0 : i1, in %in1 : i1, in %idx0 : i2, in %idx1 : i1, out out: i8) {
  // CHECK-NEXT: [[C1_I8:%.+]] = hw.constant 1 : i8
  // CHECK-NEXT: [[C0_I8:%.+]] = hw.constant 0 : i8
  // CHECK-NEXT: [[C0_I6:%.+]] = hw.constant 0 : i6
  // CHECK-NEXT: [[V0:%.+]] = comb.concat [[C0_I6]], %idx0 : i6, i2
  // CHECK-NEXT: [[V1:%.+]] = comb.add [[V0]], [[C0_I8]] : i8
  // CHECK-NEXT: [[C0_I7:%.+]] = hw.constant 0 : i7
  // CHECK-NEXT: [[V2:%.+]] = comb.concat [[C0_I7]], %idx1 : i7, i1
  // CHECK-NEXT: [[V3:%.+]] = comb.add [[V1]], [[V2]] : i8
  // CHECK-NEXT: [[V4:%.+]] = comb.shl [[C1_I8]], [[V3]] : i8
  // CHECK-NEXT: [[C_1_I8:%.+]] = hw.constant -1 : i8
  // CHECK-NEXT: [[V5:%.+]] = comb.xor [[V4]], [[C_1_I8]] : i8
  // CHECK-NEXT: [[V6:%.+]] = comb.and %init, [[V5]] : i8
  // CHECK-NEXT: [[V7:%.+]] = comb.concat [[C0_I7]], %in0 : i7, i1
  // CHECK-NEXT: [[V8:%.+]] = comb.shl [[V7]], [[V3]] : i8
  // CHECK-NEXT: [[V9:%.+]] = comb.or [[V8]], [[V6]] : i8
  // CHECK-NEXT: [[C4_I8:%.+]] = hw.constant 4 : i8
  // CHECK-NEXT: [[V10:%.+]] = comb.add [[V0]], [[C4_I8]] : i8
  // CHECK-NEXT: [[V11:%.+]] = comb.shl [[C1_I8]], [[V10]] : i8
  // CHECK-NEXT: [[V12:%.+]] = comb.xor [[V11]], [[C_1_I8]] : i8
  // CHECK-NEXT: [[V13:%.+]] = comb.and [[V9]], [[V12]] : i8
  // CHECK-NEXT: [[V14:%.+]] = comb.concat [[C0_I7]], %in1 : i7, i1
  // CHECK-NEXT: [[V15:%.+]] = comb.shl [[V14]], [[V10]] : i8
  // CHECK-NEXT: [[V16:%.+]] = comb.or [[V15]], [[V13]] : i8
  // CHECK-NEXT: hw.output [[V16]] : i8

  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c4_c3 = hw.constant 4 : i3
  %c0_c3 = hw.constant 0 : i3
  %out = llhd.sig %init : i8
  %3 = llhd.sig.extract %out from %c0_c3 : <i8> -> <i4>
  %4 = llhd.sig.extract %out from %c4_c3 : <i8> -> <i4>
  %5 = llhd.sig.extract %3 from %idx0 : <i4> -> <i2>
  %6 = llhd.sig.extract %5 from %idx1 : <i2> -> <i1>
  %7 = llhd.sig.extract %4 from %idx0 : <i4> -> <i1>
  llhd.drv %6, %in0 after %0 : i1
  llhd.drv %7, %in1 after %0 : i1
  %8 = llhd.prb %out : i8
  hw.output %8 : i8
}

// CHECK-LABEL: hw.module @aliasDynamicFailure
hw.module @aliasDynamicFailure(in %init : i4, in %in0 : i1, in %in1 : i1, in %idx0 : i1, in %idx1 : i1, out out: i4) {
  // CHECK-NEXT: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: [[C_2_I2:%.+]] = hw.constant -2 : i2
  // CHECK-NEXT: [[C0_I2:%.+]] = hw.constant 0 : i2
  // CHECK-NEXT: [[OUT:%.+]] = llhd.sig %init : i4
  // CHECK-NEXT: [[V1:%.+]] = llhd.sig.extract [[OUT]] from [[C0_I2]] : <i4> -> <i2>
  // CHECK-NEXT: [[V2:%.+]] = llhd.sig.extract [[OUT]] from [[C_2_I2]] : <i4> -> <i2>
  // CHECK-NEXT: [[V3:%.+]] = llhd.sig.extract [[V1]] from %idx0 : <i2> -> <i1>
  // CHECK-NEXT: [[V4:%.+]] = llhd.sig.extract [[V2]] from %idx0 : <i2> -> <i1>
  // CHECK-NEXT: [[V5:%.+]] = llhd.sig.extract [[V2]] from %idx1 : <i2> -> <i1>
  // CHECK-NEXT: llhd.drv [[V3]], %in0 after [[TIME]] : i1
  // CHECK-NEXT: llhd.drv [[V4]], %in1 after [[TIME]] : i1
  // CHECK-NEXT: llhd.drv [[V5]], %in0 after [[TIME]] : i1
  // CHECK-NEXT: [[V6:%.+]] = llhd.prb [[OUT]] : i4
  // CHECK-NEXT: hw.output [[V6]] : i4

  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c2_c2 = hw.constant 2 : i2
  %c0_c2 = hw.constant 0 : i2
  %out = llhd.sig %init : i4
  %3 = llhd.sig.extract %out from %c0_c2 : <i4> -> <i2>
  %4 = llhd.sig.extract %out from %c2_c2 : <i4> -> <i2>
  %5 = llhd.sig.extract %3 from %idx0 : <i2> -> <i1>
  %6 = llhd.sig.extract %4 from %idx0 : <i2> -> <i1>
  %7 = llhd.sig.extract %4 from %idx1 : <i2> -> <i1>
  llhd.drv %5, %in0 after %0 : i1
  llhd.drv %6, %in1 after %0 : i1
  llhd.drv %7, %in0 after %0 : i1
  %8 = llhd.prb %out : i4
  hw.output %8 : i4
}

// CHECK-LABEL: @RemoveDriveOnlySignals
hw.module @RemoveDriveOnlySignals(in %d: i42, in %e: i1) {
  %0 = hw.constant 0 : i42
  %1 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NOT: llhd.sig
  %a = llhd.sig %0 : i42
  %b = llhd.sig %0 : i42
  // CHECK-NOT: llhd.drv
  llhd.drv %a, %d after %1 : i42
  llhd.drv %b, %d after %1 if %e : i42
  // CHECK: hw.output
}
