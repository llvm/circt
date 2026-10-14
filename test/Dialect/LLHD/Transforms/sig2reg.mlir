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
  %sig1 = llhd.sig %init : <i32>
  // Promoted with delay op
  %sig2 = llhd.sig %init : <i32>
  // Not promoted because time is opaque, can support if the delay op takes a
  // time value instead of attribute
  %sig3 = llhd.sig %init : <i32>
  // Promoted to %init because no drive present
  %sig4 = llhd.sig %init : <i32>
  // Not promoted because of drive condition
  %sig5 = llhd.sig %init : <i32>
  // Not promoted because a user is in a nested region
  %sig6 = llhd.sig %init : <i32>
  // Not promoted because of multiple drivers
  %sig7 = llhd.sig %init : <i32>

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
  %out = llhd.sig %init : <i4>
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

// The nested dynamic extracts have a conservative maximum offset of 4. Start
// the second slice at bit 5 so the computed write intervals are disjoint.
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
  // CHECK-NEXT: [[C5_I8:%.+]] = hw.constant 5 : i8
  // CHECK-NEXT: [[V10:%.+]] = comb.add [[V0]], [[C5_I8]] : i8
  // CHECK-NEXT: [[V11:%.+]] = comb.shl [[C1_I8]], [[V10]] : i8
  // CHECK-NEXT: [[V12:%.+]] = comb.xor [[V11]], [[C_1_I8]] : i8
  // CHECK-NEXT: [[V13:%.+]] = comb.and [[V9]], [[V12]] : i8
  // CHECK-NEXT: [[V14:%.+]] = comb.concat [[C0_I7]], %in1 : i7, i1
  // CHECK-NEXT: [[V15:%.+]] = comb.shl [[V14]], [[V10]] : i8
  // CHECK-NEXT: [[V16:%.+]] = comb.or [[V15]], [[V13]] : i8
  // CHECK-NEXT: hw.output [[V16]] : i8

  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c5_c3 = hw.constant 5 : i3
  %c0_c3 = hw.constant 0 : i3
  %out = llhd.sig %init : <i8>
  %3 = llhd.sig.extract %out from %c0_c3 : <i8> -> <i4>
  %4 = llhd.sig.extract %out from %c5_c3 : <i8> -> <i3>
  %5 = llhd.sig.extract %3 from %idx0 : <i4> -> <i2>
  %6 = llhd.sig.extract %5 from %idx1 : <i2> -> <i1>
  %7 = llhd.sig.extract %4 from %idx0 : <i3> -> <i1>
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
  // CHECK-NEXT: [[OUT:%.+]] = llhd.sig %init : <i4>
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
  %out = llhd.sig %init : <i4>
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
  %a = llhd.sig %0 : <i42>
  %b = llhd.sig %0 : <i42>
  // CHECK-NOT: llhd.drv
  llhd.drv %a, %d after %1 : i42
  llhd.drv %b, %d after %1 if %e : i42
  // CHECK: hw.output
}

// CHECK-LABEL: hw.module @UninitializedFullDrive
hw.module @UninitializedFullDrive(in %value : i4, out out : i4) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NOT: llhd.sig
  %sig = llhd.sig : <i4>
  llhd.drv %sig, %value after %time : i4
  %read = llhd.prb %sig : i4
  // CHECK: hw.output %value : i4
  hw.output %read : i4
}

// The drive writes bits 5..1 back unchanged and sets bit 0. The unchanged part
// is dropped, so the signal is promoted.
// CHECK-LABEL: hw.module @identityConcatDrive
hw.module @identityConcatDrive(in %init : i6, out out : i6) {
  // CHECK: [[MASK:%.+]] = hw.constant -2 : i6
  // CHECK: [[AND:%.+]] = comb.and %init, [[MASK]] : i6
  // CHECK: [[ONE:%.+]] = hw.constant 1 : i6
  // CHECK: [[OR:%.+]] = comb.or [[AND]], [[ONE]] : i6
  // CHECK: hw.output [[OR]] : i6
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %out = llhd.sig %init : <i6>
  %1 = llhd.prb %out : i6
  %2 = comb.extract %1 from 1 : (i6) -> i5
  %3 = comb.concat %2, %true : i5, i1
  llhd.drv %out, %3 after %0 : i6
  hw.output %1 : i6
}

// Same, but the unchanged part sits inside a nested concat.
// CHECK-LABEL: hw.module @nestedIdentityConcatDrive
hw.module @nestedIdentityConcatDrive(in %init : i6, out out : i6) {
  // CHECK: [[MASK0:%.+]] = hw.constant -2 : i6
  // CHECK: [[AND0:%.+]] = comb.and %init, [[MASK0]] : i6
  // CHECK: [[MASK1:%.+]] = hw.constant -3 : i6
  // CHECK: [[AND1:%.+]] = comb.and [[AND0]], [[MASK1]] : i6
  // CHECK: [[TWO:%.+]] = hw.constant 2 : i6
  // CHECK: [[OR:%.+]] = comb.or [[AND1]], [[TWO]] : i6
  // CHECK: hw.output [[OR]] : i6
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %false = hw.constant false
  %out = llhd.sig %init : <i6>
  %1 = llhd.prb %out : i6
  %2 = comb.extract %1 from 2 : (i6) -> i4
  %3 = comb.concat %2, %true : i4, i1
  %4 = comb.concat %3, %false : i5, i1
  llhd.drv %out, %4 after %0 : i6
  hw.output %1 : i6
}

// Not promoted: `out[in] = 1` with a variable index becomes a masked
// read-modify-write, which reads the signal it drives. Only concats can be
// split, so nothing can be dropped here.
// CHECK-LABEL: hw.module @maskedSelfReferentialDrive
hw.module @maskedSelfReferentialDrive(in %init : i6, in %in : i1, out out : i6) {
  // CHECK-NEXT: [[TIME:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: [[C0_I5:%.+]] = hw.constant 0 : i5
  // CHECK-NEXT: [[C1_I6:%.+]] = hw.constant 1 : i6
  // CHECK-NEXT: [[C_1_I6:%.+]] = hw.constant -1 : i6
  // CHECK-NEXT: [[OUT:%.+]] = llhd.sig %init : <i6>
  // CHECK-NEXT: [[PRB:%.+]] = llhd.prb [[OUT]] : i6
  // CHECK-NEXT: [[IDX:%.+]] = comb.concat [[C0_I5]], %in : i5, i1
  // CHECK-NEXT: [[MASK:%.+]] = comb.shl [[C1_I6]], [[IDX]] : i6
  // CHECK-NEXT: [[INV:%.+]] = comb.xor [[MASK]], [[C_1_I6]] : i6
  // CHECK-NEXT: [[AND:%.+]] = comb.and [[PRB]], [[INV]] : i6
  // CHECK-NEXT: [[OR:%.+]] = comb.or [[AND]], [[MASK]] : i6
  // CHECK-NEXT: llhd.drv [[OUT]], [[OR]] after [[TIME]] : i6
  // CHECK-NEXT: hw.output [[PRB]] : i6
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i5 = hw.constant 0 : i5
  %c1_i6 = hw.constant 1 : i6
  %c-1_i6 = hw.constant -1 : i6
  %out = llhd.sig %init : <i6>
  %1 = llhd.prb %out : i6
  %2 = comb.concat %c0_i5, %in : i5, i1
  %3 = comb.shl %c1_i6, %2 : i6
  %4 = comb.xor %3, %c-1_i6 : i6
  %5 = comb.and %1, %4 : i6
  %6 = comb.or %5, %3 : i6
  llhd.drv %out, %6 after %0 : i6
  hw.output %1 : i6
}

// A delayed drive captures bits 5..1 at the time of the probe. They cannot be
// dropped as identity fragments: another drive updates part of them before this
// one takes effect, so the delayed drive must restore the captured value.
// CHECK-LABEL: hw.module @delayedIdentityConcatDrive
// CHECK: [[OUT:%.+]] = llhd.sig %init : <i6>
// CHECK: llhd.drv [[OUT]],
hw.module @delayedIdentityConcatDrive(in %init : i6, out out : i6) {
  %time1 = llhd.constant_time <1ns, 0d, 0e>
  %time2 = llhd.constant_time <2ns, 0d, 0e>
  %c0_i3 = hw.constant 0 : i3
  %c0_i5 = hw.constant 0 : i5
  %true = hw.constant true
  %out = llhd.sig %init : <i6>
  %0 = llhd.prb %out : i6
  %1 = comb.extract %0 from 1 : (i6) -> i5
  %2 = comb.concat %1, %true : i5, i1
  llhd.drv %out, %2 after %time2 : i6
  %3 = llhd.sig.extract %out from %c0_i3 : <i6> -> <i5>
  llhd.drv %3, %c0_i5 after %time1 : i5
  hw.output %0 : i6
}

// A delayed drive that reads the signal it drives is still promoted: `promote`
// wraps the driven value in an `llhd.delay`, so the feedback runs through the
// delay instead of becoming a combinational cycle.
// CHECK-LABEL: hw.module @delayedSelfDrive
// CHECK-NEXT: [[ONE:%.+]] = hw.constant 1 : i8
// CHECK-NEXT: [[DELAY:%.+]] = llhd.delay [[ADD:%.+]] by <1ns, 0d, 0e> : i8
// CHECK-NEXT: [[ADD]] = comb.add [[DELAY]], [[ONE]] : i8
// CHECK-NEXT: hw.output [[DELAY]] : i8
hw.module @delayedSelfDrive(in %init : i8, out out : i8) {
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c1_i8 = hw.constant 1 : i8
  %sig = llhd.sig %init : <i8>
  %0 = llhd.prb %sig : i8
  %1 = comb.add %0, %c1_i8 : i8
  llhd.drv %sig, %1 after %time : i8
  hw.output %0 : i8
}

// Splitting the immediate concat must not hide overlapping drives. Its one-bit
// intervals overlap both endpoints of the delayed two-bit interval.
// CHECK-LABEL: hw.module @overlappingConcatDrivers
// CHECK-NEXT: [[EPSILON:%.+]] = llhd.constant_time <0ns, 0d, 1e>
// CHECK-NEXT: [[LATER:%.+]] = llhd.constant_time <1ns, 0d, 0e>
// CHECK-NEXT: [[SIG:%.+]] = llhd.sig %init : <i2>
// CHECK-NEXT: [[FIRST:%.+]] = comb.concat %a, %b : i1, i1
// CHECK-NEXT: [[SECOND:%.+]] = comb.concat %c, %d : i1, i1
// CHECK-NEXT: llhd.drv [[SIG]], [[FIRST]] after [[EPSILON]] : i2
// CHECK-NEXT: llhd.drv [[SIG]], [[SECOND]] after [[LATER]] : i2
// CHECK-NEXT: [[PROBE:%.+]] = llhd.prb [[SIG]] : i2
// CHECK-NEXT: hw.output [[PROBE]] : i2
hw.module @overlappingConcatDrivers(in %init : i2, in %a : i1, in %b : i1, in %c : i1, in %d : i1, out out : i2) {
  %epsilon = llhd.constant_time <0ns, 0d, 1e>
  %later = llhd.constant_time <1ns, 0d, 0e>
  %sig = llhd.sig %init : <i2>
  %first = comb.concat %a, %b : i1, i1
  %second = comb.concat %c, %d : i1, i1
  llhd.drv %sig, %first after %epsilon : i2
  llhd.drv %sig, %second after %later : i2
  %probe = llhd.prb %sig : i2
  hw.output %probe : i2
}

// A drive fed by a register is not a combinational loop: the register breaks
// the feedback in time, so the signal is promoted and the register ends up
// driving itself.
// CHECK-LABEL: hw.module @clockedFeedback
// CHECK-NEXT: [[ONE:%.+]] = hw.constant 1 : i8
// CHECK-NEXT: [[ADD:%.+]] = comb.add [[REG:%.+]], [[ONE]] : i8
// CHECK-NEXT: [[CLK:%.+]] = seq.to_clock %clk
// CHECK-NEXT: [[REG]] = seq.firreg [[ADD]] clock [[CLK]] : i8
// CHECK-NEXT: hw.output [[REG]] : i8
hw.module @clockedFeedback(in %clk : i1, out count : i8) {
  %c1_i8 = hw.constant 1 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %false = hw.constant false
  %clk_0 = llhd.sig name "clk" %false : <i1>
  %1 = llhd.prb %clk_0 : i1
  %count = llhd.sig %c0_i8 : <i8>
  %2 = comb.add %4, %c1_i8 : i8
  %3 = seq.to_clock %1
  %count_1 = seq.firreg %2 clock %3 {name = "count"} : i8
  llhd.drv %count, %count_1 after %0 : i8
  llhd.drv %clk_0, %clk after %0 : i1
  %4 = llhd.prb %count : i8
  hw.output %4 : i8
}

hw.module @feedbackReg(in %clk : !seq.clock, in %d : i8, out q : i8) {
  %r = seq.firreg %d clock %clk : i8
  hw.output %r : i8
}

// Same feedback, but the register sits inside a submodule. The pass cannot see
// into the instance, so it does not treat it as a combinational path: the
// signal is promoted and the instance ends up feeding itself.
// CHECK-LABEL: hw.module @instanceFeedback
// CHECK-NEXT: [[ONE:%.+]] = hw.constant 1 : i8
// CHECK-NEXT: [[Q:%.+]] = hw.instance "u" @feedbackReg(clk: %clk: !seq.clock, d: [[ADD:%.+]]: i8) -> (q: i8)
// CHECK-NEXT: [[ADD]] = comb.add [[Q]], [[ONE]] : i8
// CHECK-NEXT: hw.output [[Q]] : i8
hw.module @instanceFeedback(in %clk : !seq.clock, out out : i8) {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %s = llhd.sig %c0_i8 : <i8>
  %q = hw.instance "u" @feedbackReg(clk: %clk: !seq.clock, d: %next: i8) -> (q: i8)
  %next = comb.add %p, %c1_i8 : i8
  llhd.drv %s, %q after %eps : i8
  %p = llhd.prb %s : i8
  hw.output %p : i8
}
