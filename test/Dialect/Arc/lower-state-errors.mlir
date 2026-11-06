// RUN: circt-opt %s --arc-lower-state --verify-diagnostics --split-input-file

hw.module @CombLoop(in %a: i42, out z: i42) {
  // expected-error @below {{'comb.add' op is on a combinational loop}}
  // expected-remark @below {{computing new phase here}}
  %0 = comb.add %a, %1 : i42
  // expected-remark @below {{computing new phase here}}
  %1 = comb.mul %a, %0 : i42
  // expected-remark @below {{computing new phase here}}
  hw.output %0 : i42
}

// -----

// expected-error @+1 {{Failed to remove external module because it is still referenced/instantiated}}
hw.module.extern @myModule()

hw.instance "alligator" @myModule() -> ()

// -----

// Reject self-dependencies through signal assignments,
// even when the feedback would not change the signal's value.
hw.module @LLHDSignalSelfLoop() {
  %init = hw.constant 0 : i8
  %zero = llhd.constant_time <0fs, 0d, 0e>
  %signal = llhd.sig %init : i8
  %read = llhd.prb %signal : i8
  // expected-error @below {{'llhd.drv' op is on a combinational loop}}
  // expected-remark @below {{computing new phase here}}
  llhd.drv %signal, %read after %zero : i8
}

// -----

hw.module @LLHDSignalDependencyLoop() {
  %init = hw.constant 0 : i8
  %zero = llhd.constant_time <0fs, 0d, 0e>
  %a = llhd.sig %init : i8
  %b = llhd.sig %init : i8
  %read_a = llhd.prb %a : i8
  %read_b = llhd.prb %b : i8
  // expected-error @below {{'llhd.drv' op is on a combinational loop}}
  // expected-remark @below {{computing new phase here}}
  llhd.drv %a, %read_b after %zero : i8
  // expected-remark @below {{computing new phase here}}
  llhd.drv %b, %read_a after %zero : i8
}

// -----

arc.coroutine.define @SignalLoop(%arg: i8) -> (i8, i1, i64) {
  %false = hw.constant false
  %never = hw.constant -1 : i64
  arc.coroutine.halt %arg, %false, %never : i8, i1, i64
}

hw.module @LLHDSignalCoroutineLoop() {
  %init = hw.constant 0 : i8
  %zero = llhd.constant_time <0fs, 0d, 0e>
  %signal = llhd.sig %init : i8
  %read = llhd.prb %signal : i8
  // expected-error @below {{'arc.coroutine.instance' op is on a combinational loop}}
  // expected-remark @below {{computing new phase here}}
  %value = arc.coroutine.instance @SignalLoop(%read) sensitive [true] : (i8) -> i8
  // expected-remark @below {{computing new phase here}}
  llhd.drv %signal, %value after %zero : i8
}
