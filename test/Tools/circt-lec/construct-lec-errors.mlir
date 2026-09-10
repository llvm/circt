// RUN: circt-opt --construct-lec="first-module=modA0 second-module=modB0" --split-input-file --verify-diagnostics %s

// expected-error @below {{module named 'modA0' not found}}
builtin.module {
  hw.module @modA() { }
  hw.module @modB0() { }
}

// -----

// expected-error @below {{module's IO types don't match second modules: '!hw.modty<input in0 : i32, input in1 : i32>' vs '!hw.modty<input in0 : i32, input in1 : i32, output out : i32>'}}
hw.module @modA0(in %in0: i32, in %in1: i32) {
}

hw.module @modB0(in %in0: i32, in %in1: i32, out out: i32) {
  hw.output %in0 : i32
}

// -----

// expected-error @below {{module's IO types don't match second modules: '!hw.modty<input in0 : i32, output out : i32>' vs '!hw.modty<input in0 : i32, input in1 : i32, output out : i32>'}}
hw.module @modA0(in %in0: i32, out out: i32) {
  hw.output %in0 : i32
}

hw.module @modB0(in %in0: i32, in %in1: i32, out out: i32) {
  hw.output %in0 : i32
}
