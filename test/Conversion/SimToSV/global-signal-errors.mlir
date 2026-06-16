// RUN: circt-opt --lower-sim-to-sv --verify-diagnostics %s

// expected-error @below {{cannot lower default body of 'sim.global_signal' to a SystemVerilog macro expression}}
sim.global_signal @unsupported : i8 {
  %value = hw.constant 42 : i8
  %cast = hw.bitcast %value : (i8) -> i8
  sim.yield %cast : i8
}
