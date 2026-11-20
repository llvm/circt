// RUN: circt-opt %s --convert-to-arcs --verify-diagnostics --split-input-file

// Replacing signal probes with feedback arguments does not break the direct
// SSA cycle between %sum and %next. Reject it instead of producing a body with
// invalid dominance.
hw.module @SignalFixedPointRemainingCycle(out result: i8) {
  %zero = hw.constant 0 : i8
  %one = hw.constant 1 : i8
  %delay = llhd.constant_time <0ns, 0d>
  // expected-error @below {{'arc.fixed_point' op contains an SSA dependency cycle after resolving signal feedback}}
  %signal = llhd.sig %zero : <i8>
  %previous = llhd.prb %signal : i8
  %sum = comb.add %previous, %next : i8
  %next = comb.and %sum, %one : i8
  llhd.drv %signal, %next after %delay : i8
  hw.output %previous : i8
}

// -----

// Immediate assignments must also undergo the dominance check after signal
// feedback is replaced by region arguments.
hw.module @AssignmentFixedPointRemainingCycle(out result: i8) {
  %zero = hw.constant 0 : i8
  %one = hw.constant 1 : i8
  // expected-error @below {{'arc.fixed_point' op contains an SSA dependency cycle after resolving signal feedback}}
  %signal = llhd.sig %zero : <i8>
  %previous = llhd.prb %signal : i8
  %sum = comb.add %previous, %next : i8
  %next = comb.and %sum, %one : i8
  llhd.assign %signal, %next : i8
  hw.output %previous : i8
}

// -----

// A cycle containing a drive cannot also have an external immediate writer.
hw.module @SignalFixedPointExternalAssignment(in %input: i8) {
  %zero = hw.constant 0 : i8
  %delay = llhd.constant_time <0ns, 0d>
  %signal = llhd.sig %zero : <i8>
  %previous = llhd.prb %signal : i8
  llhd.drv %signal, %previous after %delay : i8
  // expected-error @below {{signals in fixed-point regions cannot have external drives}}
  llhd.assign %signal, %input : i8
}

// -----

// Supporting immediate assignments must not relax the delay restriction on
// drives participating in fixed-point evaluation.
hw.module @SignalFixedPointDelayedDrive() {
  %zero = hw.constant 0 : i8
  %delay = llhd.constant_time <1ns, 0d>
  %signal = llhd.sig %zero : <i8>
  %previous = llhd.prb %signal : i8
  // expected-error @below {{fixed-point regions support only zero-time, zero-delta drives}}
  llhd.drv %signal, %previous after %delay : i8
}
