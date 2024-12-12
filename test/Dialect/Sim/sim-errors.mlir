// RUN: circt-opt %s --split-input-file --verify-diagnostics --canonicalize

hw.module @fmt_infinite_concat_verify() {
  %lp = sim.fmt.lit ", {"
  %rp = sim.fmt.lit "}"
  // expected-error @below {{op is infinitely recursive.}}
  %ordinal = sim.fmt.concat (%ordinal, %lp, %ordinal, %rp)
}
// -----

hw.module @fmt_infinite_concat_canonicalize(in %val : i8, out res: !sim.fstring) {
  %c = sim.fmt.char %val : i8
  %0 = sim.fmt.lit "Here we go round the"
  %1 = sim.fmt.lit "prickly pear"
  // expected-warning @below {{Cyclic concatenation detected.}}
  %2 = sim.fmt.concat (%1, %c, %4)
  // expected-warning @below {{Cyclic concatenation detected.}}
  %3 = sim.fmt.concat (%1, %c, %1, %c, %2, %c)
  // expected-warning @below {{Cyclic concatenation detected.}}
  %4 = sim.fmt.concat (%0, %c, %3)
  %5 = sim.fmt.lit "At five o'clock in the morning."
  // expected-warning @below {{Cyclic concatenation detected.}}
  %cat = sim.fmt.concat (%4, %c, %5)
  hw.output %cat : !sim.fstring
}

// -----

hw.module @proc_print_hw() {
  %lit = sim.fmt.lit "Nope"
  // expected-error @below {{must be within a procedural region.}}
  sim.proc.print %lit
}

// -----

sv.macro.decl @SOMEMACRO
hw.module @proc_print_sv() {
  %lit = sim.fmt.lit "Nope"
  sv.ifdef  @SOMEMACRO {
    // expected-error @below {{must be within a procedural region.}}
    sim.proc.print %lit
  }
}

// -----

hw.module.extern @non_func(out arg0: i1, in %arg1: i1, out arg2: i1)

hw.module @dpi_call(in %clock : !seq.clock, in %in: i1) {
  // expected-error @below {{callee must be 'sim.dpi.func' or 'func.func' but got 'hw.module.extern'}}
  %0, %1 = sim.func.dpi.call @non_func(%in) : (i1) -> (i1, i1)
}

// -----

hw.module @not_enough_triggers(in %in : !sim.trigger.edge<posedge>) {
  // expected-error @below {{operation defines 1 results but was provided 2 to bind}}
  %res:2 = sim.trigger_sequence %in, 1 : !sim.trigger.edge<posedge>
}

// -----

hw.module @recursive_trigger() {
  // expected-warning @below {{Recursive trigger sequence}}
  %res = sim.trigger_sequence %res, 1 : !sim.trigger.edge<posedge>
}

// -----

hw.module @missing_tieoffs(in %trig : !sim.trigger.edge<posedge>) {
  // expected-error @below {{Tie-off constants must be provided for all results}}
  %res = sim.triggered () on (%trig : !sim.trigger.edge<posedge>) {
    %cst = hw.constant 0 : i2
    sim.yield_seq %cst : i2
  } : () -> i2
}

// -----

hw.module @wrong_tieoff(in %trig : !sim.trigger.edge<posedge>) {
  // expected-error @below {{Tie-off type does not match for result at index 0}}
  %res = sim.triggered () on (%trig : !sim.trigger.edge<posedge>) tieoff [0 : i1] {
    %cst = hw.constant 0 : i2
    sim.yield_seq %cst : i2
  } : () -> i2
}

// -----

hw.module @too_many_tieoffs(in %trig : !sim.trigger.edge<posedge>) {
  // expected-error @below {{Number of tie-off constants does not match number of results}}
  %res = sim.triggered () on (%trig : !sim.trigger.edge<posedge>) tieoff [0 : i2, 0 : i2] {
    %cst = hw.constant 0 : i2
    sim.yield_seq %cst : i2
  } : () -> i2
}
