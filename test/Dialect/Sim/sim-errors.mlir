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
